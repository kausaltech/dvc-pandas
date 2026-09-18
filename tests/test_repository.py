from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import dvc.repo
import polars as pl
import pytest
from dvc_pandas import Dataset, DatasetMeta, Repository
from dvc_pandas.dvc import set_dvc_file_metadata
from polars.testing import assert_frame_equal

from .helpers import configure_git, git

if TYPE_CHECKING:
    from collections.abc import Iterator


class Upstream:
    def __init__(self, path: Path):
        self.bare_repo_dir = path / 'upstream.git'
        self.workspace_dir = path / 'workspace'
        self.cloud_dir = path / 'cloud'
        git(path, 'init', '--bare', '-b', 'main', str(self.bare_repo_dir))
        git(path, 'clone', str(self.bare_repo_dir), str(self.workspace_dir))
        configure_git(self.workspace_dir)
        self.cloud_dir.mkdir()
        with dvc.repo.Repo.init(str(self.workspace_dir)) as repo, repo.config.edit() as config:
            config['remote']['cloud'] = {'url': str(self.cloud_dir)}
            config['core']['remote'] = 'cloud'
        git(self.workspace_dir, 'add', '.')
        git(self.workspace_dir, 'commit', '-m', 'Initialize DVC')
        git(self.workspace_dir, 'push', '-u', 'origin', 'main')

    def add_dataset(self, dataset: Dataset) -> None:
        parquet = self.workspace_dir / f'{dataset.identifier}.parquet'
        parquet.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(parquet)
        with dvc.repo.Repo(str(self.workspace_dir)) as repo:
            repo.add(str(parquet))
            set_dvc_file_metadata(parquet.with_suffix('.parquet.dvc'), dataset.dvc_metadata)
            repo.push()
        git(self.workspace_dir, 'add', '.')
        git(self.workspace_dir, 'commit', '-m', f'Add {dataset.identifier}')
        git(self.workspace_dir, 'push')

    def clear_cloud(self) -> None:
        shutil.rmtree(self.cloud_dir)
        self.cloud_dir.mkdir()

    def contains(self, path: str) -> bool:
        return path in git(self.bare_repo_dir, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines()


@pytest.fixture
def upstream(tmp_path: Path, dataset: Dataset) -> Upstream:
    upstream = Upstream(tmp_path)
    upstream.add_dataset(dataset)
    return upstream


@pytest.fixture
def repo(upstream: Upstream, tmp_path: Path) -> Iterator[Repository]:
    repo = Repository(str(upstream.bare_repo_dir), cache_root=str(tmp_path / 'cache'))
    configure_git(repo.repo_dir)
    try:
        yield repo
    finally:
        if repo._dvc_repo is not None:
            repo.dvc_repo.close()
        repo.git_repo.free()


@pytest.fixture
def uncached_dataset(upstream: Upstream, repo: Repository) -> Dataset:
    # Initialize the consumer clone before changing upstream.
    dataset = Dataset(pl.DataFrame({'foo': [0.0]}), DatasetMeta(identifier='uncached_dataset'))
    upstream.add_dataset(dataset)
    return dataset


@pytest.fixture
def new_dataset() -> Dataset:
    return Dataset(pl.DataFrame({'bar': [123.0]}), DatasetMeta(identifier='new_dataset'))


def test_load_dataset_cached(upstream: Upstream, repo: Repository, dataset: Dataset) -> None:
    repo.load_dataset(dataset.identifier)
    upstream.clear_cloud()
    assert_frame_equal(repo.load_dataset(dataset.identifier).df, dataset.df)


def test_load_dataset_not_cached(repo: Repository, dataset: Dataset) -> None:
    assert not repo.is_dataset_cached(dataset.identifier)
    loaded = repo.load_dataset(dataset.identifier)
    assert repo.is_dataset_cached(dataset.identifier)
    assert_frame_equal(loaded.df, dataset.df)


def test_load_dataset_does_not_exist(repo: Repository) -> None:
    with pytest.raises(Exception, match=r'File does-not-exist\.parquet\.dvc not in git repo'):
        repo.load_dataset('does-not-exist')


def test_load_dataset_returns_cached_despite_update(upstream: Upstream, repo: Repository, dataset: Dataset) -> None:
    original = repo.load_dataset(dataset.identifier)
    updated = original.copy()
    assert updated.df is not None
    updated.df = updated.df.with_columns(pl.col('b') + 1)
    upstream.add_dataset(updated)
    assert_frame_equal(repo.load_dataset(dataset.identifier).df, original.df)
    repo.pull_datasets()
    assert_frame_equal(repo.load_dataset(dataset.identifier).df, updated.df)


def test_load_dataset_fail_before_pull(repo: Repository, uncached_dataset: Dataset) -> None:
    with pytest.raises(Exception, match=r'File uncached_dataset\.parquet\.dvc not in git repo'):
        repo.load_dataset(uncached_dataset.identifier)


def test_load_dataframe(repo: Repository, dataset: Dataset) -> None:
    assert_frame_equal(repo.load_dataframe(dataset.identifier), dataset.df)


def test_has_dataset_yes(repo: Repository) -> None:
    assert repo.has_dataset('dataset')


def test_has_dataset_no(repo: Repository) -> None:
    assert not repo.has_dataset('does-not-exist')


def test_pull_dataset_enables_loading(repo: Repository, uncached_dataset: Dataset) -> None:
    repo.pull_datasets()
    assert_frame_equal(repo.load_dataset(uncached_dataset.identifier).df, uncached_dataset.df)


def test_push_dataset_fails_if_stage_nonempty(repo: Repository, new_dataset: Dataset) -> None:
    repo.add(new_dataset)
    with pytest.raises(ValueError, match='nonempty stage'):
        repo.push_dataset(new_dataset)


def test_push_dataset_creates_dvc_file_in_cache(repo: Repository, new_dataset: Dataset) -> None:
    path = repo.repo_dir / f'{new_dataset.identifier}.parquet.dvc'
    assert not path.exists()
    repo.push_dataset(new_dataset)
    assert path.is_file()


def test_push_dataset_creates_dvc_file_in_upstream_repo(
    repo: Repository, upstream: Upstream, new_dataset: Dataset,
) -> None:
    path = f'{new_dataset.identifier}.parquet.dvc'
    assert not upstream.contains(path)
    repo.push_dataset(new_dataset)
    assert upstream.contains(path)
    # A separate consumer must be able to load what was published.
    git(upstream.workspace_dir, 'pull', '--ff-only')
    consumer = Repository(str(upstream.workspace_dir))
    try:
        assert_frame_equal(consumer.load_dataset(new_dataset.identifier).df, new_dataset.df)
    finally:
        consumer.dvc_repo.close()
        consumer.git_repo.free()


def test_push_dataset_creates_parquet_file_in_cache(repo: Repository, new_dataset: Dataset) -> None:
    path = repo.repo_dir / f'{new_dataset.identifier}.parquet'
    assert not path.exists()
    repo.push_dataset(new_dataset)
    assert path.is_file()
    assert repo.is_dataset_cached(new_dataset.identifier)


def test_push_dataset_creates_parquet_file_in_cloud(repo: Repository, new_dataset: Dataset) -> None:
    repo.push_dataset(new_dataset)
    manifest = repo.get_dataset_manifest(new_dataset.identifier)
    assert Path(manifest.object_url.removeprefix('file://')).is_file()
