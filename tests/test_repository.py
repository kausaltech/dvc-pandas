import dvc.repo
import git
import os
import pandas as pd
import pytest
import shutil
from contextlib import contextmanager
from dvc.exceptions import NoOutputOrStageError
from pathlib import Path
from ruamel.yaml import YAML

from dvc_pandas import Dataset, Repository
from dvc_pandas.git import get_cache_repo
from dvc_pandas.dvc import set_dvc_file_metadata


def _git_repo_contains(repo, path):
    path = Path(path)
    tree = repo.head.commit.tree
    for part in path.parent.parts:
        try:
            tree = tree[part]
        except KeyError:
            return False
    return str(path) in tree


class Upstream:
    def __init__(self, path):
        self.bare_repo_dir = path / 'upstream'
        self.workspace_dir = path / 'upstream_workspace'
        self.cloud_dir = path / 'cloud'

        self.bare_repo = git.Repo.init(str(self.bare_repo_dir), bare=True)

        self.git_workspace = self.bare_repo.clone(str(self.workspace_dir))
        self.dvc_workspace = dvc.repo.Repo.init(str(self.workspace_dir))

        # Set DVC remote
        self.cloud_dir.mkdir()
        with self.dvc_workspace.config.edit() as conf:
            conf["remote"]['cloud'] = {'url': str(self.cloud_dir)}
            conf["core"]["remote"] = 'cloud'
        self.git_workspace.index.add([self.dvc_workspace.config.files['repo']])

        self.git_workspace.index.commit("Initialize DVC")
        self.git_workspace.remote().push()

    def add_dataset(self, dataset, commit_message=None, push=True, delete_output=True, clear_cache=True):
        parquet_path = self.workspace_dir / f'{dataset.identifier}.parquet'
        dataset.df.to_parquet(parquet_path)
        self.dvc_workspace.add(str(parquet_path))

        if push:
            self.dvc_workspace.push(str(parquet_path))

        if delete_output:
            parquet_path.unlink()

        if clear_cache:
            cache_dir = Path(self.dvc_workspace.odb.local.cache_dir)
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()

        dvc_file_path = parquet_path.parent / (parquet_path.name + '.dvc')
        set_dvc_file_metadata(dvc_file_path, dataset.dvc_metadata)

        # Stage .dvc file and .gitignore
        gitignore_path = dvc_file_path.parent / '.gitignore'
        self.git_workspace.index.add([str(dvc_file_path), str(gitignore_path)])

        if commit_message is None:
            commit_message = f"Add {dataset.identifier}"
        self.git_workspace.index.commit(commit_message)
        self.git_workspace.remote().push()

    def clear_cloud(self):
        shutil.rmtree(self.cloud_dir)
        self.cloud_dir.mkdir()

    def cloud_contains_data_for_dvc_file(self, dvc_file):
        yaml = YAML()
        with open(dvc_file, 'rt') as file:
            out, = yaml.load(file).get('outs')
        md5 = out['md5']
        prefix, filename = md5[:2], md5[2:]
        path = self.cloud_dir / prefix / filename
        return path.exists()

    def repo_contains(self, item):
        if isinstance(item, Dataset):
            item = item.identifier + '.parquet'
        return _git_repo_contains(self.bare_repo, item)


class Cache:
    def __init__(self, cache_root, upstream):
        self.cache_root = cache_root
        self.git_repo = get_cache_repo(upstream.bare_repo_dir, cache_local_repository=True, cache_root=cache_root)
        self.repo_dir = Path(self.git_repo.working_dir)
        self.dvc_repo = dvc.repo.Repo(self.repo_dir)

    def __contains__(self, item):
        if isinstance(item, Dataset):
            item = item.identifier + '.parquet'
        path = self.repo_dir / item
        return path.exists()

    @contextmanager
    def chdir(self):
        old = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            yield
        finally:
            os.chdir(old)


@pytest.fixture
def upstream(tmp_path, dataset):
    u = Upstream(tmp_path)
    u.add_dataset(dataset)
    return u


@pytest.fixture
def repo(upstream, cache):
    return Repository(upstream.bare_repo_dir, cache_local_repository=True, cache_root=cache.cache_root)


@pytest.fixture(autouse=True)
def cache(upstream, tmp_path):
    cache_root = tmp_path / 'cache'
    # repo_dir = git_repo.working_dir
    # workspace = TmpDir(repo_dir)
    # workspace.dvc = dvc.repo.Repo(repo_dir)
    # workspace.scm = workspace.dvc.scm
    return Cache(cache_root, upstream)


@pytest.fixture
def uncached_dataset(upstream):
    """Dataset that is added to upstream but not yet in cache."""
    df = pd.DataFrame([(0.0,)], columns=['foo'])
    ds = Dataset(df, 'uncached_dataset')
    upstream.add_dataset(ds)
    return ds


@pytest.fixture
def new_dataset():
    """Dataset that is neither in upstream nor in cache."""
    df = pd.DataFrame([(123.0,)], columns=['bar'])
    return Dataset(df, 'new_dataset')


def test_load_dataset_cached(upstream, repo, dataset):
    # Load dataset to get it into cache
    repo.load_dataset('dataset')
    # Remove dataset from cloud to make sure it is returned from cache
    upstream.clear_cloud()
    loaded = repo.load_dataset('dataset')
    assert loaded.equals(dataset)


def test_load_dataset_not_cached(repo, cache, dataset):
    assert dataset not in cache
    loaded = repo.load_dataset(dataset.identifier)
    assert dataset in cache
    assert loaded.equals(dataset)


def test_load_dataset_does_not_exist(repo):
    with pytest.raises(NoOutputOrStageError):
        repo.load_dataset('does-not-exist')


def test_load_dataset_returns_cached_despite_update(upstream, repo, dataset):
    ds = repo.load_dataset('dataset')
    ds_updated = ds.copy()
    ds_updated.df['b'] += 1
    upstream.add_dataset(ds_updated, commit_message="Update dataset")
    loaded = repo.load_dataset('dataset')
    assert loaded.equals(ds)


def test_load_dataset_fail_before_pull(repo, uncached_dataset):
    with pytest.raises(NoOutputOrStageError):
        repo.load_dataset(uncached_dataset.identifier)


def test_load_dataframe(repo, dataset):
    loaded = repo.load_dataframe('dataset')
    assert loaded.equals(dataset.df)


def test_has_dataset_yes(repo):
    assert repo.has_dataset('dataset')


def test_has_dataset_no(repo):
    assert not repo.has_dataset('does-not-exist')


def test_pull_dataset_enables_loading(repo, uncached_dataset):
    repo.pull_datasets()
    loaded = repo.load_dataset(uncached_dataset.identifier)
    assert loaded.equals(uncached_dataset)


def test_push_dataset_fails_if_stage_nonempty(repo, new_dataset):
    repo.add(new_dataset)
    with pytest.raises(ValueError):
        repo.push_dataset(new_dataset)


def test_push_dataset_creates_dvc_file_in_cache(repo, cache, new_dataset):
    new_dvc_file = new_dataset.identifier + '.parquet.dvc'
    assert new_dvc_file not in cache
    repo.push_dataset(new_dataset)
    assert new_dvc_file in cache


def test_push_dataset_creates_dvc_file_in_upstream_repo(repo, upstream, new_dataset):
    new_dvc_file = new_dataset.identifier + '.parquet.dvc'
    assert not upstream.repo_contains(new_dvc_file)
    repo.push_dataset(new_dataset)
    assert upstream.repo_contains(new_dvc_file)


def test_push_dataset_creates_parquet_file_in_cache(repo, cache, new_dataset):
    new_parquet_file = new_dataset.identifier + '.parquet'
    assert new_parquet_file not in cache
    repo.push_dataset(new_dataset)
    assert new_parquet_file in cache


def test_push_dataset_creates_parquet_file_in_cloud(repo, upstream, new_dataset):
    repo.push_dataset(new_dataset)
    new_dvc_file = repo.repo_dir / (new_dataset.identifier + '.parquet.dvc')
    assert upstream.cloud_contains_data_for_dvc_file(new_dvc_file)
