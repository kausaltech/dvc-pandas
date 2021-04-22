import pandas as pd
import pytest
from dvc.exceptions import NoOutputOrStageError

from dvc_pandas import has_dataset, load_dataframe
from dvc_pandas.git import get_cache_repo

from tests.dir_helpers import TmpDir


def add_dataset(workspace, df, dataset_name=None, commit_message=None, push=True):
    if dataset_name is None:
        dataset_name = 'dataset'
    if commit_message is None:
        commit_message = "Add dataset"

    # Add dataset
    dataset_path = workspace / f'{dataset_name}.parquet'
    df.to_parquet(dataset_path)
    workspace.dvc_add(dataset_path, commit=commit_message)

    if push:
        # Push to remote
        workspace.dvc.push(str(dataset_path))


@pytest.fixture
def local_cloud(make_tmp_dir):
    ret = make_tmp_dir("local-cloud")
    ret.url = str(ret)
    ret.config = {"url": ret.url}
    return ret


@pytest.fixture
def upstream(tmp_path, local_cloud, dataset):
    # Initialize repo
    repo_dir = tmp_path / 'upstream'
    workspace = TmpDir(repo_dir)
    workspace.init(scm=True, dvc=True)

    # Add remote
    workspace.add_remote(config=local_cloud.config)

    add_dataset(workspace, dataset, 'dataset')
    return workspace


@pytest.fixture(autouse=True)
def cache(upstream, tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr('dvc_pandas.git.CACHE_DIR', cache_dir)

    # Clone
    git_repo = get_cache_repo(upstream, cache_local_repository=True)

    repo_dir = git_repo.working_dir
    workspace = TmpDir(repo_dir)
    import dvc
    workspace.dvc = dvc.repo.Repo(repo_dir)
    workspace.scm = workspace.dvc.scm
    return workspace


@pytest.fixture
def dataset():
    return pd.DataFrame([(1, 2), (3, 4)], columns=['a', 'b'])


def test_load_dataframe_cached(upstream, local_cloud, cache, dataset):
    # TODO: Make sure the file from the cache is returned and not one fetched from upstream / cloud
    # assert False
    loaded = load_dataframe('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(dataset)


def test_load_dataframe_not_cached(upstream, cache, dataset):
    with cache.chdir():
        cache.dvc.remove('dataset.parquet.dvc', outs=True)
    cache.dvc.gc(workspace=True)
    cache.scm.checkout('master')
    loaded = load_dataframe('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(dataset)


def test_load_dataframe_does_not_exist(upstream):
    with pytest.raises(NoOutputOrStageError):
        load_dataframe('does-not-exist', upstream, cache_local_repository=True)


def test_has_dataset_exists(upstream):
    assert has_dataset('dataset', upstream, cache_local_repository=True)


def test_has_dataset_does_not_exist(upstream):
    assert not has_dataset('does-not-exist', upstream, cache_local_repository=True)


def test_load_dataframe_returns_cached_even_if_updated(upstream, dataset):
    df = load_dataframe('dataset', upstream, cache_local_repository=True)
    df_updated = df.copy()
    df_updated['b'] += 1
    add_dataset(upstream, df_updated, 'dataset', commit_message="Update dataset")

    loaded = load_dataframe('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(df)
