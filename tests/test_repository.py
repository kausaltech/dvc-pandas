import pytest
from dvc.exceptions import NoOutputOrStageError

from dvc_pandas.dvc import set_dvc_file_metadata
from dvc_pandas.git import get_cache_repo
from dvc_pandas.repository import Repository

from tests.dir_helpers import TmpDir


def add_dataset(workspace, dataset, commit_message=None, push=True):
    parquet_path = workspace / f'{dataset.identifier}.parquet'
    dataset.df.to_parquet(parquet_path)
    workspace.dvc.add(str(parquet_path))

    if push:
        workspace.dvc.push(str(parquet_path))

    dvc_file_path = parquet_path.parent / (parquet_path.name + '.dvc')
    set_dvc_file_metadata(dvc_file_path, dataset.dvc_metadata)

    # Stage .dvc file and .gitignore
    gitignore_path = dvc_file_path.parent / '.gitignore'
    workspace.scm.add([str(dvc_file_path), str(gitignore_path)])

    if commit_message is None:
        relative_path = parquet_path.relative_to(workspace)
        commit_message = f"Add {relative_path}"
    workspace.scm.commit(commit_message)


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

    add_dataset(workspace, dataset)
    return workspace


@pytest.fixture
def repo(upstream):
    return Repository(upstream, cache_local_repository=True)


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


def test_load_dataset_cached(repo, cache, dataset):
    # TODO: Make sure the file from the cache is returned and not one fetched from upstream / cloud
    loaded = repo.load_dataset('dataset')
    assert loaded.equals(dataset)


def test_load_dataset_not_cached(repo, cache, dataset):
    with cache.chdir():
        cache.dvc.remove('dataset.parquet.dvc', outs=True)
    cache.dvc.gc(workspace=True)
    cache.scm.checkout('master')
    loaded = repo.load_dataset('dataset')
    assert loaded.equals(dataset)


def test_load_dataset_does_not_exist(repo):
    with pytest.raises(NoOutputOrStageError):
        repo.load_dataset('does-not-exist')


def test_load_dataframe(repo, dataset):
    loaded = repo.load_dataframe('dataset')
    assert loaded.equals(dataset.df)


def test_has_dataset_exists(repo):
    assert repo.has_dataset('dataset')


def test_has_dataset_does_not_exist(repo):
    assert not repo.has_dataset('does-not-exist')


def test_load_dataset_returns_cached_even_if_updated(upstream, repo, dataset):
    ds = repo.load_dataset('dataset')
    ds_updated = ds.copy()
    ds_updated.df['b'] += 1
    add_dataset(upstream, ds_updated, commit_message="Update dataset")

    loaded = repo.load_dataset('dataset')
    assert loaded.equals(ds)
