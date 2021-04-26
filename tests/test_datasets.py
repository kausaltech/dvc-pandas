import pandas as pd
import pytest
from dvc.exceptions import NoOutputOrStageError

from dvc_pandas import Dataset, has_dataset, load_dataframe, load_dataset
from dvc_pandas.dvc import set_dvc_file_metadata
from dvc_pandas.git import get_cache_repo

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
def df():
    return pd.DataFrame([(1.0, 2.0), (3.0, 4.0)], columns=['a', 'b'])


@pytest.fixture
def dataset(df):
    return Dataset(df, 'dataset')


def test_init_metadata_with_units_key(df):
    with pytest.raises(ValueError):
        Dataset(df, 'test', metadata={'units': {df.columns[0]: 'foo'}})


def test_init_unit_for_invalid_column(df):
    with pytest.raises(ValueError):
        Dataset(df, 'test', units={'does-not-exist': 'kg'})


def test_dvc_metadata_contains_units(df):
    units = {df.columns[0]: 'kg'}
    ds = Dataset(df, 'test', units=units)
    assert ds.dvc_metadata['units'] == units


def test_dataset_units():
    data = {
        'speed': ([1.0], 'KiB/s'),
        'time': ([24/60], 'min'),
    }
    df = pd.DataFrame({column: values for column, (values, _) in data.items()})
    units = {column: unit for column, (_, unit) in data.items()}
    ds = Dataset(df, 'test', units=units)
    ds.df['foo'] = ds.df.speed * ds.df.time
    assert list(ds.df.foo.pint.to('MB').values.data) == [0.024576]


def test_load_dataset_cached(upstream, cache, dataset):
    # TODO: Make sure the file from the cache is returned and not one fetched from upstream / cloud
    # assert False
    loaded = load_dataset('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(dataset)


def test_load_dataset_not_cached(upstream, cache, dataset):
    with cache.chdir():
        cache.dvc.remove('dataset.parquet.dvc', outs=True)
    cache.dvc.gc(workspace=True)
    cache.scm.checkout('master')
    loaded = load_dataset('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(dataset)


def test_load_dataset_does_not_exist(upstream):
    with pytest.raises(NoOutputOrStageError):
        load_dataset('does-not-exist', upstream, cache_local_repository=True)


def test_load_dataframe(upstream, dataset):
    loaded = load_dataframe('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(dataset.df)


def test_has_dataset_exists(upstream):
    assert has_dataset('dataset', upstream, cache_local_repository=True)


def test_has_dataset_does_not_exist(upstream):
    assert not has_dataset('does-not-exist', upstream, cache_local_repository=True)


def test_load_dataset_returns_cached_even_if_updated(upstream, dataset):
    ds = load_dataset('dataset', upstream, cache_local_repository=True)
    ds_updated = ds.copy()
    ds_updated.df['b'] += 1
    add_dataset(upstream, ds_updated, commit_message="Update dataset")

    loaded = load_dataset('dataset', upstream, cache_local_repository=True)
    assert loaded.equals(ds)
