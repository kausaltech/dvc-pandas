import pandas as pd
import pytest
from dataclasses import dataclass
from pathlib import Path

from dvc_pandas import load_dataset


@pytest.fixture
def local_cloud(make_tmp_dir):
    ret = make_tmp_dir("local-cloud")
    ret.url = str(ret)
    ret.config = {"url": ret.url}
    return ret


@pytest.fixture
def local_remote(tmp_dir, dvc, local_cloud):
    tmp_dir.add_remote(config=local_cloud.config)
    yield local_cloud


@dataclass
class Dataset:
    identifier: str
    df: pd.DataFrame
    parquet_file: Path


@pytest.fixture
def dataset(tmp_dir):
    df = pd.DataFrame([(1, 2), (3, 4)], columns=['a', 'b'])
    path = tmp_dir / 'test.parquet'
    df.to_parquet(path)
    return Dataset('test', df, path)


def test_load_dataset_cached(tmp_dir, scm, dvc, dataset, local_remote):
    tmp_dir.dvc_add([dataset.parquet_file])
    dataset.parquet_file.unlink()
    loaded = load_dataset(dataset.identifier, tmp_dir)
    assert loaded.equals(dataset.df)


def test_load_dataset_not_cached(tmp_dir, scm, dvc, dataset, local_remote):
    stage = tmp_dir.dvc_add([dataset.parquet_file], commit="Add test file")[0]
    dvc.push(stage.path)
    dvc.remove(stage.path, outs=True)
    dvc.gc(workspace=True)
    scm.checkout('master')
    loaded = load_dataset(dataset.identifier, tmp_dir)
    assert loaded.equals(dataset.df)
