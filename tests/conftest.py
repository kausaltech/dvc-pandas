import pandas as pd
import pytest

from dvc_pandas import Dataset

from .dir_helpers import *  # noqa, pylint: disable=wildcard-import


@pytest.fixture
def df():
    return pd.DataFrame([(1.0, 2.0), (3.0, 4.0)], columns=['a', 'b'])


@pytest.fixture
def dataset(df):
    return Dataset(df, 'dataset')
