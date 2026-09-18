from __future__ import annotations

import polars as pl
import pytest
from dvc_pandas import Dataset, DatasetMeta


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({'a': [1.0, 3.0], 'b': [2.0, 4.0]})


@pytest.fixture
def dataset(df: pl.DataFrame) -> Dataset:
    return Dataset(df, DatasetMeta(identifier='dataset'))
