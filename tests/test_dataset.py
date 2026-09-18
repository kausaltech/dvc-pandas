from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dvc_pandas import Dataset, DatasetMeta
from polars.testing import assert_frame_equal


def test_init_metadata_with_units_key(df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match='metadata may not contain'):
        Dataset(df, DatasetMeta(identifier='test', metadata={'units': {'a': 'kg'}}))


def test_init_unit_for_invalid_column(df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match='unknown column'):
        Dataset(df, DatasetMeta(identifier='test', units={'missing': 'kg'}))


def test_dvc_metadata_contains_units(df: pl.DataFrame) -> None:
    dataset = Dataset(df, DatasetMeta(identifier='test', units={'a': 'kg'}))
    assert dataset.dvc_metadata == {'units': {'a': 'kg'}}


def test_parquet_roundtrip_preserves_units_and_index(tmp_path: Path) -> None:
    frame = pl.DataFrame({'Year': [2020, 2021], 'Value': [1.0, 2.0]})
    dataset = Dataset(frame, DatasetMeta(identifier='test', units={'Value': 'kg'}, index_columns=['Year']))
    path = tmp_path / 'dataset.parquet'
    dataset.to_parquet(path)
    loaded = Dataset.from_parquet(path, DatasetMeta(identifier='test'))
    assert loaded.units == dataset.units
    assert loaded.index_columns == dataset.index_columns
    assert_frame_equal(loaded.df.select(frame.columns), frame)


def test_copy_is_independent(dataset: Dataset) -> None:
    copied = dataset.copy()
    assert copied.df is not None
    copied.df = copied.df.with_columns(pl.col('a') + 1)
    assert dataset.df is not None
    assert not copied.df.equals(dataset.df)


@pytest.mark.parametrize('name', [None, 'Year'])
def test_legacy_parquet_range_index(tmp_path: Path, name: str | None) -> None:
    frame = pd.DataFrame({'Value': [1.0, 2.0]}, index=pd.RangeIndex(2020, 2024, 2, name=name))
    path = tmp_path / 'legacy.parquet'
    pq.write_table(pa.Table.from_pandas(frame), path)
    loaded = Dataset.from_parquet(path, DatasetMeta(identifier='legacy'))
    assert loaded.index_columns == ([] if name is None else ['Year'])
    assert loaded.df is not None
    if name is None:
        assert loaded.df.columns == ['Value']
    else:
        assert loaded.df['Year'].to_list() == [2020, 2022]
    # Loaded metadata must also be valid input to the writer.
    loaded.to_parquet(tmp_path / 'rewritten.parquet')
