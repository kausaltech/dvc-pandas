import pandas as pd
import pytest

from dvc_pandas import Dataset


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
        'time': ([24 / 60], 'min'),
    }
    df = pd.DataFrame({column: values for column, (values, _) in data.items()})
    units = {column: unit for column, (_, unit) in data.items()}
    ds = Dataset(df, 'test', units=units)
    ds.df['foo'] = ds.df.speed * ds.df.time
    assert list(ds.df.foo.pint.to('MB').values.data) == [0.024576]
