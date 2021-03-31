from dvc_pandas import load_dataset


def test_load():
    df = load_dataset('git@github.com:kausaltech/dvctest', 'statfi/fuel.parquet')
    assert 'co2e_emission_factor' in df.columns
    # TODO: Set up mock repository
