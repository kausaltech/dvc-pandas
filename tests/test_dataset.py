from dvc_pandas import load_dataset


def test_load():
    df = load_dataset('https://github.com/juyrjola/dvctest', 'statfi/fuel')
    # TODO
    assert False
