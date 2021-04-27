from ruamel.yaml import YAML


def set_dvc_file_metadata(dvc_file_path, metadata=None):
    yaml = YAML()
    with open(dvc_file_path, 'rt') as file:
        data = yaml.load(file)
    if metadata is None:
        data.pop('meta', None)
    else:
        data['meta'] = metadata
    with open(dvc_file_path, 'w') as file:
        yaml.dump(data, file)
