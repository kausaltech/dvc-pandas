import os
import logging
import pandas as pd
from pathlib import Path
from pint_pandas import PintArray
from ruamel.yaml import YAML

from .dvc import add_file_to_repo, pull as dvc_pull
from .git import get_cache_repo, local_cache_dir

logger = logging.getLogger(__name__)


def _quantify_series(series, unit):
    if unit:
        return PintArray(series, unit)
    return series


class Dataset:
    def __init__(self, df, identifier, units=None, metadata=None):
        """
        Create a dataset from a Pandas DataFrame, an identifier and optional metadata.

        If `units` is specified, it should be a dict that maps (some of) the columns of `df` to physical units. If any
        key of this dict is not a column in `df`, a ValueError is raised. The DataFrame in this dataset will use
        instances of PintArray of the respective unit for each column specified in `units`.

        You can specify metadata to be stored in the .dvc file by setting the `metadata` parameter to a dict.  Units
        will be stored in the metadata using the key `units`, so the `metadata` dict is not allowed to contain this key
        and a ValueError will be raised if it does.
        """
        if metadata and 'units' in metadata:
            raise ValueError("Dataset metadata may not contain the key 'units'.")
        if units is not None:
            for column in units.keys():
                if column not in df.columns:
                    raise ValueError(f"Unit specified for unknown column name '{column}'.")

        if units:
            self.df = pd.DataFrame({
                column: _quantify_series(df[column], units.get(column))
                for column in df.columns
            })
        else:
            self.df = df.copy()

        self.identifier = identifier
        self.units = units
        self.metadata = metadata

    def copy(self):
        return Dataset(self.df.copy(), self.identifier, units=self.units, metadata=self.metadata)

    @property
    def dvc_metadata(self):
        """
        Return the metadata as it should be stored in the .dvc file.

        Physical units will be stored as part of the metadata using the key `units`.
        """
        if self.metadata is None and self.units is None:
            return None
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata
        if self.units is not None:
            metadata['units'] = self.units
        return metadata

    def equals(self, other):
        compare_attrs = ('identifier', 'units', 'metadata')
        return self.df.equals(other.df) and all(getattr(self, a) == getattr(other, a) for a in compare_attrs)

    def __str__(self):
        return self.identifier


def load_dataset(identifier, repo_url=None, cache_local_repository=False):
    """
    Load dataset with the given identifier from the given repository.

    Returns cached dataset if possible, otherwise clones git repository (if necessary) and pulls dataset from DVC.
    """
    import pandas as pd

    repo_dir = local_cache_dir(repo_url, cache_local_repository=cache_local_repository)
    parquet_path = Path(repo_dir) / (identifier + '.parquet')
    if not parquet_path.exists():
        git_repo = get_cache_repo(repo_url, cache_local_repository=cache_local_repository)
        logger.debug(f"Pull dataset {parquet_path} from DVC")
        dvc_pull(parquet_path, git_repo.working_dir)
    df = pd.read_parquet(parquet_path)

    # Get metadata (including units) from .dvc file
    dvc_file_path = parquet_path.parent / (parquet_path.name + '.dvc')
    yaml = YAML()
    with open(dvc_file_path, 'rt') as file:
        metadata = yaml.load(file).get('meta')

    if metadata is None:
        units = None
    else:
        units = metadata.pop('units', None)

    return Dataset(df, identifier, units=units, metadata=metadata)


def load_dataframe(identifier, repo_url=None, cache_local_repository=False):
    """
    Same as load_dataset, but only provides the DataFrame for convenience.
    """
    dataset = load_dataset(identifier, repo_url=repo_url, cache_local_repository=cache_local_repository)
    return dataset.df


def has_dataset(identifier, repo_url=None, cache_local_repository=False):
    """
    Check if a dataset with the given identifier exists in the given repository.

    Clones git repository if it's not in the cache.
    """
    repo_dir = local_cache_dir(repo_url, cache_local_repository=cache_local_repository)
    if not repo_dir.exists():
        get_cache_repo(repo_url, cache_local_repository=cache_local_repository)
    dvc_file_path = Path(repo_dir) / (identifier + '.parquet.dvc')
    return os.path.exists(dvc_file_path)


def pull_datasets(repo_url=None, cache_local_repository=False):
    """
    Make sure the given git repository exists in the cache and is pulled to the latest version.

    Returns the git repo.
    """
    git_repo = get_cache_repo(repo_url, cache_local_repository=cache_local_repository)
    logger.debug(f"Pull from git remote for repository {repo_url}")
    git_repo.remote().pull()
    return git_repo


def push_dataset(dataset, repo_url=None, dvc_remote=None):
    """
    Add the given dataset as a parquet file to DVC.

    Updates the git repository first.
    """
    if dvc_remote is None:
        dvc_remote = os.environ.get('DVC_PANDAS_DVC_REMOTE')

    git_repo = pull_datasets(repo_url)
    parquet_path = Path(git_repo.working_dir) / (dataset.identifier + '.parquet')
    os.makedirs(parquet_path.parent, exist_ok=True)
    dataset.df.to_parquet(parquet_path)
    add_file_to_repo(parquet_path, git_repo, dvc_remote=dvc_remote, metadata=dataset.dvc_metadata)
