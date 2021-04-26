import os
import logging
from pathlib import Path
from ruamel.yaml import YAML

from .dvc import add_file_to_repo, pull as dvc_pull
from .git import get_cache_repo
from .dataset import Dataset

logger = logging.getLogger(__name__)


class Repository:
    def __init__(self, repo_url=None, dvc_remote=None, cache_local_repository=False):
        """
        Initialize repository.

        Clones git repository if it's not in the cache.
        """
        if dvc_remote is None:
            dvc_remote = os.environ.get('DVC_PANDAS_DVC_REMOTE')

        self.repo_url = repo_url
        self.dvc_remote = dvc_remote
        self.cache_local_repository = cache_local_repository
        self.git_repo = get_cache_repo(repo_url, cache_local_repository=cache_local_repository)
        self.repo_dir = self.git_repo.working_dir

    def load_dataset(self, identifier):
        """
        Load dataset with the given identifier from the given repository.

        Returns cached dataset if possible, otherwise clones git repository (if necessary) and pulls dataset from
        DVC.
        """
        import pandas as pd

        parquet_path = Path(self.repo_dir) / (identifier + '.parquet')
        if not parquet_path.exists():
            logger.debug(f"Pull dataset {parquet_path} from DVC")
            dvc_pull(parquet_path, self.repo_dir)
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

    def load_dataframe(self, identifier):
        """
        Same as load_dataset, but only provides the DataFrame for convenience.
        """
        dataset = self.load_dataset(identifier)
        return dataset.df

    def has_dataset(self, identifier):
        """
        Check if a dataset with the given identifier exists.
        """
        dvc_file_path = Path(self.repo_dir) / (identifier + '.parquet.dvc')
        return os.path.exists(dvc_file_path)

    def pull_datasets(self):
        """
        Make sure the git repository is pulled to the latest version.
        """
        logger.debug("Pull from git remote")
        self.git_repo.remote().pull()

    def push_dataset(self, dataset):
        """
        Add the given dataset as a parquet file to DVC.

        Updates the git repository first.
        """
        self.pull_datasets()
        parquet_path = Path(self.repo_dir) / (dataset.identifier + '.parquet')
        os.makedirs(parquet_path.parent, exist_ok=True)
        dataset.df.to_parquet(parquet_path)
        add_file_to_repo(parquet_path, self.git_repo, dvc_remote=self.dvc_remote, metadata=dataset.dvc_metadata)
