import os
import logging
from pathlib import Path

from .dvc import add_file_to_repo, pull as dvc_pull
from .git import get_cache_repo, local_cache_dir

logger = logging.getLogger(__name__)


def load_dataset(identifier, repo_url=None, cache_local_repository=False):
    """
    Load dataset with the given identifier from the given repository.

    Returns cached dataset if possible, otherwise clones git repository (if necessary) and pulls dataset from DVC.
    """
    import pandas as pd

    repo_dir = local_cache_dir(repo_url, cache_local_repository=cache_local_repository)
    dataset_path = Path(repo_dir) / (identifier + '.parquet')
    if not dataset_path.exists():
        git_repo = get_cache_repo(repo_url, cache_local_repository=cache_local_repository)
        logger.debug(f"Pull dataset {dataset_path} from DVC")
        dvc_pull(dataset_path, git_repo.working_dir)
    return pd.read_parquet(dataset_path)


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


def push_dataset(dataset, identifier, repo_url=None, dvc_remote=None, metadata=None):
    """
    Add the given dataset as a parquet file to DVC.

    Updates the git repository first.
    """
    if dvc_remote is None:
        dvc_remote = os.environ.get('DVC_PANDAS_DVC_REMOTE')

    git_repo = pull_datasets(repo_url)
    dataset_path = Path(git_repo.working_dir) / (identifier + '.parquet')
    os.makedirs(dataset_path.parent, exist_ok=True)
    dataset.to_parquet(dataset_path)
    add_file_to_repo(dataset_path, git_repo, dvc_remote=dvc_remote, metadata=metadata)
