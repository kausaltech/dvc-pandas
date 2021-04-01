import dvc.exceptions
import dvc.repo
import pandas as pd
from pathlib import Path

from .git import get_repo


def load_dataset(repo_url, dataset_path):
    """Load dataset from (cached) repo

    `dataset_path` is a relative path to a parquet file in the repo.
    """
    git_repo = get_repo(repo_url)
    dvc_repo = dvc.repo.Repo(git_repo.working_dir)
    # Make dataset_path absolute
    dataset_path = Path(dvc_repo.root_dir) / dataset_path
    dvc_repo.pull(str(dataset_path))
    return pd.read_parquet(dataset_path)


def pull_datasets(repo_url):
    """Make sure the given repo exists in the cache and is pulled to the latest version"""
    git_repo = get_repo(repo_url)
    git_repo.remote().pull()


def push_dataset(dataset, repo_url, dataset_path, remote):
    git_repo = get_repo(repo_url)
    dvc_repo = dvc.repo.Repo(git_repo.working_dir)
    # Make dataset_path absolute
    dataset_path = Path(dvc_repo.root_dir) / dataset_path
    dataset.to_parquet(dataset_path)
    try:
        # Remove old .dvc file and replace it with a new one
        dvc_file_path = dataset_path.parent / (dataset_path.name + '.dvc')
        dvc_repo.remove(str(dvc_file_path))
        dvc_repo.add(str(dataset_path))
        dvc_repo.push(str(dvc_file_path))
        # Stage .dvc file and .gitignore
        gitignore_path = dvc_file_path.parent / '.gitignore'
        git_repo.index.add([str(dvc_file_path), str(gitignore_path)])
        git_repo.index.commit(f'Update {dataset_path.name}')
        git_repo.remote().push()
    except Exception:
        # Restore original data
        # git reset --hard origin/master
        git_repo.head.reset('origin/master', index=True, working_tree=True)
        dvc_repo.checkout(str(dataset_path))
        # Remove `dataset` from cache
        dvc_repo.gc(all_commits=True)
        # TODO: Run gc with cloud=True, but at the moment this produces an error
        raise
    # Remove old version from cache
    dvc_repo.gc(workspace=True)
