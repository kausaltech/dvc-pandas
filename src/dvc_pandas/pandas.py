import dvc.repo
import os
import pandas as pd
from pathlib import Path

from .git import get_repo


def load_dataset(identifier, repo_url=None):
    """Load dataset from (cached) repo"""
    git_repo = get_repo(repo_url)
    dvc_repo = dvc.repo.Repo(git_repo.working_dir)
    dataset_path = Path(dvc_repo.root_dir) / (identifier + '.parquet')
    dvc_repo.pull(str(dataset_path))
    return pd.read_parquet(dataset_path)


def pull_datasets(repo_url=None):
    """Make sure the given repo exists in the cache and is pulled to the latest version"""
    git_repo = get_repo(repo_url)
    git_repo.remote().pull()


def push_dataset(dataset, identifier, repo_url=None, dvc_remote=None):
    if dvc_remote is None:
        dvc_remote = os.environ.get('DVC_PANDAS_DVC_REMOTE')
    git_repo = get_repo(repo_url)
    dvc_repo = dvc.repo.Repo(git_repo.working_dir)
    dataset_path = Path(dvc_repo.root_dir) / (identifier + '.parquet')
    os.makedirs(dataset_path.parent, exist_ok=True)
    dataset.to_parquet(dataset_path)
    try:
        # Remove old .dvc file (if it exists) and replace it with a new one
        dvc_file_path = dataset_path.parent / (dataset_path.name + '.dvc')
        if dvc_file_path.exists():
            dvc_repo.remove(str(dvc_file_path))
        dvc_repo.add(str(dataset_path))
        dvc_repo.push(str(dvc_file_path), remote=dvc_remote)
        # Stage .dvc file and .gitignore
        gitignore_path = dvc_file_path.parent / '.gitignore'
        git_repo.index.add([str(dvc_file_path), str(gitignore_path)])
        git_repo.index.commit(f'Update {dataset_path.name}')
        git_repo.remote().push()
    except Exception:
        # Restore original data
        # git reset --hard origin/master
        git_repo.head.reset('origin/master', index=True, working_tree=True)
        dvc_repo.checkout(str(dataset_path), force=True)
        # Remove rolled back stuff that may be left on the remote
        # dvc_repo.gc(all_commits=True, cloud=True, remote=dvc_remote)
        # TODO: Before reinstating the previous line, make sure gc doesn't delete blobs for which we don't have a git
        # commit because our git repository is outdated.
        raise
    finally:
        # Remove old version (or new if we rolled back) from cache
        dvc_repo.gc(workspace=True)
