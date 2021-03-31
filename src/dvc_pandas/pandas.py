import dvc.repo
import pandas as pd

from .git import get_updated_repo


def load_dataset(repo_url, dataset_path):
    git_repo = get_updated_repo(repo_url)
    dvc_repo = dvc.repo.Repo(git_repo.working_dir)
    dataset_path_abs = f'{dvc_repo.root_dir}/{dataset_path}'
    dvc_repo.pull(dataset_path_abs)
    return pd.read_parquet(dataset_path_abs)
