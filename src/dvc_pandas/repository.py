import dvc.repo
import os
import logging
from pathlib import Path
from ruamel.yaml import YAML

from .git import get_cache_repo, push as git_push
from .dvc import set_dvc_file_metadata
from .dataset import Dataset

logger = logging.getLogger(__name__)


def _need_commit(git_repo):
    if not git_repo.is_dirty():
        return False
    # We don't commit if the only files changed are .gitignore files. DVC sometimes reorders the lines in .gitignore
    # even though no dataset has changed.
    diffs = git_repo.index.diff(git_repo.head.commit)
    return any(os.path.basename(diff.a_path) != '.gitignore' for diff in diffs)


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
        self.dvc_repo = dvc.repo.Repo(self.repo_dir)

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
            self.dvc_repo.pull(str(parquet_path))
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
        self._add_file(parquet_path, metadata=dataset.dvc_metadata)

    def _add_file(self, parquet_path, metadata=None):
        try:
            # Remove old .dvc file (if it exists) and replace it with a new one
            dvc_file_path = parquet_path.parent / (parquet_path.name + '.dvc')
            if dvc_file_path.exists():
                logger.debug(f"Remove file {dvc_file_path} from DVC")
                self.dvc_repo.remove(str(dvc_file_path))

            logger.debug(f"Add file {parquet_path} to DVC")
            self.dvc_repo.add(str(parquet_path))

            logger.debug(f"Set metadata to {metadata}")
            set_dvc_file_metadata(dvc_file_path, metadata)

            logger.debug(f"Push file {dvc_file_path} to DVC remote {self.dvc_remote}")
            self.dvc_repo.push(str(dvc_file_path), remote=self.dvc_remote)

            # Stage .dvc file and .gitignore
            gitignore_path = dvc_file_path.parent / '.gitignore'
            self.git_repo.index.add([str(dvc_file_path), str(gitignore_path)])

            # Commit and push
            if _need_commit(self.git_repo):
                relative_path = parquet_path.relative_to(self.git_repo.working_dir)
                self.git_repo.index.commit(f'Update {relative_path}')
                logger.debug("Push to git repository")
                git_push(self.git_repo)
            else:
                logger.debug("No commit since datasets have not changed")
        except Exception:
            # Restore original data
            # git reset --hard origin/master
            logger.debug("Hard-reset master branch to origin/master")
            self.git_repo.head.reset('origin/master', index=True, working_tree=True)

            logger.debug(f"Checkout DVC dataset {parquet_path}")
            self.dvc_repo.checkout(str(parquet_path), force=True)

            # Remove rolled back stuff that may be left on the remote
            # self.dvc_repo.gc(all_commits=True, cloud=True, remote=self.dvc_remote)
            # TODO: Before reinstating the previous line, make sure gc doesn't delete blobs for which we don't have
            # a git commit because our git repository is outdated.
            raise
        finally:
            # Remove old version (or new if we rolled back) from cache
            logger.debug("Collect garbage in local DVC repository")
            self.dvc_repo.gc(workspace=True)
