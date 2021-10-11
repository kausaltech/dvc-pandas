import dvc.repo
import fasteners
import logging
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from ruamel.yaml import YAML

from .dataset import Dataset
from .dvc import set_dvc_file_metadata
from .git import get_cache_repo
from .git import push as git_push

logger = logging.getLogger(__name__)


class TemporaryGitCheckout:
    def __init__(self, repo, commit_id=None):
        self.repo = repo
        self.commit_id = commit_id
        self.original_branch = None

    def __enter__(self):
        if self.commit_id:
            self.original_branch = self.repo.active_branch
            self.repo.head.reference = self.repo.commit(self.commit_id)
            self.repo.head.reset(index=True, working_tree=True)
            # TODO: Pull if commit ID doesn't exist? Flag for enabling this beha

    def __exit__(self, *exc):
        if self.original_branch:
            self.original_branch.checkout()


class OptionalLock:
    def __init__(self, lock, should_lock=True):
        self.lock = lock
        self.should_lock = should_lock

    def __enter__(self):
        if self.should_lock:
            self.lock.acquire()

    def __exit__(self, *exc):
        if self.should_lock:
            self.lock.release()


class DatasetStageItem:
    def __init__(self, dataset):
        self.identifier = dataset.identifier
        self.metadata = dataset.dvc_metadata


class Repository:
    dvc_remote: Optional[str]
    repo_url: Optional[str]
    commit_id: Optional[str]
    dataset_stage: List[DatasetStageItem]

    def __init__(
        self, repo_url: str = None, dvc_remote: str = None, cache_local_repository=False,
        cache_root=None, commit_id=None
    ):
        """
        Initialize repository.

        Clones git repository if it's not in the cache.

        If `commit_id` is specified, the given commit ID will be checked out temporarily for most commands and the
        repository will be read-only.
        """
        if dvc_remote is None:
            dvc_remote = os.environ.get('DVC_PANDAS_DVC_REMOTE')

        self.repo_url = repo_url
        self.dvc_remote = dvc_remote
        self.cache_local_repository = cache_local_repository
        self.git_repo = get_cache_repo(
            repo_url, cache_local_repository=cache_local_repository, cache_root=cache_root
        )
        self.repo_dir = Path(self.git_repo.working_dir)
        self.dvc_repo = dvc.repo.Repo(self.repo_dir)
        self.dataset_stage = []
        self.commit_id = commit_id
        self.lock = fasteners.InterProcessLock(self.repo_dir / '.dvc-pandas.lock')

    def load_dataset(self, identifier: str, skip_pull_if_exists=False, lock=True) -> Dataset:
        """
        Load dataset with the given identifier from the given repository.

        If `skip_pull_if_exists` is True, does not update the dataset if a parquet file exists for the identifier,
        regardless of the content.
        """
        with OptionalLock(self.lock, lock):
            with TemporaryGitCheckout(self.git_repo, self.commit_id):
                parquet_path = self.repo_dir / (identifier + '.parquet')
                if not (parquet_path.exists() and skip_pull_if_exists):
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

                mtime = dvc_file_path.stat().st_mtime
                modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc)

                return Dataset(df, identifier, modified_at=modified_at, units=units, metadata=metadata)

    def load_dataframe(self, identifier: str, skip_pull_if_exists=False, lock=True) -> pd.DataFrame:
        """
        Same as load_dataset, but only provides the DataFrame for convenience.
        """
        dataset = self.load_dataset(identifier, skip_pull_if_exists, lock)
        return dataset.df

    def has_dataset(self, identifier: str, lock=True) -> bool:
        """
        Check if a dataset with the given identifier exists.
        """
        with OptionalLock(self.lock, lock):
            with TemporaryGitCheckout(self.git_repo, self.commit_id):
                dvc_file_path = self.repo_dir / (identifier + '.parquet.dvc')
                return os.path.exists(dvc_file_path)

    def pull_datasets(self, lock=True):
        """
        Make sure the git repository is pulled to the latest version.
        """
        with OptionalLock(self.lock, lock):
            logger.debug("Pull from git remote")
            self.git_repo.remote().pull()

    def push_dataset(self, dataset: Dataset, lock=True):
        """
        Add the given dataset as a parquet file to DVC, create a commit and push it.

        Updates the git repository first. If something goes wrong, restores the repository and all files to the state
        of origin/master.

        This is a convenience method wrapping add() and push(). If the dataset stage (the one populated by add()) is not
        empty, this raises a ValueError.
        """
        with OptionalLock(self.lock, lock):
            if self.dataset_stage:
                raise ValueError("push_dataset was called with nonempty stage.")
            if self.read_only:
                raise ValueError("push_dataset was called while repository is read-only.")

            self.pull_datasets(lock=False)
            parquet_path = self.repo_dir / (dataset.identifier + '.parquet')
            self.add(dataset, lock=False)
            try:
                self.push(lock=False)
            except Exception:
                # Restore original data
                # git reset --hard origin/master
                logger.debug("Hard-reset master branch to origin/master")
                self.git_repo.head.reset('origin/master', index=True, working_tree=True)

                logger.debug(f"Checkout {parquet_path} from DVC")
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

    def add(self, dataset: Dataset, lock=True):
        """Create or update parquet file from dataset, but do not create .dvc file, commit or push."""
        with OptionalLock(self.lock, lock):
            if self.read_only:
                raise ValueError("add was called while repository is read-only.")

            parquet_path = self.repo_dir / (dataset.identifier + '.parquet')
            os.makedirs(parquet_path.parent, exist_ok=True)
            dataset.df.to_parquet(str(parquet_path))
            self.dataset_stage.append(DatasetStageItem(dataset))

    def push(self, lock=True):
        """Upload, commit and push the staged files and clear the stage."""
        with OptionalLock(self.lock, lock):
            if self.read_only:
                raise ValueError("add was called while repository is read-only.")

            for stage_item in self.dataset_stage:
                path = self.repo_dir / (stage_item.identifier + '.parquet')

                # Remove old .dvc file (if it exists) and replace it with a new one
                dvc_file_path = path.parent / (path.name + '.dvc')
                if dvc_file_path.exists():
                    logger.debug(f"Remove file {dvc_file_path} from DVC")
                    self.dvc_repo.remove(str(dvc_file_path))

                logger.debug(f"Add file {path} to DVC")
                self.dvc_repo.add(str(path))

                logger.debug(f"Set metadata to {stage_item.metadata}")
                set_dvc_file_metadata(dvc_file_path, stage_item.metadata)

                # Add .dvc file and .gitignore to index
                gitignore_path = dvc_file_path.parent / '.gitignore'
                self.git_repo.index.add([str(dvc_file_path), str(gitignore_path)])

                logger.debug(f"Push file {dvc_file_path} to DVC remote {self.dvc_remote}")
                self.dvc_repo.push(str(dvc_file_path), remote=self.dvc_remote)

            # Commit and push
            if self._need_commit(lock=False):
                identifiers = [stage_item.identifier for stage_item in self.dataset_stage]
                commit_message = f"Update {', '.join(identifiers)}"
                logger.debug(f"Create commit: {commit_message}")
                self.git_repo.index.commit(commit_message)
                logger.debug("Push to git repository")
                git_push(self.git_repo)
            else:
                logger.debug("No commit needed")

            self.dataset_stage = []

    @property
    def read_only(self):
        return self.commit_id is not None

    def _need_commit(self, lock=True) -> bool:
        with OptionalLock(self.lock, lock):
            if not self.git_repo.is_dirty():
                return False
            # We don't commit if the only files changed are .gitignore files. DVC sometimes reorders the lines in
            # .gitignore even though no file has changed.
            diffs = self.git_repo.index.diff(self.git_repo.head.commit)
            return any(os.path.basename(diff.a_path) != '.gitignore' for diff in diffs)


# FIXME: Make common base class instead of extending Repository
class StaticRepository(Repository):
    def __init__(self, config):
        self.datasets = {dataset['identifier']: dataset for dataset in config}

    def load_dataset(self, identifier, skip_pull_if_exists=False):
        import pandas as pd
        spec = self.datasets[identifier]
        # TODO: Index?
        df = pd.DataFrame(columns=spec['columns'], data=spec['data'])
        # TODO: units and metadata
        return Dataset(df, identifier, units=None, metadata=None)

    def has_dataset(self, identifier):
        return identifier in self.datasets

    def pull_datasets(self):
        pass

    def push_dataset(self, dataset):
        raise Exception("Cannot push dataset to static repository")

    def add(self, dataset):
        raise Exception("Cannot add to static repository")

    def push(self):
        raise Exception("Cannot push to static repository")
