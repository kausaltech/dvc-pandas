from __future__ import annotations

import dataclasses
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Concatenate, cast

import dvc.api
import dvc.repo
import filelock
import pygit2
import yaml
from pygit2 import Blob, Commit, GitError, Repository as GitRepo
from pygit2.enums import CredentialType, RepositoryOpenFlag, ResetMode

from .dataset import Dataset, DatasetMeta
from .dvc import set_dvc_file_metadata
from .utils import get_cache_repo_dir

if TYPE_CHECKING:
    from collections.abc import Generator

    from dvc_data.hashfile.db.local import LocalHashFileDB
    from polars import DataFrame as PolarsDataFrame

logger = logging.getLogger(__name__)


class ReentrantLock:
    """Lock that keeps track of invocations."""

    lock: filelock.BaseFileLock

    def __init__(self, lock_file: str):
        self.lock = filelock.FileLock(lock_file, thread_local=True)


type RepoFuncT[**P, R] = Callable[Concatenate[Repository, P], R]


def ensure_repo_lock[**P, R](func: RepoFuncT[P, R]) -> RepoFuncT[P, R]:
    """Wrap a Repository class method with a lock."""

    @wraps(func)
    def acquire_lock(self: Repository, *args: P.args, **kwargs: P.kwargs) -> R:
        with self.lock.lock.acquire():
            return func(self, *args, **kwargs)

    return acquire_lock


class DatasetStageItem:
    identifier: str
    metadata: dict[str, Any] | None

    def __init__(self, dataset: Dataset):
        self.identifier = dataset.identifier
        self.metadata = dataset.dvc_metadata


class AuthenticationDetailsUnavailableError(Exception):
    pass


class GitRemoteCallbacks(pygit2.RemoteCallbacks):
    repo: Repository

    def __init__(self, repo: Repository):
        self.repo = repo
        super().__init__()

    def credentials(
        self, url: str, username_from_url: str | None, allowed_types: CredentialType,
    ) -> pygit2.UserPass | pygit2.Username | pygit2.Keypair:
        creds = self.repo.repo_creds
        if allowed_types & CredentialType.USERPASS_PLAINTEXT:
            if not creds.git_username or not creds.git_token:
                raise AuthenticationDetailsUnavailableError("Requested username and password, but they were not supplied")
            return pygit2.UserPass(creds.git_username, creds.git_token)
        if allowed_types & pygit2.enums.CredentialType.USERNAME:
            return pygit2.Username("git")
        if allowed_types & pygit2.enums.CredentialType.SSH_KEY:
            if not creds.git_ssh_public_key_file or not creds.git_ssh_private_key_file:
                raise AuthenticationDetailsUnavailableError("Requested username and password, but they were not supplied")
            return pygit2.Keypair("git", creds.git_ssh_public_key_file, creds.git_ssh_private_key_file, "")
        raise Exception("git requested unknown credentials: 0x%x" % allowed_types)


@dataclass
class RepositoryCredentials:
    git_username: str | None = None
    git_token: str | None = field(default=None, repr=False)
    git_ssh_public_key_file: str | None = None
    git_ssh_private_key_file: str | None = None


class Repository:
    dvc_remote: str | None
    repo_url: str
    git_remote_name: str = 'origin'
    git_default_branch: str
    target_commit_id: str | None
    dataset_stage: list[DatasetStageItem]
    git_repo: GitRepo
    lock: ReentrantLock
    _dvc_repo: dvc.repo.Repo | None
    repo_creds: RepositoryCredentials

    def __init__(  # noqa: PLR0913
        self, repo_url: str, dvc_remote: str | None = None,
        cache_prefix: str | None = None, cache_local_repository: bool = True,
        cache_root: str | None = None, repo_credentials: RepositoryCredentials | None = None,
    ):
        """
        Initialize repository.

        Clones git repository if it's not in the cache.
        """
        if dvc_remote is None:
            dvc_remote = os.environ.get('DVC_PANDAS_DVC_REMOTE')

        self.repo_url = repo_url
        self.dvc_remote = dvc_remote
        self.cache_local_repository = cache_local_repository
        self.repo_creds = dataclasses.replace(repo_credentials) if repo_credentials else RepositoryCredentials()
        self.repo_dir = get_cache_repo_dir(
            location=repo_url, prefix=cache_prefix, cache_local_repository=cache_local_repository,
            cache_root=cache_root,
        )
        self.log_debug("Local git repo: %s" % self.repo_dir)
        self.git_repo = self._get_cached_repo()
        for branch in ('master', 'main'):
            if branch in self.git_repo.branches:
                break
        else:
            raise Exception("Unable to detect the main branch name for git repo at %s" % self.git_repo.path)
        self.git_default_branch = branch
        self.dataset_stage = []
        self.target_commit_id = None
        self.lock = ReentrantLock(str(self.repo_dir / '.dvc-pandas.lock'))
        self._dvc_repo = None

    def _get_cached_repo(self) -> GitRepo:
        try:
            repo = GitRepo(str(self.repo_dir), flags=RepositoryOpenFlag.NO_SEARCH)
        except pygit2.GitError:
            self.log_info("Clone git repository to %s" % self.repo_dir)
            cloned_repo = pygit2.clone_repository(self.repo_url, str(self.repo_dir), callbacks=GitRemoteCallbacks(self))
            repo = cast(GitRepo, cloned_repo)
        return repo

    def log_debug(self, message: str):
        logger.debug('[%s] %s' % (self.repo_url, message))

    def log_info(self, message: str):
        logger.info('[%s] %s' % (self.repo_url, message))

    def acquire_lock(self):
        self.lock.lock.acquire()

    def release_lock(self):
        self.lock.lock.release()

    @property
    def dvc_repo(self) -> dvc.repo.Repo:
        repo = self._dvc_repo
        if repo is not None:
            return repo

        repo = dvc.repo.Repo(str(self.repo_dir), rev=self.target_commit_id)
        self._dvc_repo = repo
        return repo

    @contextmanager
    def _dvc_context(self, rev: str) -> Generator[None, None, None]:
        with self.dvc_repo.switch(rev):
            yield

    def _get_from_dvc_cache(self, identifier: str, metadata: dict[str, Any]) -> Path | None:
        repo_cache: LocalHashFileDB = self.dvc_repo.cache.repo
        legacy_cache: LocalHashFileDB = self.dvc_repo.cache.legacy
        assert len(metadata['outs']) == 1
        m = metadata['outs'][0]
        file_hash = m['md5']
        repo_cache_file = Path(repo_cache.get(file_hash).path)
        if repo_cache_file.exists():
            return repo_cache_file
        legacy_cache_file = Path(legacy_cache.get(file_hash).path)
        if legacy_cache_file.exists():
            return legacy_cache_file
        return None

    def _get_git_commit(self) -> Commit:
        target = self.target_commit_id or self.git_repo.head.target
        commit = self.git_repo.get(target)
        assert isinstance(commit, Commit)
        return commit

    def _get_dvc_metadata(self, identifier: str, commit: Commit | None = None) -> dict[str, Any]:
        if commit is None:
            commit = self._get_git_commit()

        base_path = Path(identifier)
        meta_path = base_path.with_suffix('.parquet.dvc')

        # Get metadata (including units) from .dvc file
        git_tree = commit.tree
        if str(meta_path) not in git_tree:
            raise Exception("File %s not in git repo %s (commit id %s)" % (meta_path, self.repo_url, commit.id))
        blob: Blob = cast(Blob, git_tree[str(meta_path)])
        return yaml.load(blob.data, yaml.CSafeLoader)

    def _load_datasets(self, identifiers: list[str]) -> list[Dataset]:
        datasets_to_pull = set()
        commit = self._get_git_commit()

        ds_file_meta: list[tuple[str, Path | None, dict[str, Any]]] = []

        for identifier in identifiers:
            dvc_data = self._get_dvc_metadata(identifier, commit)
            parquet_path = self._get_from_dvc_cache(identifier, dvc_data)
            if parquet_path is None:
                datasets_to_pull.add(identifier)
            else:
                # self.log_debug("Using cached file '%s' for '%s'" % (parquet_path, identifier))
                pass
            ds_file_meta.append((identifier, parquet_path, dvc_data))

        if datasets_to_pull:
            logger.info("Datasets [%s] (rev. %s) are missing from DVC cache; pulling" % (', '.join(datasets_to_pull), commit.id))
            with self.lock.lock.acquire():
                parquet_files = [str(Path(identifier).with_suffix('.parquet')) for identifier in datasets_to_pull]
                self.dvc_repo.fetch(targets=parquet_files, remote=self.dvc_remote, revs=[str(commit.id)])

        datasets: list[Dataset] = []
        for identifier, cached_parquet_path, dvc_data in ds_file_meta:
            mtime = datetime.fromtimestamp(commit.commit_time, tz=timezone(offset=timedelta(minutes=commit.commit_time_offset)))
            modified_at = mtime.astimezone(tz=UTC)

            if cached_parquet_path is None:
                parquet_path = self._get_from_dvc_cache(identifier, dvc_data)
                if parquet_path is None:
                    raise Exception("Unable to pull dataset: %s" % identifier)
            else:
                parquet_path = cached_parquet_path

            parquet_hash = dvc_data['outs'][0]['md5']
            metadata = dvc_data.get('meta')
            if metadata is None:
                units = None
                index_columns = None
            else:
                units = metadata.pop('units', None)
                index_columns = metadata.pop('index_columns', None)

            ds_meta = DatasetMeta(
                identifier=identifier, modified_at=modified_at, units=units,
                index_columns=index_columns, metadata=metadata, hash=parquet_hash,
            )
            ds = Dataset.from_parquet(parquet_path, meta=ds_meta)
            datasets.append(ds)
        return datasets

    def load_dataset(self, identifier: str) -> Dataset:
        """
        Load dataset with the given identifier from the given repository.
        """

        dss = self._load_datasets([identifier])
        return dss[0]

    def load_datasets(self, identifiers: list[str]) -> list[Dataset]:
        """Load dataset with the given identifier from the given repository."""

        dss = self._load_datasets(identifiers)
        return dss

    def load_dataframe(self, identifier: str) -> PolarsDataFrame:
        """Perform like `load_dataset`, but return just the DataFrame."""
        dataset = self.load_dataset(identifier)
        df = dataset.df
        assert df is not None
        return df

    def has_dataset(self, identifier: str) -> bool:
        """Check if a dataset with the given identifier exists."""

        commit = self._get_git_commit()
        if (identifier + '.parquet.dvc') in commit.tree:
            return True
        return False

    def is_dataset_cached(self, identifier: str) -> bool:
        """Check if a dataset with the given identifier can be loaded directly from the DVC cache directory."""
        metadata = self._get_dvc_metadata(identifier)
        return self._get_from_dvc_cache(identifier, metadata) is not None

    @ensure_repo_lock
    def pull_datasets(self) -> str | None:
        """Make sure the git repository is pulled to the latest version."""

        main_branch = self.git_repo.branches[self.git_default_branch]
        remote_name = main_branch.upstream.remote_name
        remote = self.git_repo.remotes[self.git_remote_name]
        self.log_info("Pull from git remote %s" % remote_name)
        remote.fetch(callbacks=GitRemoteCallbacks(self))
        if main_branch.upstream.target == main_branch.target:
            self.log_info("Already up-to-date")
            return str(main_branch.target)

        current_head = self.git_repo.head.target
        nr_ahead, _ = self.git_repo.ahead_behind(current_head, main_branch.upstream.target)
        if nr_ahead:
            raise Exception("Current HEAD is %d commits ahead of %s" % (nr_ahead, main_branch.upstream.branch_name))
        main_branch.set_target(main_branch.upstream.target)
        self.git_repo.reset(main_branch.target, ResetMode.MIXED)
        return str(main_branch.target)

    @ensure_repo_lock
    def push_dataset(self, dataset: Dataset):
        """
        Add the given dataset as a parquet file to DVC, create a commit and push it.

        Updates the git repository first. If something goes wrong, restores the repository and all files to the state
        of origin/master.

        This is a convenience method wrapping add() and push(). If the dataset stage (the one populated by add()) is not
        empty, this raises a ValueError.
        """
        if self.dataset_stage:
            raise ValueError("push_dataset was called with nonempty stage.")
        if self.read_only:
            raise ValueError("push_dataset was called while repository is read-only.")

        self.pull_datasets()

        prev_head = self.git_repo.head.target

        self.add(dataset)
        try:
            self.push()
        except Exception:
            # Restore original data
            self.git_repo.reset(prev_head, reset_type=ResetMode.MIXED)
            # Remove rolled back stuff that may be left on the remote
            # self.dvc_repo.gc(all_commits=True, cloud=True, remote=self.dvc_remote)
            # TODO: Before reinstating the previous line, make sure gc doesn't delete blobs for which we don't have
            # a git commit because our git repository is outdated.
            raise
        finally:
            # Remove old version (or new if we rolled back) from cache
            # logger.debug("Collect garbage in local DVC repository")
            # self.dvc_repo.gc(workspace=True, force=True)
            pass

    @ensure_repo_lock
    def add(self, dataset: Dataset):
        """Create or update parquet file from dataset, but do not create .dvc file, commit or push."""
        if self.read_only:
            raise ValueError("add was called while repository is read-only.")

        parquet_path = self.repo_dir / Path(dataset.identifier).with_suffix('.parquet')
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(parquet_path)
        self.dataset_stage.append(DatasetStageItem(dataset))

    @ensure_repo_lock
    def push(self):
        """Upload, commit and push the staged files and clear the stage."""
        if self.read_only:
            raise ValueError("add was called while repository is read-only.")
        index = self.git_repo.index
        for stage_item in self.dataset_stage:
            path = self.repo_dir / (stage_item.identifier + '.parquet')

            # Remove old .dvc file (if it exists) and replace it with a new one
            dvc_file_path = path.with_suffix('.parquet.dvc')
            if dvc_file_path.exists():
                self.log_debug(f"Remove file {dvc_file_path} from DVC")
                dvc_file_path.unlink()

            self.log_debug(f"Add file {path} to DVC")
            self.dvc_repo.add(targets=[str(path)], force=True, remote=self.dvc_remote)  # type: ignore  # pyright: ignore

            self.log_debug(f"Set metadata to {stage_item.metadata}")
            set_dvc_file_metadata(dvc_file_path, stage_item.metadata)

            # Add .dvc file and .gitignore to index
            gitignore_path = dvc_file_path.parent / '.gitignore'
            for fn in (dvc_file_path, gitignore_path):
                index.add(str(fn.relative_to(self.repo_dir)))
            self.log_debug(f"Push file {dvc_file_path} to DVC remote {self.dvc_remote}")
            self.dvc_repo.push(str(dvc_file_path), remote=self.dvc_remote)  # pyright: ignore

        index.write()
        # Commit and push
        if not self._need_commit():
            self.dataset_stage = []
            self.log_debug("No commit needed")
            return

        identifiers = [stage_item.identifier for stage_item in self.dataset_stage]
        commit_message = f"Update {', '.join(identifiers)}"
        self.log_debug(f"Create commit: {commit_message}")
        ref = self.git_repo.head.name
        tree = index.write_tree()
        diff_index = self.git_repo.diff(self.git_repo.head.target, tree)
        for delta in diff_index.deltas:
            if delta.status_char() not in ('A', 'M'):
                raise Exception("Invalid changes in the git commit (only additions and modifications allowed)")

        prev_head = self.git_repo.head.target
        signature = self.git_repo.default_signature
        commit = self.git_repo.create_commit(ref, signature, signature, commit_message, tree, [prev_head])
        main_branch = self.git_repo.branches[self.git_default_branch]
        main_branch.set_target(commit)
        remote = self.git_repo.remotes[self.git_remote_name]
        self.log_info("Push commit %s to %s" % (commit, remote.name))
        remote.push([main_branch.name], callbacks=GitRemoteCallbacks(self))
        self.dataset_stage = []
        self.git_repo.checkout(main_branch)

    @property
    def read_only(self):
        return self.target_commit_id is not None

    @ensure_repo_lock
    def _need_commit(self) -> bool:
        status = self.git_repo.status()
        if not status:
            return False
        # We don't commit if the only files changed are .gitignore files. DVC sometimes reorders the lines in
        # .gitignore even though no file has changed.
        return any(Path(fn).name != '.gitignore' for fn in status.keys())

    @property
    def commit_id(self) -> str:
        """Git commit ID that this object will operate on."""
        return str(self._get_git_commit().id)

    def set_target_commit(self, commit_id: str | None):
        """
        Set commit ID that this object will operate on.

        If `commit_id` is not None, the given commit ID will be checked out temporarily for most commands and the
        repository will be read-only. Set it to None to use the latest commit.
        """
        self._dvc_repo = None
        if commit_id is None:
            self.target_commit_id = None
            return
        if not self.git_repo.odb.exists(commit_id):
            remote = self.git_repo.remotes[self.git_remote_name]
            logger.info("Commit %s does not exist in cached copy; pulling from %s" % (commit_id, remote.url))
            try:
                with self.lock.lock.acquire():
                    remote.fetch([commit_id])
            except GitError:
                logger.exception("Unable to fetch commit %s from %s" % (commit_id, remote.url))
                raise
        self.target_commit_id = commit_id
