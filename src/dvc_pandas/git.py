import git
import hashlib
import logging
import os
from appdirs import user_cache_dir
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = user_cache_dir('dvc-pandas', 'kausaltech')
PUSH_SUCCEEDED_FLAGS = git.remote.PushInfo.FAST_FORWARD | git.remote.PushInfo.NEW_HEAD


def cache_dir_for_url(url, cache_root=None):
    """Return directory within the given cache_root directory for the given URL."""
    if cache_root is None:
        cache_root = DEFAULT_CACHE_ROOT

    dir_name = hashlib.md5(str(url).encode('utf-8')).hexdigest()
    return Path(cache_root) / dir_name


def local_cache_dir(location=None, cache_local_repository=False, cache_root=None):
    """
    Return local repository directory for `location`.

    If `location` is a URL, returns the corresponding directory in the cache.
    If it is a path to a local directory and cache_local_repository is False,
    returns the directory.
    """
    if location is None:
        try:
            location = os.environ['DVC_PANDAS_REPOSITORY']
        except KeyError:
            raise Exception("No repository location provided and DVC_PANDAS_REPOSITORY not set")

    # Don't use cache if location is a local repository
    if not cache_local_repository and Path(location).exists():
        return location

    return cache_dir_for_url(location, cache_root)


def get_cache_repo(location=None, cache_local_repository=False, cache_root=None) -> git.Repo:
    """
    Return git repository for the given location, which can either be a URL or a path to a local repository.

    If given a URL and the repository does not exist in the cache, clones it first into the cache.
    """
    repo_dir = local_cache_dir(location, cache_local_repository=cache_local_repository, cache_root=cache_root)
    try:
        repo = git.Repo(repo_dir)
    except git.exc.NoSuchPathError:
        logger.debug(f"Clone git repository {location} to {repo_dir}")
        repo = git.Repo.clone_from(location, repo_dir)
    return repo


def push(repo):
    """Push to remote; raise an exception if it failed."""
    result, = repo.remote().push()
    # We don't accept any flags except those in PUSH_SUCCEEDED_FLAGS, and there should be at least one of them
    success_flags = result.flags & PUSH_SUCCEEDED_FLAGS
    fail_flags = result.flags ^ success_flags
    if fail_flags or not success_flags:
        raise Exception(f"Push to git repository failed: {result.summary}")
