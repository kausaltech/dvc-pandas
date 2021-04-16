import git
import hashlib
import logging
import os
from appdirs import user_cache_dir
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = user_cache_dir('dvc-pandas', 'kausaltech')
PUSH_SUCCEEDED_FLAGS = git.remote.PushInfo.FAST_FORWARD | git.remote.PushInfo.NEW_HEAD


def repo_cache_dir(url):
    """Return cache directory for the given URL."""
    dir_name = hashlib.md5(url.encode('utf-8')).hexdigest()
    return Path(CACHE_DIR) / dir_name


def local_repo_dir(location=None):
    """
    Return local repository directory for `location`.

    If `location` is a URL, returns the corresponding directory in the cache. If it is a path to a local directory,
    returns the directory.
    """
    if location is None:
        try:
            location = os.environ['DVC_PANDAS_REPOSITORY']
        except KeyError:
            raise Exception("No repository location provided and DVC_PANDAS_REPOSITORY not set")

    # Don't use cache if location is a local repository
    if Path(location).exists():
        return location

    return repo_cache_dir(location)


def get_repo(location=None):
    """
    Return git repository for the given location, which can either be a URL or a path to a local repository.

    If given a URL and the repository does not exist in the cache, clones it first into the cache.
    """
    repo_dir = local_repo_dir(location)
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
