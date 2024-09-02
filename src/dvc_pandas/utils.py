from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from appdirs import user_cache_dir
from pygit2 import GitError, Repository as GitRepo

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = user_cache_dir('dvc-pandas', 'kausaltech')


def cache_dir_for_url(url, prefix: str | None = None, cache_root: str | None = None):
    """Return directory within the given cache_root directory for the given URL."""
    if cache_root is None:
        cache_root = DEFAULT_CACHE_ROOT

    dir_name = hashlib.md5(str(url).encode('utf-8'), usedforsecurity=False).hexdigest()
    p = Path(cache_root)
    if prefix:
        p /= prefix
    return p / dir_name


def local_cache_dir(
    location: str | None = None, prefix: str | None = None, cache_local_repository: bool = False,
    cache_root: str | None = None,
) -> Path:
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
            raise Exception("No repository location provided and DVC_PANDAS_REPOSITORY not set") from None

    # Don't use cache if location is a local repository
    repo_dir = Path(location)
    if not cache_local_repository and repo_dir.exists():
        return repo_dir

    return cache_dir_for_url(location, prefix=prefix, cache_root=cache_root)


def get_cache_repo_dir(location: str, prefix: str | None = None, cache_local_repository: bool = False, cache_root=None) -> Path:
    """
    Return git repository for the given location, which can either be a URL or a path to a local repository.

    If given a URL and the repository does not exist in the cache, clones it first into the cache.
    """
    repo_dir = local_cache_dir(location, prefix=prefix, cache_local_repository=cache_local_repository, cache_root=cache_root)
    return repo_dir
