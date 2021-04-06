import git
import hashlib
import logging
import os
from appdirs import user_cache_dir
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = user_cache_dir('dvc-pandas', 'kausaltech')


def repo_path(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def repo_cache_dir(url):
    return f'{CACHE_DIR}/{repo_path(url)}'


def get_repo(url=None):
    if url is None:
        try:
            url = os.environ['DVC_PANDAS_REPOSITORY']
        except KeyError:
            raise Exception("No repository URL provided and DVC_PANDAS_REPOSITORY not set")

    # Don't use cache if url is a local repository
    if Path(url).exists():
        return git.Repo(url)

    repo_dir = repo_cache_dir(url)
    try:
        repo = git.Repo(repo_dir)
    except git.exc.NoSuchPathError:
        logger.debug(f"Clone git repository {url} to {repo_dir}")
        repo = git.Repo.clone_from(url, repo_dir)
    return repo
