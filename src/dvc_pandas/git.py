import git
import hashlib
import os
from appdirs import user_cache_dir


CACHE_DIR = user_cache_dir('dvc-pandas', 'kausaltech')


def repo_path(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def repo_cache_dir(url):
    return f'{CACHE_DIR}/{repo_path(url)}'


def get_repo(url=None):
    """Get repo from cache, clone it if it is not cached yet"""
    if url is None:
        try:
            url = os.environ['DVC_PANDAS_REPOSITORY']
        except KeyError:
            raise Exception("No repository URL provided and DVC_PANDAS_REPOSITORY not set")

    repo_dir = repo_cache_dir(url)
    try:
        repo = git.Repo(repo_dir)
    except git.exc.NoSuchPathError:
        repo = git.Repo.clone_from(url, repo_dir)
    return repo
