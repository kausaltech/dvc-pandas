import git
import hashlib
import os
from appdirs import user_cache_dir


CACHE_DIR = user_cache_dir('dvc-pandas', 'kausaltech')


def repo_path(url):
    scheme, path = url.split(':', 1)
    path = path.strip('/')
    path = hashlib.md5(path.encode('utf-8')).hexdigest()
    return f'{scheme}/{path}'


def repo_cache_dir(url):
    return f'{CACHE_DIR}/{repo_path(url)}'


def get_repo(url):
    """Get repo from cache, clone it if it is not cached yet"""
    try:
        os.mkdir(CACHE_DIR)
    except FileExistsError:
        pass
    repo_dir = repo_cache_dir(url)
    try:
        repo = git.Repo(repo_dir)
    except git.exc.NoSuchPathError:
        repo = git.Repo.clone_from(url, repo_dir)
    return repo
