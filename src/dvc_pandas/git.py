import git
from appdirs import user_cache_dir


CACHE_DIR = user_cache_dir('dvc-pandas', 'kausaltech')


def repo_path(url):
    scheme, path = url.split(':', 1)
    path = path.strip('/')
    return f'{scheme}/{path}'


def repo_cache_dir(url):
    return f'{CACHE_DIR}/{repo_path(url)}'


def get_repo(url):
    """Get repo from cache, clone it if it is not cached yet"""
    repo_dir = repo_cache_dir(url)
    try:
        repo = git.Repo(repo_dir)
    except git.exc.NoSuchPathError:
        repo = git.Repo.clone_from(url, repo_dir)
    return repo
