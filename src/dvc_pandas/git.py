import git
from appdirs import user_cache_dir


def repo_dir_from_url(url):
    return url.replace('https://', '', 1)


def get_updated_repo(repo_url):
    cache_dir = user_cache_dir('dvc-pandas', 'kausaltech')
    repo_dir = f'{cache_dir}/{repo_dir_from_url(repo_url)}'
    try:
        repo = git.Repo(repo_dir)
    except git.exc.NoSuchPathError:
        repo = git.Repo.clone_from(repo_url, repo_dir)
    else:
        repo.remote('origin').pull()
    return repo
