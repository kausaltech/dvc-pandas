import dvc.repo
import logging
from ruamel.yaml import YAML

from .git import push as git_push


logger = logging.getLogger(__name__)


def set_dvc_file_metadata(dvc_file_path, metadata=None):
    yaml = YAML()
    with open(dvc_file_path, 'rt') as file:
        data = yaml.load(file)
    if metadata is None:
        data.pop('meta', None)
    else:
        data['meta'] = metadata
    with open(dvc_file_path, 'w') as file:
        yaml.dump(data, file)


def add_file_to_repo(dataset_path, git_repo, dvc_repo=None, dvc_remote=None, metadata=None):
    if dvc_repo is None:
        dvc_repo = dvc.repo.Repo(git_repo.working_dir)

    try:
        # Remove old .dvc file (if it exists) and replace it with a new one
        dvc_file_path = dataset_path.parent / (dataset_path.name + '.dvc')
        if dvc_file_path.exists():
            logger.debug(f"Remove file {dvc_file_path} from DVC")
            dvc_repo.remove(str(dvc_file_path))

        logger.debug(f"Add file {dataset_path} to DVC")
        dvc_repo.add(str(dataset_path))

        logger.debug(f"Set metadata to {metadata}")
        set_dvc_file_metadata(dvc_file_path, metadata)

        logger.debug(f"Push file {dvc_file_path} to DVC remote {dvc_remote}")
        dvc_repo.push(str(dvc_file_path), remote=dvc_remote)

        # Stage .dvc file and .gitignore
        gitignore_path = dvc_file_path.parent / '.gitignore'
        git_repo.index.add([str(dvc_file_path), str(gitignore_path)])

        # Commit and push
        if git_repo.is_dirty():
            git_repo.index.commit(f'Update {dataset_path.name}')
            logger.debug("Push to git repository")
            git_push(git_repo)
        else:
            logger.debug("No changes to git repository")
    except Exception:
        # Restore original data
        # git reset --hard origin/master
        logger.debug("Hard-reset master branch to origin/master")
        git_repo.head.reset('origin/master', index=True, working_tree=True)

        logger.debug(f"Checkout DVC dataset {dataset_path}")
        dvc_repo.checkout(str(dataset_path), force=True)

        # Remove rolled back stuff that may be left on the remote
        # dvc_repo.gc(all_commits=True, cloud=True, remote=dvc_remote)
        # TODO: Before reinstating the previous line, make sure gc doesn't delete blobs for which we don't have a git
        # commit because our git repository is outdated.
        raise
    finally:
        # Remove old version (or new if we rolled back) from cache
        logger.debug("Collect garbage in local DVC repository")
        dvc_repo.gc(workspace=True)


def pull(file_path, repo_dir):
    dvc_repo = dvc.repo.Repo(repo_dir)
    dvc_repo.pull(str(file_path))
