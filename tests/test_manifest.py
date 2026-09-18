from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import dvc.repo
import fsspec
import polars as pl
import pytest
import yaml
from dvc_pandas import Dataset, DatasetLoader, DatasetManifest, Repository, RepositoryManifest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

from .helpers import git


@pytest.fixture
def source(tmp_path: Path) -> tuple[Repository, Path, pl.DataFrame]:
    workspace = tmp_path / "source"
    workspace.mkdir()
    git(workspace, "init", "-b", "main")
    git(workspace, "config", "user.email", "test@example.invalid")
    git(workspace, "config", "user.name", "Test")
    remote = tmp_path / "remote"
    remote.mkdir()
    frame = pl.DataFrame({"Year": [2020, 2021], "Value": [1.0, 2.0]})
    frame.write_parquet(workspace / "activity.parquet")
    with dvc.repo.Repo.init(str(workspace)) as dvc_repo:
        with dvc_repo.config.edit() as conf:
            conf["remote"]["storage"] = {"url": str(remote)}
            conf["core"]["remote"] = "storage"
        dvc_repo.add(str(workspace / "activity.parquet"))
        dvc_repo.push()
    metadata_path = workspace / "activity.parquet.dvc"
    metadata = yaml.safe_load(metadata_path.read_text())
    metadata["meta"] = {"units": {"Value": "kg"}, "index_columns": ["Year"], "source": {"title": "Example"}}
    metadata_path.write_text(yaml.safe_dump(metadata))
    git(workspace, "add", ".")
    git(workspace, "-c", "commit.gpgsign=false", "commit", "-m", "Dataset")
    repo = Repository(str(workspace), cache_root=str(tmp_path / "git-cache"))
    return repo, remote, frame


def test_manifest_load_without_git_and_offline_reuse(
    source: tuple[Repository, Path, pl.DataFrame], tmp_path: Path,
) -> None:
    repo, remote, frame = source
    regular = repo.load_dataset("activity")
    manifest = DatasetManifest.model_validate_json(regular.manifest.model_dump_json())
    repository_manifest = repo.get_manifest(["activity"])
    assert RepositoryManifest.model_validate_json(repository_manifest.model_dump_json()) == repository_manifest
    repo.dvc_repo.close()
    repo.git_repo.free()
    repo.repo_dir.rename(tmp_path / "unavailable-git-repository")
    loader = DatasetLoader(cache_root=tmp_path / "empty-cache")
    with patch("dvc_pandas.repository.Repository.__init__", side_effect=AssertionError("Git forbidden")):
        with patch.object(Dataset, "from_parquet", side_effect=AssertionError("Prefetch must not deserialize")):
            paths = loader.prefetch([manifest])
        for file in remote.rglob("*"):
            if file.is_file():
                file.unlink()
        loaded = loader.load(manifest)
    assert paths[0].is_file()
    assert_frame_equal(loaded.df, frame)
    assert loaded.meta == regular.meta
    assert loaded.copy().manifest == manifest
    assert loaded.manifest.metadata == {"source": {"title": "Example"}}


def test_corruption_is_not_published(source: tuple[Repository, Path, pl.DataFrame], tmp_path: Path) -> None:
    repo, _, _ = source
    manifest = repo.get_dataset_manifest("activity")
    remote_file = Path(manifest.object_url.removeprefix("file://"))
    remote_file.unlink()
    remote_file.write_bytes(b"corrupt")
    loader = DatasetLoader(cache_root=tmp_path / "cache")
    with pytest.raises(ValueError, match="Content hash mismatch"):
        loader.prefetch([manifest])
    assert not list((tmp_path / "cache").rglob(".download-*"))
    assert not (tmp_path / "cache" / manifest.content_hash[:2] / manifest.content_hash[2:]).exists()


def test_manifest_validation(source: tuple[Repository, Path, pl.DataFrame]) -> None:
    manifest = source[0].get_dataset_manifest("activity").model_dump()
    for field, value in [
        ("content_hash", "../escape"),
        ("version", 2),
        ("object_url", "https://host/file?secret=value"),
    ]:
        with pytest.raises(ValidationError):
            DatasetManifest.model_validate({**manifest, field: value})


def test_legacy_remote_layout(source: tuple[Repository, Path, pl.DataFrame], tmp_path: Path) -> None:
    repo, remote, frame = source
    manifest = repo.get_dataset_manifest("activity")
    modern = Path(manifest.object_url.removeprefix("file://"))
    legacy = remote / manifest.content_hash[:2] / manifest.content_hash[2:]
    legacy.parent.mkdir(parents=True)
    modern.rename(legacy)
    with patch.object(
        repo,
        "_get_dvc_metadata",
        return_value={
            "outs": [{"md5": manifest.content_hash, "path": "activity.parquet"}],
        },
    ):
        legacy_manifest = repo.get_dataset_manifest("activity")
    assert legacy_manifest.object_url == legacy.as_uri()
    loaded = DatasetLoader(cache_root=tmp_path / "legacy-cache").load(legacy_manifest)
    assert_frame_equal(loaded.df, frame)
    assert hashlib.md5(legacy.read_bytes(), usedforsecurity=False).hexdigest() == loaded.hash


def test_s3_manifest_keeps_credentials_runtime_only(
    source: tuple[Repository, Path, pl.DataFrame], tmp_path: Path,
) -> None:
    repo, _, frame = source
    local_manifest = repo.get_dataset_manifest("activity")
    content = Path(local_manifest.object_url.removeprefix("file://")).read_bytes()
    repo.dvc_repo.close()
    with dvc.repo.Repo(str(repo.repo_dir)) as dvc_repo, dvc_repo.config.edit() as config:
        config["remote"]["storage"] = {
            "url": "s3://example/datasets",
            "endpointurl": "https://s3.example.invalid",
            "region": "test-region",
            "access_key_id": "private-key",
            "secret_access_key": "private-secret",
        }
    # DVC reads repository configuration at the selected Git revision.
    git(repo.repo_dir, "add", ".dvc/config")
    git(repo.repo_dir, "-c", "commit.gpgsign=false", "commit", "-m", "S3 remote")
    repo._dvc_repo = None
    manifest = repo.get_dataset_manifest("activity")
    assert (
        manifest.object_url
        == f"s3://example/datasets/files/md5/{manifest.content_hash[:2]}/{manifest.content_hash[2:]}"
    )
    assert manifest.endpoint_url == "https://s3.example.invalid"
    assert "private-" not in manifest.model_dump_json()
    loader = DatasetLoader(
        cache_root=tmp_path / "s3-cache", storage_options={"key": "runtime-key", "secret": "runtime-secret"},
    )
    with patch("dvc_pandas.loader.fsspec.open", return_value=io.BytesIO(content)) as download:
        loaded = loader.load(manifest)
    download.assert_called_once_with(
        manifest.object_url,
        "rb",
        key="runtime-key",
        secret="runtime-secret",  # noqa: S106 - test credential
        client_kwargs={"endpoint_url": "https://s3.example.invalid", "region_name": "test-region"},
    )
    assert_frame_equal(loaded.df, frame)


def test_pinned_revision_preserves_metadata(source: tuple[Repository, Path, pl.DataFrame]) -> None:
    repo, _, _ = source
    first = repo.get_dataset_manifest('activity')
    path = repo.repo_dir / 'activity.parquet.dvc'
    data = yaml.safe_load(path.read_text())
    data['meta']['units']['Value'] = 'g'
    path.write_text(yaml.safe_dump(data))
    git(repo.repo_dir, 'add', 'activity.parquet.dvc')
    git(repo.repo_dir, '-c', 'commit.gpgsign=false', 'commit', '-m', 'Change units only')
    updated = repo.get_dataset_manifest('activity')
    assert updated.content_hash == first.content_hash
    assert updated.units != first.units
    assert updated.revision != first.revision
    repo.set_target_commit(first.revision)
    assert repo.get_dataset_manifest('activity') == first


def test_concurrent_prefetch_downloads_once(source: tuple[Repository, Path, pl.DataFrame], tmp_path: Path) -> None:
    manifest = source[0].get_dataset_manifest('activity')
    loaders = [DatasetLoader(cache_root=tmp_path / 'shared-cache') for _ in range(2)]
    with patch('dvc_pandas.loader.fsspec.open', wraps=fsspec.open) as download, ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda loader: loader.prefetch([manifest]), loaders))
    assert results[0] == results[1]
    assert download.call_count == 1
