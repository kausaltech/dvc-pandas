from __future__ import annotations

import hashlib
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec  # type: ignore[import-untyped]
from filelock import FileLock

from .dataset import Dataset, DatasetMeta
from .utils import DEFAULT_CACHE_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .manifest import DatasetManifest


class DatasetLoader:
    """Load immutable source files without constructing a Git or DVC repository."""

    def __init__(self, cache_root: str | Path | None = None, storage_options: dict[str, Any] | None = None):
        self.cache_root = Path(cache_root) if cache_root is not None else Path(DEFAULT_CACHE_ROOT) / "dataset-objects"
        self.storage_options = deepcopy(storage_options or {})

    def prefetch(self, manifests: Iterable[DatasetManifest]) -> list[Path]:
        """Download missing files atomically, without deserializing Parquet."""
        return [self._ensure_file(manifest) for manifest in manifests]

    def _ensure_file(self, manifest: DatasetManifest) -> Path:
        path = self.cache_root / manifest.content_hash[:2] / manifest.content_hash[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            if path.is_file():
                return path
            options = deepcopy(self.storage_options)
            if manifest.object_url.startswith("s3://"):
                client = options.setdefault("client_kwargs", {})
                if manifest.endpoint_url is not None:
                    client.setdefault("endpoint_url", manifest.endpoint_url)
                if manifest.region is not None:
                    client.setdefault("region_name", manifest.region)
            fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
            try:
                digest = hashlib.md5(usedforsecurity=False)
                with os.fdopen(fd, "wb") as destination, fsspec.open(manifest.object_url, "rb", **options) as source:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
                        digest.update(block)
                if digest.hexdigest() != manifest.content_hash:
                    message = f"Content hash mismatch for {manifest.identifier}"
                    raise ValueError(message)
                Path(temporary).replace(path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return path

    def load(self, manifest: DatasetManifest) -> Dataset:
        path = self._ensure_file(manifest)
        meta = DatasetMeta(
            identifier=manifest.identifier,
            modified_at=manifest.modified_at,
            units=deepcopy(manifest.units),
            index_columns=deepcopy(manifest.index_columns),
            metadata=deepcopy(manifest.metadata),
            hash=manifest.content_hash,
            manifest=manifest,
        )
        return Dataset.from_parquet(path, meta=meta)
