# dvc-pandas

## Loading from persisted source manifests

For pinned sources, resolve metadata once with Git available and persist the
Pydantic model as JSON:

```python
from dvc_pandas import Repository

repo = Repository(repo_url, dvc_remote="storage")
repo.set_target_commit(commit_sha)
manifest = repo.get_manifest(["transport/activity"])
stored_json = manifest.model_dump_json()
```

A later process can download and load the same datasets without constructing a
Git or DVC repository:

```python
from dvc_pandas import DatasetLoader, RepositoryManifest

manifest = RepositoryManifest.model_validate_json(stored_json)
loader = DatasetLoader(cache_root="/cache/dataset-objects")
loader.prefetch(manifest.datasets.values())  # Download only; no Parquet decoding.
dataset = loader.load(manifest.datasets["transport/activity"])
assert dataset.manifest == manifest.datasets["transport/activity"]
```

Manifest transport currently supports local filesystem and S3 DVC remotes,
including legacy and current DVC object layouts. Other remotes continue through
the existing Git-backed loader, without manifest provenance. Object URLs are
resolved by DVC when exporting a manifest; the downloader does not guess the
remote layout. Manifests contain full commit IDs, file hashes, metadata and
non-secret endpoint/region settings. Credentials are supplied through the normal
S3 environment/provider chain or `DatasetLoader(storage_options=...)` using
fsspec options; they are not serialized. Local remote paths must remain accessible
on the machine loading the manifest.

Downloads are checked against the manifest MD5, locked per content hash, and
published atomically. Existing local content-addressed files are trusted.
`cache_root` is the directory containing the two-character hash subdirectories.
Repository-backed loads use DVC's current local object directory; standalone
loaders default to `dataset-objects` under the dvc-pandas user cache. Pass the same
cache root to share files between loaders.

`Dataset.manifest` records source provenance and is also preserved by `copy()`;
it does not identify subsequent in-memory edits. Datasets constructed from a
DataFrame have no manifest unless one is explicitly supplied through
`DatasetMeta`. Effective loaded units and index columns retain the existing
fallback to metadata embedded in Parquet.

The manifest format is versioned. Repository manifests describe the requested
subset of datasets, not necessarily the entire repository. Callers own persistence,
refresh policy for moving references, and selection of runtime credentials.

Run the full test suite:

```sh
uv run pytest
```
