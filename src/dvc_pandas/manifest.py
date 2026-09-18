from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic evaluates this annotation
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetManifest(BaseModel):
    """Serializable source identity and metadata, without runtime credentials."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    repository_url: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    identifier: str
    hash_algorithm: Literal["md5"] = "md5"
    content_hash: str = Field(pattern=r"^[0-9a-f]{32}$")
    object_url: str
    endpoint_url: str | None = None
    region: str | None = None
    modified_at: datetime
    units: dict[str, str] | None = None
    index_columns: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("repository_url", "object_url", "endpoint_url")
    @classmethod
    def credential_free_url(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlsplit(value)
            if parsed.password or (parsed.username and parsed.scheme != "ssh") or parsed.query or parsed.fragment:
                raise ValueError("Manifest URLs must not contain credentials, queries or fragments")
        return value


class RepositoryManifest(BaseModel):
    """A resolved repository revision and its selected dataset sources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    repository_url: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    datasets: dict[str, DatasetManifest]

    @field_validator("repository_url")
    @classmethod
    def credential_free_url(cls, value: str) -> str:
        DatasetManifest.credential_free_url(value)
        return value

    @model_validator(mode="after")
    def consistent_sources(self) -> RepositoryManifest:
        for identifier, dataset in self.datasets.items():
            if (identifier, self.repository_url, self.revision) != (
                dataset.identifier,
                dataset.repository_url,
                dataset.revision,
            ):
                raise ValueError("Dataset identity must match its repository manifest")
        return self
