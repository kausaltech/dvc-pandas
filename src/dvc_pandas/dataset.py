from __future__ import annotations

import dataclasses
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self, cast

import polars as pl
import pyarrow as pa
from pyarrow.parquet import core as pq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from .manifest import DatasetManifest, IndexColumn


def index_column_names(index_columns: Sequence[IndexColumn] | None) -> list[str] | None:
    """
    Reduce index column entries to the column names they stand for.

    Arrow describes an unnamed pandas RangeIndex with a descriptor dict rather than a column
    name. A named descriptor stands for that column; an unnamed one stands for none.
    """
    if index_columns is None:
        return None
    names: list[str] = []
    for column in index_columns:
        if isinstance(column, str):
            names.append(column)
        elif isinstance(column.get('name'), str):
            names.append(cast('str', column['name']))
    return names


@dataclasses.dataclass
class DatasetMeta:
    identifier: str
    modified_at: datetime | None = None
    units: dict[str, str] | None = None
    index_columns: list[IndexColumn] | None = None
    metadata: dict[str, Any] | None = None
    hash: str | None = None
    manifest: DatasetManifest | None = None


class Dataset:
    identifier: str
    modified_at: datetime | None
    units: dict[str, str] | None
    index_columns: list[IndexColumn] | None
    hash: str | None
    manifest: DatasetManifest | None
    metadata: dict[str, Any] | None

    df: pl.DataFrame | None

    @classmethod
    def read_parquet_schema(cls, path: Path) -> pa.Schema:
        return cast(pa.Schema, pq.read_schema(str(path)))  # pyright: ignore

    @classmethod
    def update_meta_from_parquet(cls, schema: pa.Schema, meta: DatasetMeta) -> DatasetMeta:
        md = schema.metadata
        if not md:
            return meta

        pdmeta = schema.pandas_metadata or {}
        if meta.index_columns is None:
            index_columns = pdmeta.get('index_columns')
            if index_columns is not None:
                names = cast('list[IndexColumn] | None', index_column_names(index_columns))
                meta = dataclasses.replace(meta, index_columns=names)

        if meta.units is None:
            units = {}
            cols = pdmeta.get('columns') or []
            for col in cols:
                col_md = col.get('metadata')
                if not isinstance(col_md, dict):
                    continue
                if 'unit' in col_md:
                    units[col['name']] = col_md['unit']
            if units:
                meta = dataclasses.replace(meta, units=units)

        return meta

    @classmethod
    def from_parquet(cls, path: Path, meta: DatasetMeta) -> Self:
        schema = cls.read_parquet_schema(path)
        meta = cls.update_meta_from_parquet(schema, meta)
        try:
            pldf = pl.read_parquet(path)
        except Exception:
            pldf = pl.read_parquet(path, use_pyarrow=True)
        # Arrow may store a named RangeIndex only in metadata, not in the file columns.
        for index in (schema.pandas_metadata or {}).get('index_columns', []):
            if isinstance(index, dict) and index.get('kind') == 'range':
                name = index.get('name')
                if name is not None and name not in pldf.columns:
                    pldf = pldf.with_columns(pl.Series(name, range(index['start'], index['stop'], index['step'])))
        return cls(pldf, meta)

    def __init__(self, df: pl.DataFrame | None, meta: DatasetMeta):
        """
        Create a dataset from a Pandas DataFrame, an identifier and optional metadata.

        If `units` is specified, it should be a dict that maps (some of) the columns of `df` to physical units. If any
        key of this dict is not a column in `df`, a ValueError is raised. The DataFrame in this dataset will use
        instances of PintArray of the respective unit for each column specified in `units`.

        You can specify metadata to be stored in the .dvc file by setting the `metadata` parameter to a dict.  Units
        will be stored in the metadata using the key `units`, so the `metadata` dict is not allowed to contain this key
        and a ValueError will be raised if it does.
        """

        metadata = meta.metadata
        if metadata and 'units' in metadata:
            raise ValueError("Dataset metadata may not contain the key 'units'.")
        units = meta.units
        if units is not None and df is not None:
            for column in units.keys():
                if column not in df.columns:
                    msg = f"Unit specified for unknown column name '{column}'."
                    raise ValueError(msg)

        self.identifier = meta.identifier
        self.units = meta.units
        # Older .dvc files may carry Arrow's RangeIndex descriptors; only names are meaningful here.
        self.index_columns = cast('list[IndexColumn] | None', index_column_names(meta.index_columns))
        self.metadata = meta.metadata
        self.modified_at = meta.modified_at
        self.hash = meta.hash
        self.manifest = meta.manifest
        self.df = df

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            self.identifier,
            modified_at=self.modified_at,
            units=self.units,
            index_columns=self.index_columns,
            metadata=self.metadata,
            hash=self.hash,
            manifest=self.manifest,
        )

    def copy(self):
        df = self.df
        if df is not None:
            df = df.clone()
        return Dataset(df, meta=deepcopy(self.meta))

    @property
    def dvc_metadata(self) -> dict[str, Any] | None:
        """
        Return the metadata as it should be stored in the .dvc file.

        Physical units will be stored as part of the metadata using the key `units`, and the
        index columns, as column names, using the key `index_columns`, so that a reader learns a
        dataset's shape from the .dvc file without fetching the data. A dataset with a frame always
        records them; an empty list says the data has no index columns.
        """
        recorded = self._index_columns_to_record()
        if self.metadata is None and self.units is None and recorded is None:
            return None
        metadata = dict(self.metadata) if self.metadata is not None else {}
        if self.units is not None:
            metadata['units'] = self.units
        if recorded is not None:
            metadata['index_columns'] = recorded
        return metadata

    def _index_columns_to_record(self) -> list[str] | None:
        own = index_column_names(self.index_columns)
        stated = None
        if self.metadata is not None and 'index_columns' in self.metadata:
            stated = index_column_names(self.metadata['index_columns'])
            if own is not None and stated != own:
                msg = f'metadata index_columns {stated} disagree with the dataset index columns {own}.'
                raise ValueError(msg)
        recorded = own if own is not None else stated
        if self.df is None:
            return recorded
        if recorded is None:
            # The frame is written without an index, which is what an empty list records.
            return []
        missing = [column for column in recorded if column not in self.df.columns]
        if missing:
            msg = f'Index columns {missing} are not columns of the dataset.'
            raise ValueError(msg)
        return recorded

    def _replace_table_metadata(self, table: pa.Table, pandas_md: dict[str, Any]) -> pa.Table:
        assert table.schema.metadata is not None
        new_md = deepcopy(table.schema.metadata)
        new_md[b'pandas'] = json.dumps(pandas_md).encode('utf8')
        new_md[b'dvc-pandas'] = json.dumps(self.metadata).encode('utf8') if self.metadata is not None else b'{}'
        table = table.replace_schema_metadata(cast(dict[str | bytes, str | bytes], new_md))
        return table

    def to_parquet(self, path: Path):
        assert self.df is not None

        df = self.df.to_pandas()
        if self.index_columns:
            df = df.set_index(self.index_columns)

        table = pa.Table.from_pandas(df, preserve_index=bool(self.index_columns))
        pd_meta: dict[str, Any] = table.schema.pandas_metadata
        assert pd_meta is not None
        if self.units:
            for col_name, unit in self.units.items():
                for pd_col in pd_meta['columns']:
                    if pd_col['name'] == col_name:
                        col_md = pd_col.get('metadata', None)
                        if not isinstance(col_md, dict):
                            col_md = {}
                            pd_col['metadata'] = col_md
                        col_md['unit'] = unit
                        break
                else:
                    raise Exception('Column %s referred to in units was not found' % col_name)

        table = self._replace_table_metadata(table, pd_meta)
        pq.write_table(table, str(path), compression='snappy')

    def __str__(self) -> str:
        return self.identifier
