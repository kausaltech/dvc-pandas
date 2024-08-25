from __future__ import annotations

from copy import deepcopy
import dataclasses
from datetime import datetime
import json
from pathlib import Path
from typing import cast

import polars as pl
from pyarrow.parquet import core as pq
import pyarrow as pa


@dataclasses.dataclass
class DatasetMeta:
    identifier: str
    modified_at: datetime | None = None
    units: dict[str, str] | None = None
    index_columns: list[str] | None = None
    metadata: dict | None = None
    hash: str | None = None


class Dataset:
    identifier: str
    modified_at: datetime | None
    units: dict[str, str] | None
    index_columns: list[str] | None
    hash: str | None

    df: pl.DataFrame | None

    @classmethod
    def read_parquet_schema(cls, path: Path) -> pa.Schema:
        return pq.read_schema(str(path))

    @classmethod
    def update_meta_from_parquet(cls, schema: pa.Schema, meta: DatasetMeta) -> DatasetMeta:
        md = schema.metadata
        if not md:
            return meta

        pdmeta = schema.pandas_metadata or {}
        if meta.index_columns is None:
            index_columns = pdmeta.get('index_columns')
            if index_columns is not None:
                meta = dataclasses.replace(meta, index_columns=index_columns)

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
    def from_parquet(cls, path: Path, meta: DatasetMeta):
        schema = cls.read_parquet_schema(path)
        meta = cls.update_meta_from_parquet(schema, meta)
        try:
            pldf = pl.read_parquet(path)
        except Exception:
            pldf = pl.read_parquet(path, use_pyarrow=True)
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
        if units is not None:
            if df is not None:
                for column in units.keys():
                    if column not in df.columns:
                        raise ValueError(f"Unit specified for unknown column name '{column}'.")

        self.identifier = meta.identifier
        self.units = meta.units
        self.index_columns = meta.index_columns
        self.metadata = meta.metadata
        self.modified_at = meta.modified_at
        self.hash = meta.hash
        self.df = df

    @property
    def meta(self) -> DatasetMeta:
        return DatasetMeta(self.identifier, modified_at=self.modified_at, units=self.units, index_columns=self.index_columns, metadata=self.metadata, hash=self.hash)

    def copy(self):
        df = self.df
        if df is not None:
            df = df.clone()
        return Dataset(df, meta=deepcopy(self.meta))

    @property
    def dvc_metadata(self) -> dict | None:
        """
        Return the metadata as it should be stored in the .dvc file.

        Physical units will be stored as part of the metadata using the key `units`.
        """
        if self.metadata is None and self.units is None:
            return None
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata
        if self.units is not None:
            metadata['units'] = self.units
        return metadata

    def _replace_table_metadata(self, table: pa.Table, pandas_md: dict) -> pa.Table:
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

        table = pa.Table.from_pandas(df)
        pd_meta = table.schema.pandas_metadata
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

    def __str__(self):
        return self.identifier
