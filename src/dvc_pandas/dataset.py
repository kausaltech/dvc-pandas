from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from pint_pandas import PintArray


def _quantify_series(series, unit):
    if unit:
        return PintArray(series, unit)
    return series


class Dataset:
    df: pd.DataFrame
    identifier: str
    modified_at: datetime
    units: Optional[Dict[str, str]]
    metadata: Optional[Dict]

    def __init__(
        self, df: pd.DataFrame, identifier: str, modified_at: datetime, units: Dict[str, str] = None,
        metadata: Dict = None
    ):
        """
        Create a dataset from a Pandas DataFrame, an identifier and optional metadata.

        If `units` is specified, it should be a dict that maps (some of) the columns of `df` to physical units. If any
        key of this dict is not a column in `df`, a ValueError is raised. The DataFrame in this dataset will use
        instances of PintArray of the respective unit for each column specified in `units`.

        You can specify metadata to be stored in the .dvc file by setting the `metadata` parameter to a dict.  Units
        will be stored in the metadata using the key `units`, so the `metadata` dict is not allowed to contain this key
        and a ValueError will be raised if it does.
        """
        if metadata and 'units' in metadata:
            raise ValueError("Dataset metadata may not contain the key 'units'.")
        if units is not None:
            for column in units.keys():
                if column not in df.columns:
                    raise ValueError(f"Unit specified for unknown column name '{column}'.")

        if units:
            self.df = pd.DataFrame({
                column: _quantify_series(df[column], units.get(column))
                for column in df.columns
            })
        else:
            self.df = df.copy()

        self.identifier = identifier
        self.units = units
        self.metadata = metadata
        self.modified_at = modified_at

    def copy(self):
        return Dataset(self.df, self.identifier, units=self.units, metadata=self.metadata)

    @property
    def dvc_metadata(self) -> Optional[Dict]:
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

    def equals(self, other: pd.DataFrame) -> bool:
        compare_attrs = ('identifier', 'units', 'metadata')
        return self.df.equals(other.df) and all(getattr(self, a) == getattr(other, a) for a in compare_attrs)

    def __str__(self):
        return self.identifier
