"""
Ordinally encodes categorical features of a dataframe
as a preprocessing step for AutoEncoder later to embed categorical features
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder

log = logging.getLogger(__name__)


class PreprocessEncoder:
    def __init__(self, df_input: DataFrame) -> None:
        print(f"PreprocessEncoder: Initiated with dataframe of shape {df_input.shape}")
        self.df_input = df_input
        self.df_attrs: Dict[str, str] = df_input.attrs
        self._find_cat_subset_df()

        self.df_output = df_input.copy()

    def _find_cat_subset_df(self) -> None:
        self.df_cat_feats = self.df_input[self.df_attrs["cat_feats"]]

    def ordinal_encode(self) -> None:
        encoder: OrdinalEncoder = OrdinalEncoder()
        encoded_df: ndarray = encoder.fit_transform(self.df_cat_feats)
        self.df_output[self.df_attrs["cat_feats"]] = encoded_df
        self._categories = [category.tolist() for category in encoder.categories_]
        log.info(
            f"PreprocessEncoder: Ordinally encoded {len(self._categories)} features"
        )

    @property
    def df_encoded(self) -> DataFrame:
        return self.df_output

    @property
    def categories(self) -> Dict[str, List[str]]:
        columns: List[str] = list(self.df_cat_feats.columns)
        categories: Dict[str, List[str]] = dict(zip(columns, self._categories))
        return categories
