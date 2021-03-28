"""
Converts categorical variables in a table into numeric form through embedding
"""

from typing import List

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from torch import Tensor


class TableEncoder:
    def __init__(self, df_input: DataFrame) -> None:
        print(f"TableEncoder: Initiating with a dataframe of shape {df_input.shape},",
              f"with categorical features {df_input.attrs['cat_feats']}",
              f"and with continuous features {df_input.attrs['cont_feats']}")
        self.df_input = df_input
        self.df_attrs = df_input.attrs

        self.df_output: DataFrame = DataFrame()

    def split_cat_cont(self) -> None:
        cat_arrays: List[ndarray] = []
        cont_arrays: List[ndarray] = []

        for cat_feat in self.df_attrs["cat_feats"]:
            comp_array = self.df_input[cat_feat].to_numpy().reshape(-1, 1)
            cat_arrays.append(comp_array)
        self.cat_array = np.concatenate(*[cat_arrays], axis = 1)
        print(f"TableEncoder: Loaded categorical feature array of shape {self.cat_array.shape}")

        for cont_feat in self.df_attrs["cont_feats"]:
            comp_array = self.df_input[cont_feat].to_numpy().reshape(-1, 1)
            cont_arrays.append(comp_array)
        self.cont_array = np.concatenate(*[cont_arrays], axis = 1)
        print(f"TableEncoder: Loaded continuous feature array of shape {self.cont_array.shape}")

    def autoencode_cat_vars(self) -> None:
        pass

    def concat_cont_cat(self) -> None:
        pass

    @property
    def df(self) -> DataFrame:
        return self.df_output
