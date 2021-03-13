"""
Unloads tabular data from a source (e.g. a .csv file)
"""

import pandas as pd
from pandas import DataFrame


class Unloader:
    """
    Converts data into different forms (e.g. from a .csv file to a pandas DataFrame)
    """

    def __init__(self) -> None:
        pass

    def csv_to_df(self, src: str) -> DataFrame:
        df: DataFrame = pd.read_csv(src)
        return df
