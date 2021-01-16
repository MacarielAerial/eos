"""
Utility functions including IO operations
"""

import pandas as pd
from pandas import DataFrame

def csv_to_df(csv_path: str) -> DataFrame:
    """
    Return a DataFrame object by loading a csv file
    """
    df: DataFrame = pd.read_csv(filepath_or_buffer = csv_path)
    return df
