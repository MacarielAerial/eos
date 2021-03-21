"""
Converts categorical variables in a table into numeric form through embedding
"""

from pandas import DataFrame


class TableEncoder:
    def __init__(self, df_input: DataFrame) -> None:
        self.df_input = df_input

        self.df_output: DataFrame = DataFrame()

    def split_cont_cat(self) -> None:
        pass

    def autoencode_cat_vars(self) -> None:
        pass

    def concat_cont_cat(self) -> None:
        pass

    @property
    def df(self) -> DataFrame:
        return self.df_output
