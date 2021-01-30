"""
Unloads tabular data from a SQL database
"""

import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


class Unloader:
    """
    Accesses a sql database, at the moment sqlite, and loads necessary data
    """

    def __init__(self, conn_str: str):
        self.engine = create_engine(conn_str)

    def get_table(self, table_name: str) -> DataFrame:
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        return df
