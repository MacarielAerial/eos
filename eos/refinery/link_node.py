"""
Attaches links between nodes according to a Pandas DataFrame
"""

from networkx import Graph
from pandas import DataFrame


class NodeLinker:
    """
    Populates a graph with edges based on a dataframe
    """

    def __init__(self, g: Graph, df_input: DataFrame, e_src: str, e_dst: str) -> None:
        print(
            f"NodeLinker: Initiating with a graph of {g.number_of_nodes()} nodes ",
            "and a dataframe of shape (df_input.shape)",
        )
        self.g = g
        self.df_input = df_input
        self.e_src = e_src
        self.e_dst = e_dst
