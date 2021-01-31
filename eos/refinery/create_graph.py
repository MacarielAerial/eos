from typing import Any, Dict, List

import networkx as nx
from networkx import Graph
from pandas import DataFrame


class GraphCreator(object):
    """
    Create a graph from a 2D array of tabular data
    """

    def __init__(self, df_input: DataFrame):
        self.df_input = df_input
        self.col_names: List[str] = list(df_input.columns)
        self.row_names: List[str] = list(df_input.index)

    def df_to_graph(self) -> Graph:
        G: Graph = Graph()
        # Populate ids for nodes
        [G.add_node(idx) for idx in self.row_names]
        # Populate attributes for nodes
        dict_nodes: Dict[str, Dict[str, Any]] = {}
        for row_name in self.row_names:
            dict_node_attrs: Dict[str, Any] = {}
            for col_name in self.col_names:
                dict_node_attr: Dict[str, Any] = {
                    col_name: self.df_input.loc[row_name, col_name]
                }
                dict_node_attrs.update(dict_node_attr)
        nx.set_node_attributes(G=G, values=dict_nodes)
        return G
