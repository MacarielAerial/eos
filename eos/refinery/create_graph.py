from typing import Any, Dict, List

import networkx as nx
from networkx import Graph
from pandas import DataFrame


class GraphCreator(object):
    """
    Create a graph from a pandas DataFrame
    """

    def __init__(self, df_input: DataFrame):
        self.df_input = df_input
        print(
            f"GraphCreator: Initiatng a graph from a DataFrame of shape {self.df_input.shape}"
        )
        self.g: Graph = Graph()

    def _add_nids(self) -> None:
        container_nids: List[Any] = list(self.df_input.index)
        print(f"GraphCreator: Adding {len(container_nids)} nodes to the graph")
        self.g.add_nodes_from(container_nids)

    def _add_attr_keys(self) -> None:
        container_attr_keys: List[str] = list(self.df_input.columns)
        container_attrs_null: Dict[str, None] = dict.fromkeys(container_attr_keys)
        print(
            f"GraphCreator: Adding the following list of node attributes with null values to the graph:\n{container_attrs_null}"
        )
        mapping_nid_attr: Dict[Any, Dict[str, None]] = {
            nid: container_attrs_null for nid in self.g.nodes
        }
        nx.set_node_attributes(self.g, mapping_nid_attr)

    def create_graph(self) -> None:
        print("GraphCreator: Creating the graph with the supplied DataFrame")
        self._add_nids()
        self._add_attr_keys()
        print("GraphCreator: Created the graph accessible by property 'graph'")

    @property
    def graph(self) -> Graph:
        return self.g
