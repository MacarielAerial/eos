"""
Converts a Pandas DataFrame into a node-only graph
"""

import logging
from typing import Any, Dict, List, Set

import networkx as nx
from networkx import Graph
from pandas import DataFrame

log = logging.getLogger(__name__)


def populate_nodes(df: DataFrame) -> Graph:
    """
    Initiates a graph with nodes inferred from a dataframe
    """
    log.info(f"GraphCreator: Initiatng a graph from a dataframe of shape {df.shape}")
    G: Graph = Graph()

    log.info(f"GraphCreator: Adding {len(df.index)} nodes to the graph")
    G.add_nodes_from(df.index.tolist())

    log.info(
        f"GraphCreator: Populating {G.number_of_nodes()} nodes "
        f"with {len(df.columns)} attributes each in the graph "
        f"with {df.shape[0] * df.shape[1]} cells from the dataframe"
    )
    mapping_nid_attr: List[Dict[str, float]] = df.to_dict("index")
    nx.set_node_attributes(G, mapping_nid_attr)

    return G


class GraphCreator(object):
    """
    Create a graph from a pandas DataFrame
    """

    def __init__(self, df_input: DataFrame):
        self.df_input = df_input
        print(
            f"GraphCreator: Initiatng a graph from a dataframe of shape {self.df_input.shape}"
        )
        self.g: Graph = Graph()

        self._check_metadata()

    def _add_nids(self) -> None:
        """
        Creates empty nodes
        """
        container_nids: List[Any] = list(self.df_input[self.node_id])
        print(f"GraphCreator: Adding {len(container_nids)} nodes to the graph")
        self.g.add_nodes_from(container_nids)

    def _add_attr_keys(self) -> None:
        """
        Creates empty node attribute dictionaries
        """
        container_attr_keys: List[str] = list(self.n_attrs)
        container_attrs_null: Dict[str, None] = dict.fromkeys(container_attr_keys)
        print(
            "GraphCreator: Adding the following list of node attributes "
            f"with null values to the graph:\n{list(container_attrs_null.keys())}"
        )
        mapping_nid_attr: Dict[Any, Dict[str, None]] = {
            nid: container_attrs_null for nid in self.g.nodes
        }
        nx.set_node_attributes(self.g, mapping_nid_attr)

    def _add_attr_vals(self) -> None:
        """
        Populates node attribue dictionaries with values
        """
        print(
            f"GraphCreator: Populating {self.g.number_of_nodes()} nodes "
            f"with {len(self.df_input.columns)} attributes each in the graph "
            f"with {self.df_input.shape[0] * self.df_input.shape[1]} cells from the dataframe"
        )
        df_input_attrs: DataFrame = self.df_input.set_index(self.node_id)
        n_vals_added: int = 0
        for nid, attrs in self.g.nodes.data():
            for k_attr in attrs.keys():
                v_attr: Any = df_input_attrs.loc[nid, k_attr]
                self.g.nodes[nid][k_attr] = v_attr
                n_vals_added += 1
        print(
            f"GraphCreator: {n_vals_added} node attribute values are appended "
            "from the dataframe to the graph"
        )

    def _check_metadata(self) -> None:
        """
        Checks if necessary metadata exists in input dataframe
        """
        metadata_keys: Set[str] = {"node_id"}
        if not metadata_keys <= self.df_input.attrs.keys():
            raise ValueError(
                f"Input dataframe missing required metadata keys {metadata_keys} in pandas.DataFrame.attrs"
            )
        else:
            self.node_id: str = self.df_input.attrs["node_id"]
            self.n_attrs: List[str] = [
                col for col in self.df_input.columns if col != self.node_id
            ]
            print(f"GraphCreator: {self.node_id} column is the node id source")
            print(
                f"GraphCreator: The following list of columns is the node attribute columns:\n{self.n_attrs}"
            )

    def create_graph(self) -> None:
        print("GraphCreator: Creating the graph with the supplied DataFrame")
        self._add_nids()
        self._add_attr_keys()
        self._add_attr_vals()
        print("GraphCreator: Created the graph accessible by property 'graph'")

    @property
    def graph(self) -> Graph:
        return self.g
