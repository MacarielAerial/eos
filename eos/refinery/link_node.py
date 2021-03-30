"""
Attaches links between nodes according to a Pandas DataFrame
"""

from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from networkx import Graph, MultiDiGraph
from networkx_query import search_nodes
from pandas import DataFrame, Series

log = logging.getLogger(__name__)


def check_metadata(df: DataFrame) -> None:
    """
    Checks if necessary metadata exists in input dataframe
    """
    metadata_keys: Set[str] = {"edge_src", "edge_dst", "node"}
    exist_keys: Set[str] = set(df.attrs.keys())
    if metadata_keys.issubset(exist_keys):
        raise ValueError(
            f"Input dataframe missing required metadata keys {metadata_keys.difference(exist_keys)} in pandas.DataFrame.attrs"
        )


def connect_nodes(G: Graph, df: DataFrame) -> Graph:
    """
    Populates a graph with edges based on a dataframe
    """
    print(
        f"NodeLinker: Initiating with a graph of {G.number_of_nodes()} nodes",
        f"and a dataframe of shape {df.shape}",
    )
    check_metadata(df=df)
    edge_src, edge_dst, node = (
        df.attrs["edge_src"],
        df.attrs["edge_dst"],
        df.attrs["node"],
    )
    ebunch: List[Tuple[int, int]] = []
    for i, row in df.iterrows():
        src_name: str = row[edge_src]
        dst_name: str = row[edge_dst]
        src_nid: int = search_nodes(G, {"==": [(node,), src_name]})
        dst_nid: int = search_nodes(G, {"==": [(node,), dst_name]})
        series_e_attrs: Series = row.drop(labels=[edge_src, edge_dst])
        e_attrs: Dict[str, float] = series_e_attrs.to_dict()
        ebunch.append((src_nid, dst_nid, e_attrs))
    print(f"NodeLinker: Adding {len(ebunch)} edges to the graph")
    G.add_edges_from(ebunch)
    return G


class NodeLinker:
    """
    Populates a graph with edges based on a dataframe
    """

    def __init__(self, g_input: Graph, df_input: DataFrame) -> None:
        print(
            f"NodeLinker: Initiating with a graph of {g_input.number_of_nodes()} nodes",
            f"and a dataframe of shape {df_input.shape}",
        )
        self.g_input = g_input
        self.df_input = df_input

        self._check_metadata()

        self.g = MultiDiGraph(deepcopy(g_input))
        self.df_mod = deepcopy(df_input)

    def _add_eids(self) -> None:
        """
        Creates empty edges
        """
        container_eids = Utils.cols_to_e_tuples(
            e_src=list(self.df_input[self.e_src]),
            e_dst=list(self.df_input[self.e_dst]),
            keys=list(self.df_input.index),
        )
        print(f"NodeLinker: Adding {len(container_eids)} edges to the graph")
        self.g.add_edges_from(container_eids)

    def _add_attr_keys(self) -> None:
        """
        Creates empty edge attribute dictionaries
        """
        container_attr_keys: List[str] = self.e_attrs
        container_attrs_null: Dict[str, None] = dict.fromkeys(container_attr_keys)
        print(
            "NodeLinker: Adding the following list of edge attributes "
            f"with null values to the graph:\n{list(container_attrs_null.keys())}"
        )
        mapping_eid_attr: Dict[Any, Dict[str, None]] = {
            (u, v, k): container_attrs_null for u, v, k in self.g.edges
        }
        nx.set_edge_attributes(self.g, mapping_eid_attr)

    def _add_attr_vals(self) -> None:
        """
        Populates edge attribue dictionaries with values
        """
        print(
            f"NodeLinker: Populating {len(self.g.out_edges)} edges "
            f"with {len(self.e_attrs)} attributes each in the graph "
            f"with {np.prod(self.df_input.loc[:, self.e_attrs].shape)} "
            "cells from the dataframe"
        )
        df_input_attrs: DataFrame = self.df_input.set_index(
            [self.e_src, self.e_dst, self.df_input.index]
        )
        n_vals_added: int = 0
        for u, v, k, attrs in self.g.edges.data(keys=True):
            for k_attr in attrs.keys():
                v_attr: Any = df_input_attrs.loc[(u, v, k), k_attr]
                self.g.edges[u, v, k][k_attr] = v_attr
                n_vals_added += 1
        print(
            f"NodeLinker: {n_vals_added} edge attribute values are appended "
            "from the dataframe to the graph"
        )

    def _agg_edges(self) -> None:
        """
        Aggregates multiple edges between the same pair of nodes together into one edge
        """
        print(
            "NodeLinker: Edge aggregation is disabled to preserve MultiDiGraph structure"
        )
        pass

    def _check_metadata(self) -> None:
        """
        Checks if necessary metadata exists in input dataframe
        """
        metadata_keys: Set[str] = {"edge_src", "edge_dst"}
        if not metadata_keys <= self.df_input.attrs.keys():
            raise ValueError(
                f"Input dataframe missing required metadata keys {metadata_keys} in pandas.DataFrame.attrs"
            )
        else:
            self.e_src: str = self.df_input.attrs["edge_src"]
            self.e_dst: str = self.df_input.attrs["edge_dst"]
            self.e_attrs: List[str] = [
                col
                for col in self.df_input.columns
                if col not in (self.e_src, self.e_dst)
            ]
            print(
                f"NodeLinker: {self.e_src} column is the edge source and {self.e_dst} is the edge destination"
            )
            print(
                f"NodeLinker: The following list of columns is the edge attribute columns:\n{self.e_attrs}"
            )

    def link_node(self) -> None:
        print("NodeLinker: Linking nodes with the supplied DataFrame")
        self._add_eids()
        self._add_attr_keys()
        self._add_attr_vals()
        self._agg_edges()
        print("NodeLinker: Created the graph accessible by property 'graph'")

    @property
    def graph(self) -> Graph:
        return self.g


class Utils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def cols_to_e_tuples(
        e_src: List[Any], e_dst: List[Any], keys: List[Any]
    ) -> List[Tuple[Any, Any, Any]]:
        """
        Converts two lists of edge id column arrays and a list of index array
        to a list of edge id tuples
        """
        ebunch: List[Tuple[Any, Any, Any]] = []
        for u, v, k in zip(e_src, e_dst, keys):
            eid: Tuple[Any, Any, Any] = (u, v, k)
            ebunch.append(eid)
        return ebunch
