"""
Converts a Pandas DataFrame into a node-only graph
"""

import logging
from typing import Dict, List, Set, Tuple

import networkx as nx
from networkx import Graph, MultiDiGraph
from networkx_query import search_nodes
from pandas import DataFrame, Series

log = logging.getLogger(__name__)


def populate_nodes(df: DataFrame) -> Graph:
    """
    Initiates a graph with nodes inferred from a dataframe
    """
    log.info(f"GraphCreator: Initiatng a graph from a dataframe of shape {df.shape}")
    G: Graph = MultiDiGraph()

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


def check_metadata(df: DataFrame) -> None:
    """
    Checks if necessary metadata exists in input dataframe
    """
    metadata_keys: Set[str] = {"edge_src", "edge_dst", "node"}
    exist_keys: Set[str] = set(df.attrs.keys())
    if not metadata_keys.issubset(exist_keys):
        raise ValueError(
            "Input dataframe missing required metadata keys "
            f"{metadata_keys.difference(exist_keys)} in pandas.DataFrame.attrs"
        )


def connect_nodes(G: Graph, df: DataFrame) -> Graph:
    """
    Populates a graph with edges based on a dataframe
    """
    log.info(
        f"NodeLinker: Initiating with a graph of {G.number_of_nodes()} nodes "
        f"and a dataframe of shape {df.shape}",
    )
    check_metadata(df=df)
    edge_src, edge_dst, node = (
        df.attrs["edge_src"],
        df.attrs["edge_dst"],
        df.attrs["node"],
    )
    ebunch: List[Tuple[int, int, Dict[str, float]]] = []
    for i, row in df.iterrows():
        src_name: str = row[edge_src]
        dst_name: str = row[edge_dst]
        src_nid: int = list(search_nodes(G, {"==": [(node,), src_name]}))[0]
        dst_nid: int = list(search_nodes(G, {"==": [(node,), dst_name]}))[0]
        series_e_attrs: Series = row.drop(labels=[edge_src, edge_dst])
        e_attrs: Dict[str, float] = series_e_attrs.to_dict()
        ebunch.append((src_nid, dst_nid, e_attrs))
    log.info(f"NodeLinker: Adding {len(ebunch)} edges to the graph")
    G.add_edges_from(ebunch)
    return G
