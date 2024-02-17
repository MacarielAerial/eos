import logging
from typing import Any, Dict, List, Set, Tuple

from networkx import Graph

from eos.data_interfaces.edge_dfs_data_interface import EdgeAttrKey, EdgeDF, EdgeDFs
from eos.data_interfaces.node_dfs_data_interface import NodeAttrKey, NodeDF, NodeDFs

logger = logging.getLogger(__name__)


def validate_node_dfs_and_edge_dfs(node_dfs: NodeDFs, edge_dfs: EdgeDFs) -> None:
    # Collect all unique node ids
    set_nid: Set[int] = set()
    for node_df in node_dfs.members:
        df = node_df.df
        set_nid = set_nid | set(df[NodeAttrKey.nid.value])

    # Collect all unique node ids referenced in eids
    set_nid_referenced = set()
    for edge_df in edge_dfs.members:
        df = edge_df.df
        for u, v in df[EdgeAttrKey.eid.value].tolist():
            set_nid_referenced.add(u)
            set_nid_referenced.add(v)

    if set_nid != set_nid_referenced:  # Island nodes are not allowed
        raise ValueError(
            f"The set of {len(set_nid)} node ids in node datagrames and "
            f"{len(set_nid_referenced)} node ids in edge dataframes are "
            "not identical"
        )


def node_tuples_from_node_df(node_df: NodeDF) -> List[Tuple[int, Dict[str, Any]]]:
    df = node_df.df

    node_tuples: List[Tuple[int, Dict[str, Any]]] = []

    for record in df.to_dict(orient="records"):
        # e.g. [{"nid": 0, "ntype": ...}, ...]
        nid = int(record.pop(NodeAttrKey.nid.value))
        # The rest is assumed all to be attributes
        record_str_key = {str(k): v for k, v in record.items()}
        node_tuple: Tuple[int, Dict[str, Any]] = (nid, record_str_key)
        node_tuples.append(node_tuple)

    logger.info(
        f"Parsed {len(node_tuples)} node tuples from {node_df.ntype.value} "
        "node dataframe"
    )

    return node_tuples


def node_tuples_from_node_dfs(node_dfs: NodeDFs) -> List[Tuple[int, Dict[str, Any]]]:
    node_tuples: List[Tuple[int, Dict[str, Any]]] = []

    for node_df in node_dfs.members:
        node_tuples.extend(node_tuples_from_node_df(node_df=node_df))

    return node_tuples


def edge_tuples_from_edge_df(edge_df: EdgeDF) -> List[Tuple[int, int, Dict[str, Any]]]:
    df = edge_df.df

    edge_tuples: List[Tuple[int, int, Dict[str, Any]]] = []

    for record in df.to_dict(orient="records"):
        # e.g. [{"eid": (0, 1), "etype": ...}, ...]
        u, v = tuple(record.pop(EdgeAttrKey.eid.value))
        # The rest is assumed all to be attributes
        record_str_key = {str(k): v for k, v in record.items()}
        edge_tuple: Tuple[int, int, Dict[str, Any]] = (u, v, record_str_key)
        edge_tuples.append(edge_tuple)

    logger.info(
        f"Parsed {len(edge_tuples)} edge tuples from {edge_df.etype.value} "
        "edge dataframe"
    )

    return edge_tuples


def edge_tuples_from_edge_dfs(
    edge_dfs: EdgeDFs,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    edge_tuples: List[Tuple[int, int, Dict[str, Any]]] = []

    for edge_df in edge_dfs.members:
        edge_tuples.extend(edge_tuples_from_edge_df(edge_df=edge_df))

    return edge_tuples


def _assemble_kg(node_dfs: NodeDFs, edge_dfs: EdgeDFs) -> Graph:
    # Sanity check input
    validate_node_dfs_and_edge_dfs(node_dfs=node_dfs, edge_dfs=edge_dfs)

    # Transform graph elements into networkx graph compatible form
    node_tuples = node_tuples_from_node_dfs(node_dfs=node_dfs)
    edge_tuples = edge_tuples_from_edge_dfs(edge_dfs=edge_dfs)

    # Initialise the knowledge graph
    nx_g = Graph()
    nx_g.add_nodes_from(node_tuples)
    nx_g.add_edges_from(edge_tuples)

    logger.info(
        f"Initialised knowledge graph has {nx_g.number_of_nodes()} nodes "
        f"and {nx_g.number_of_edges()} edges"
    )

    return nx_g
