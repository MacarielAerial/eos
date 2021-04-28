import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from networkx import Graph
from pandas import DataFrame

from eos.classes.feature_concatenator import FeatureConcatenator

log = logging.getLogger(__name__)


def link_nodes_with_no_graph(df: DataFrame, params: Dict[str, Any]) -> Graph:
    """
    Creates/Modify a NetworkX graph with a dataframe with edge tuples and edge attributes
    """
    g: Graph = nx.MultiDiGraph()
    log.info(f"Created a graph of type {g.__class__}")

    edge_tuple_src, edge_tuple_dst = params["edge_tuple_src"], params["edge_tuple_dst"]
    edge_attrs = list(df.drop([edge_tuple_src, edge_tuple_dst], axis=1).columns)
    log.info(
        f"({edge_tuple_src}, {edge_tuple_dst}) is edge tuple and "
        f"{edge_attrs} are edge attributes"
    )
    n_targets = params["n_targets"]
    e_targets = params["e_targets"]
    df["label"] = df[e_targets].apply(np.argmax, axis=1)
    log.info(
        f"Edge attribute 'label' has been created with {np.sum(df['label'])} positives"
    )
    g.graph.update({"n_targets": n_targets, "e_targets": e_targets})
    log.info(
        f"Added edge targets {e_targets} as global graph attributes and"
        f"added node targets {n_targets} as global graph attrbutes"
    )

    df_as_dict: Dict[str, Dict[str, float]] = df.to_dict(orient="index")
    ebunch: List[Tuple[float, float, Dict[str, float]]] = []
    for row in df_as_dict.values():
        u = row[edge_tuple_src]
        v = row[edge_tuple_dst]
        d = {
            d_k: d_v
            for d_k, d_v in row.items()
            if d_k not in set([edge_tuple_src, edge_tuple_dst])
        }
        edge_tuple = (u, v, d)
        ebunch.append(edge_tuple)
    log.info(f"Gathered {len(ebunch)} edge tuples from the dataframe")

    g.add_edges_from(ebunch)
    log.info(
        f"Post node linking graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges"
    )

    return g


def concat_features(g: Graph) -> Graph:
    """
    Concatenate all features, assumed to be numeric, into one feature
    """
    fe_obj: FeatureConcatenator = FeatureConcatenator(g_input=g)
    fe_obj.concat_n_attrs()
    fe_obj.concat_e_attrs()
    fe_obj.delete_originals()

    return fe_obj.graph
