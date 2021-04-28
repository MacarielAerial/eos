"""
Concatenates all node features (assumed to be numeric at this point)
into one node attribute that has the type Pytorch Tensor
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Set, Tuple, Union

import networkx as nx
import numpy as np
from networkx import Graph, MultiDiGraph

log = logging.getLogger(__name__)


class FeatureConcatenator:
    def __init__(self, g_input: Graph) -> None:
        log.info(
            f"FeatureConcatenator: Initiating with a graph of {g_input.number_of_nodes()} nodes "
            f"and {g_input.number_of_edges()} edges"
        )
        self.g_input = g_input

        self.g = MultiDiGraph(deepcopy(g_input))
        self._obtain_attrs()
        self._init_feat_attrs()

    def _obtain_attrs(self) -> None:
        """
        Obtains a list-formatted set of node and edge attributes
        """
        self.n_targets: Set[str] = (
            set(self.g.graph["n_targets"]) if self.g.graph["n_targets"] else set()
        )
        self.n_attrs: Set[str] = {
            k_attr
            for nid, attrs in self.g.nodes.data()
            for k_attr in attrs.keys()
            if k_attr not in self.n_targets
        }
        if "label" in self.n_attrs:
            self.n_attrs.remove("label")
        log.info(
            "FeatureConcatenator: The following set of node attributes "
            f"is present in the graph:\n{self.n_attrs}"
        )

        self.e_targets: Set[str] = (
            set(self.g.graph["e_targets"]) if self.g.graph["e_targets"] else set()
        )
        self.e_attrs: Set[str] = {
            k_attr
            for u, v, k, attrs in self.g.edges.data(keys=True)
            for k_attr in attrs.keys()
            if k_attr not in self.e_targets
        }
        if "label" in self.e_attrs:
            self.e_attrs.remove("label")
        log.info(
            "FeatureConcatenator: The following set of edge attributes "
            f"is present in the graph:\n{self.e_attrs}"
        )

    def _init_feat_attrs(self) -> None:
        """
        Creates an empty node attribute "nfeat" and an empty edge attribute "efeat"
        """
        log.info("FeatureConcatenator: Initiating target nfeat attribute with nulls")
        mapping_nfeat: Dict[Union[int, str], None] = {nid: None for nid in self.g.nodes}
        nx.set_node_attributes(self.g, mapping_nfeat, "nfeat")

        log.info("FeatureConcatenator: Initiating target efeat attribute with nulls")
        mapping_efeat: Dict[Any, Any] = {
            (u, v, k): None for u, v, k in self.g.edges(keys=True)
        }
        nx.set_edge_attributes(self.g, mapping_efeat, "efeat")

    def concat_n_attrs(self) -> None:
        """
        Encodes all node attributes as continous variables into attribute "nfeat"
        """
        log.info(
            f"FeatureConcatenator: Concatenating the following node attributes:\n{self.n_attrs}"
        )
        mapping_attrs: Dict[Union[int, str], List[Any]] = {
            k: np.array([v[attr] for attr in self.n_attrs]).reshape(1, -1).tolist()
            for k, v in self.g.nodes.data()
        }
        nx.set_node_attributes(self.g, mapping_attrs, "nfeat")

    def concat_e_attrs(self) -> None:
        """
        Encodes all edge attributes as continous variables into attribute "efeat"
        """
        log.info(
            "FeatureConcatenator: Concatenating the following edge attributes:\n"
            f"{self.e_attrs}"
        )
        mapping_attrs: Dict[Tuple[Any, Any, Any], List[Any]] = {
            (u, v, k): np.array([e[attr] for attr in self.e_attrs])
            .reshape(1, -1)
            .tolist()
            for u, v, k, e in self.g.edges.data(keys=True)
        }
        nx.set_edge_attributes(self.g, mapping_attrs, "efeat")

    def delete_originals(self) -> None:
        """
        Deletes original node and edge attributes after they have been concatenated
        """
        log.info(
            "FeatureConcatenator: Deleting original node attributes "
            f"{self.n_attrs.union(self.n_targets)} "
            f"and edge attributes {self.e_attrs.union(self.e_targets)}",
        )
        for n_attr in self.n_attrs.union(self.n_targets):
            for nid in self.g.nodes:
                del self.g.nodes[nid][n_attr]
        for e_attr in self.e_attrs.union(self.e_targets):
            for u, v, k in self.g.edges(keys=True):
                del self.g.edges[u, v, k][e_attr]

    @property
    def graph(self) -> Graph:
        return self.g
