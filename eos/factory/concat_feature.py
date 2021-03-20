"""
Concatenates all node features (assumed to be numeric at this point)
into one node attribute that has the type Pytorch Tensor
"""

from copy import deepcopy
from typing import Dict, List, Set, Union

import networkx as nx
import numpy as np
from networkx import Graph, MultiDiGraph
from numpy import ndarray


class FeatureConcatenator:
    def __init__(self, g_input: Graph) -> None:
        print(
            f"FeatureConcatenator: Initiating with a graph of {g_input.number_of_nodes()} nodes",
            f"and {g_input.number_of_edges()} edges",
        )
        self.g_input = g_input

        self.g = MultiDiGraph(deepcopy(g_input))
        self._obtain_attrs()
        self._init_nfeat_attr()

    def _obtain_attrs(self) -> None:
        """
        Obtains a list-formatted set of node attributes
        """
        attrs: Set[str] = {
            k_attr for nid, attrs in self.g.nodes.data() for k_attr in attrs.keys()
        }
        self.attrs: List[str] = list(attrs)
        print(
            "FeatureConcatenator: The following set of node attributes",
            f"is present in the graph:\n{self.attrs}",
        )

    def _init_nfeat_attr(self) -> None:
        """
        Creates an empty node attribute "nfeat"
        """
        print("FeatureConcatenator: Initiating target nfeat attribute with nulls")
        mapping_nfeat: Dict[Union[int, str], None] = {nid: None for nid in self.g.nodes}
        nx.set_node_attributes(self.g, mapping_nfeat, "nfeat")

    def encode_attrs(self) -> None:
        """
        Encodes a given list of attributes as continous variables into node attribute "nfeat"
        """
        print(
            f"FeatureConcatenator: Concatenating the following variables:\n{self.attrs}"
        )
        mapping_attrs: Dict[Union[int, str], ndarray] = {
            k: np.array([v[attr] for attr in self.attrs]).reshape(1, -1)
            for k, v in self.g.nodes.data()
        }
        nx.set_node_attributes(self.g, mapping_attrs, "nfeat")

    @property
    def graph(self) -> Graph:
        return self.g
