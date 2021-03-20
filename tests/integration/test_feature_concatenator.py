"""
Tests whether FeatureConcatenator can correctly aggregate all assumed-to-be numeric attributes into one attribute
"""

import random
from typing import List, Dict, Any

import networkx as nx
from networkx import Graph

from eos.factory.concat_feature import FeatureConcatenator



def test_feature_concatenator():
    cont_attrs: List[str] = ["feat_1", "feat_2"]
    cat_attrs: List[str] = ["feat_3", "feat_4"]
    g_input: Graph = Graph({1: {}, 2: {}})
    node_attrs: Dict[int, Dict[str, Any]] = {
                           1: {"feat_1": 10.0, "feat_2": 20, "feat_3": 1, "feat_4": random.uniform(-1, 1)},
                           2: {"feat_1": 11.0, "feat_2": 21, "feat_3": 0, "feat_4": random.uniform(-1, 1)}
                           }
    nx.set_node_attributes(g_input, node_attrs)

    fe_obj: FeatureConcatenator = FeatureConcatenator(g_input = g_input)
    fe_obj.encode_attrs()

    assert fe_obj.graph
