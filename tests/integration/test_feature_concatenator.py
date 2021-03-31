"""
Tests whether FeatureConcatenator can correctly aggregate
all assumed-to-be numeric attributes into one attribute
"""

import logging
import random
from typing import Any, Dict

import networkx as nx
from networkx import MultiDiGraph

from eos.warehouse.feature_concatenator import FeatureConcatenator

log = logging.getLogger(__name__)


def test_feature_concatenator():
    # Generate dummy data
    g_input: MultiDiGraph = MultiDiGraph(
        {0: {1: {"weight": 0.2}}, 1: {0: {"weight": 0.8}}}
    )
    g_input.add_edge(0, 1, weight=0.3)
    node_attrs: Dict[int, Dict[str, Any]] = {
        0: {"feat_1": 10.0, "feat_2": 20, "feat_3": 1, "feat_4": random.uniform(-1, 1)},
        1: {"feat_1": 11.0, "feat_2": 21, "feat_3": 0, "feat_4": random.uniform(-1, 1)},
    }
    nx.set_node_attributes(g_input, node_attrs)

    # Logic
    fe_obj: FeatureConcatenator = FeatureConcatenator(g_input=g_input)
    fe_obj.concat_n_attrs()
    fe_obj.concat_e_attrs()
    fe_obj.delete_originals()

    assert fe_obj.graph
