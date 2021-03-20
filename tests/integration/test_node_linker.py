"""
Tests whether node linkers can parse a csv file into edge ids and attributes
into a NetworkX graph
"""

import pandas as pd
from pandas import DataFrame
from networkx import Graph
from eos.refinery.link_node import NodeLinker

def test_df_to_edge() -> None:
    g_input: Graph = Graph({1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}})
    df: DataFrame = pd.DataFrame({"col_1": [1, 2, 3], "col_2": [4, 5, 6], "col_3": [7, 8, 9]})
    df.attrs = {"edge_src": "col_1", "edge_dst": "col_2"}

    nl_obj: NodeLinker = NodeLinker(g_input = g_input, df_input = df)
    nl_obj.link_node()

    G: Graph = nl_obj.graph

    assert G.nodes.data()
    assert G.edges[1, 4, 0]["col_3"] == 7
