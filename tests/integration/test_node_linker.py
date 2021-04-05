"""
Tests whether node linkers can parse a csv file into edge ids and attributes
into a NetworkX graph
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
from networkx import Graph, MultiDiGraph
from pandas import DataFrame

from eos.refinery.create_graph import connect_nodes


def test_df_to_edge() -> None:
    node_container: List[Tuple[int, Dict[str, Any]]] = [
        (1, {"col_0": 1}),
        (2, {"col_0": 2}),
        (3, {"col_0": 3}),
        (4, {"col_0": 4}),
        (5, {"col_0": 5}),
        (6, {"col_0": 6}),
    ]
    G_input: Graph = MultiDiGraph()
    G_input.add_nodes_from(node_container)
    df: DataFrame = pd.DataFrame(
        {
            "col_1": [1, 2, 3],
            "col_2": [4, 5, 6],
            "col_3": [7, 8, 9],
            "col_4": [10, 11, 12],
        }
    )
    df.attrs = {
        "edge_src": "col_1",
        "edge_dst": "col_2",
        "node": "col_0",
        "n_targets": None,
        "e_targets": ["col_4"],
    }

    G: Graph = connect_nodes(G=G_input, df=df)

    assert G.nodes.data()
    assert G.edges[1, 4, 0]["col_3"] == 7
