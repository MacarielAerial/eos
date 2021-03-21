import pandas as pd
from networkx import Graph
from pandas import DataFrame

from eos.refinery.create_graph import GraphCreator


def test_df_to_graph() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "col_1": ["nid_1", "nid_2", "nid_3"],
            "col_2": ["attr_1_1_1", "attr_2_1_2", "attr_3_1_3"],
            "col_3": ["attr_1_2_1", "attr_2_2_2", "attr_3_2_3"],
        }
    )
    df.attrs = {"node_id": "col_1"}
    gc_obj: GraphCreator = GraphCreator(df_input=df)
    gc_obj.create_graph()
    G: Graph = gc_obj.graph
    assert G.nodes.data()
    assert G.nodes["nid_1"]["col_2"] == "attr_1_1_1"
