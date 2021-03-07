import pandas as pd
from pandas import DataFrame
from networkx import Graph
from eos.refinery.create_graph import GraphCreator

def test_df_to_graph() -> None:
    df: DataFrame = pd.DataFrame({"col_1": [1, 2, 3], "col_2": [4, 5, 6], "col_3": [7, 8, 9]})
    gc_obj: GraphCreator = GraphCreator(df_input = df)
    gc_obj.create_graph()
    G: Graph = gc_obj.graph
    assert False
    assert G.nodes.data()
    assert G.nodes[0]["col_1"] == 1
