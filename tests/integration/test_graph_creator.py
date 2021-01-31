import pandas as pd
from networkx import Graph
from eos.refinery.create_graph import GraphCreator

def test_df_to_graph():
    df = pd.DataFrame()
    gc_obj = GraphCreator(df_input = df)
    G = gc_obj.df_to_graph()
    assert isinstance(G, Graph)
