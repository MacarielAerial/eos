"""
Calls all relevant objects in an arbitary order
"""

from eos.dock.unload import Unloader
from eos.factory.create_graph import GraphCreator


def orchestrator(conn_str: str, table_name: str):
    # Unload data from SQL database
    ul_obj = Unloader(conn_str=conn_str)
    df = ul_obj.get_table(table_name=table_name)

    # Convert tabular data into graph data
    gc_obj = GraphCreator(df_input=df)
    G = gc_obj.df_to_graph()
    return G
