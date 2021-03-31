import pandas as pd
from pandas import DataFrame

from eos.refinery.create_graph import populate_nodes


def test_populate_nodes() -> None:
    df: DataFrame = pd.DataFrame(
        {
            "col_1": ["nid_1", "nid_2", "nid_3"],
            "col_2": ["attr_1_1_1", "attr_2_1_2", "attr_3_1_3"],
            "col_3": ["attr_1_2_1", "attr_2_2_2", "attr_3_2_3"],
        }
    )
    G = populate_nodes(df=df)
    assert G.nodes.data()
    assert G.nodes[0]["col_2"] == "attr_1_1_1"
