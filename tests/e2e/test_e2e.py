"""
Tests the end-to-end workflow for the package
"""

import yaml

from pipelinex import HatchDict
from pandas import DataFrame
from kedro.extras.datasets.pandas import CSVDataSet
from networkx import Graph

from eos.refinery.create_graph import GraphCreator
from eos.refinery.link_node import NodeLinker
from eos.warehouse.networkx_dataset import NetworkXDataSetE

def test_e2e() -> None:
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_e2e"

    # CSV Data access operations
    node_dataset: CSVDataSet = HatchDict(catalog[input_key]).get("node_dataset")
    node_data: DataFrame = node_dataset.load()

    edge_dataset: CSVDataSet = HatchDict(catalog[input_key]).get("edge_dataset")
    edge_data: DataFrame = edge_dataset.load()

    # Graph conversion
    gc_obj: GraphCreator = GraphCreator(df_input = node_data)
    gc_obj.create_graph()

    # Node linking
    nl_obj: NodeLinker = NodeLinker(g = gc_obj.graph, df_input = edge_data, e_src = edge_dataset.e_src, e_dst = edge_dataset.e_dst)

    # NetworkX Data access operations
    graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_dataset")
    graph_dataset.save(nl_obj.g)

    nx_g_reloaded: Graph = graph_dataset.load()

    # DGL Graph conversion

    assert nx_g_reloaded.nodes.data()
