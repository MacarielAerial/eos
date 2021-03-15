"""
Tests the end-to-end workflow for the package
"""

import yaml

from pipelinex import HatchDict
from pandas import DataFrame
from networkx import Graph

from eos.refinery.create_graph import GraphCreator
from eos.refinery.link_node import NodeLinker

from eos.warehouse.csv_dataset import CSVDataSetE
from eos.warehouse.networkx_dataset import NetworkXDataSetE

def test_e2e() -> None:
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_e2e"

    # CSV Data access operations
    node_dataset: CSVDataSetE = HatchDict(catalog[input_key]).get("node_dataset")
    node_data: DataFrame = node_dataset.load()

    edge_dataset: CSVDataSetE = HatchDict(catalog[input_key]).get("edge_dataset")
    edge_data: DataFrame = edge_dataset.load()

    # Node population
    gc_obj: GraphCreator = GraphCreator(df_input = node_data)
    gc_obj.create_graph()

    # Node: NetworkX Data access operations
    nx_node_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_node_dataset")
    nx_node_dataset.save(gc_obj.g)

    nx_node_g_reloaded: Graph = nx_node_dataset.load()

    # Edge linking
    nl_obj: NodeLinker = NodeLinker(g_input = nx_node_g_reloaded,
                                    df_input = edge_data)
    nl_obj.link_node()

    # Node with edge: NetworkX Data access operations
    nx_graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_graph_dataset")
    nx_graph_dataset.save(nl_obj.g)

    nx_graph_g_reloaded: Graph = nx_graph_dataset.load()

    # DGL Graph conversion

    assert nx_graph_g_reloaded.nodes.data()
    assert nx_graph_g_reloaded.edges.data()
