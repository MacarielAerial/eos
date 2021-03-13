"""
Tests the end-to-end workflow for the package
"""

import yaml

from pipelinex import HatchDict
from pandas import DataFrame
from kedro.extras.datasets.pandas import CSVDataSet
from networkx import Graph

from eos.refinery.create_graph import GraphCreator
from eos.refinery.create_nx_interface import NetworkXDataSetE

def test_e2e() -> None:
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_e2e"

    # CSV Data access operations
    csv_dataset: CSVDataSet = HatchDict(catalog[input_key]).get("csv_dataset")
    csv_data: DataFrame = csv_dataset.load()

    # Graph conversion
    gc_obj: GraphCreator = GraphCreator(df_input = csv_data)
    gc_obj.create_graph()

    # NetworkX Data access operations
    graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("graph_dataset")
    graph_dataset.save(gc_obj.graph)

    nx_g_reloaded: Graph = graph_dataset.load()

    assert nx_g_reloaded.nodes.data()
