"""
Tests the end-to-end workflow for the package
"""

from pipelinex import HatchDict
from pandas import DataFrame
from kedro.extras.datasets.pandas import CSVDataSet
from networkx import Graph
from eos.refinery.create_graph import GraphCreator
import yaml

def test_e2e() -> None:
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    input_key: str = "test_e2e"

    # Data access operations
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    csv_dataset: CSVDataSet = HatchDict(catalog[input_key]).get("csv_dataset")
    csv_data: DataFrame = csv_dataset.load()

    # Graph conversion
    gc_obj: GraphCreator = GraphCreator(df_input = csv_data)
    gc_obj.create_graph()
    G: Graph = gc_obj.graph
    assert G.nodes.data()
