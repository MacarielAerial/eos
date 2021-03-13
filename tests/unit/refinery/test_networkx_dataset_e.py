"""
Tests custom NetworkX data interface class
"""

from eos.refinery.create_nx_interface import NetworkXDataSetE
from pipelinex import HatchDict
import yaml

def test_networkx_dataset_e():
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_networkx_dataset_e"

    # NetworkX Data access operations
    graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("graph_dataset")
    assert graph_dataset
