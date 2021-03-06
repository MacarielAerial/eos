"""
Tests custom NetworkX data interface class
"""

import networkx as nx
import yaml
from networkx import Graph
from pipelinex import HatchDict

from eos.warehouse.networkx_dataset import NetworkXDataSetE

G: Graph = nx.complete_graph(10)


def test_networkx_dataset_e():
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_networkx_dataset_e"

    # NetworkX Data access operations
    graph_dataset: NetworkXDataSetE = HatchDict(catalog[input_key]).get("nx_dataset")
    graph_dataset.save(G)
    G_reloaded = graph_dataset.load()
    assert G_reloaded
