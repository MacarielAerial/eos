"""
Tests custom DGL data interface class
"""

import yaml
import tempfile

from eos.warehouse.dgl_dataset import DGLDataSet
from pipelinex import HatchDict
import torch
import dgl
from dgl.heterograph import DGLHeteroGraph

u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
G: DGLHeteroGraph = dgl.graph((u, v))

def test_dgl_dataset():
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_dgl_dataset"

    # DGL Data access operations
    graph_dataset: DGLDataSet = HatchDict(catalog[input_key]).get("dgl_dataset")
    graph_dataset.save(G)
    G_reloaded = graph_dataset.load()
    assert G_reloaded
