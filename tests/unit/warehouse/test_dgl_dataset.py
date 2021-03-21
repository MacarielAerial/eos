"""
Tests custom DGL data interface class
"""

import dgl
import torch
import yaml
from dgl.heterograph import DGLHeteroGraph
from pipelinex import HatchDict

from eos.warehouse.dgl_dataset import DGLDataSet

u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
G: DGLHeteroGraph = dgl.graph((u, v))


def test_dgl_dataset() -> None:
    # Test-specific parameter definitions
    path_catalog_yml: str = "tests/data/catalog.yml"
    catalog: dict = yaml.safe_load(open(path_catalog_yml, "r"))
    input_key: str = "test_dgl_dataset"

    # DGL Data access operations
    graph_dataset: DGLDataSet = HatchDict(catalog[input_key]).get("dgl_dataset")
    graph_dataset.save(G)
    glist, label_dict = graph_dataset.load()
    G_reloaded = glist[0]

    assert G_reloaded
