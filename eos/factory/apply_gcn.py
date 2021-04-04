"""
Trains a GCN and infers data with it
"""

import logging
import random
from typing import Any, Dict, List, Tuple

import torch
from dgl import from_networkx
from dgl.heterograph import DGLHeteroGraph
from networkx import Graph
from numpy import ndarray
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from eos.utils import evaluate, sample_mask
from eos.warehouse.gcn import Model

log = logging.getLogger(__name__)


def convert_nx_to_dgl(G: Graph) -> DGLHeteroGraph:
    """
    Convert NetworkX graph import DGL graph
    """
    return from_networkx(
        nx_graph=G, node_attrs=["nfeat"], edge_attrs=["efeat", "label"]
    )


def process_dgl(
    dgl_tuple: Tuple[List[DGLHeteroGraph], dict], params: Dict[str, float]
) -> DGLHeteroGraph:
    """
    Assign train/val/test masks for DGL graph
    """
    G: DGLHeteroGraph = dgl_tuple[0][0]
    train_ratio, val_ratio = params["train_ratio"], params["val_ratio"]
    number_of_edges: int = G.edata["efeat"].shape[0]

    idx_range: List[int] = list(range(number_of_edges))
    random.shuffle(idx_range)

    idx_train: List[int] = idx_range[: round(number_of_edges * train_ratio)]
    idx_val: List[int] = idx_range[
        round(number_of_edges * train_ratio) : round(
            number_of_edges * train_ratio + number_of_edges * val_ratio
        )
    ]
    idx_test: List[int] = idx_range[
        round(number_of_edges * train_ratio + number_of_edges * val_ratio) :
    ]

    train_mask: ndarray = sample_mask(idx_train, number_of_edges)
    val_mask: ndarray = sample_mask(idx_val, number_of_edges)
    test_mask: ndarray = sample_mask(idx_test, number_of_edges)
    log.info(
        "Generated train, val and test masks with numbers of examples "
        f"{sum(train_mask)}, {sum(val_mask)}, {sum(test_mask)}"
    )

    G.edata["train_mask"] = torch.tensor(data=train_mask, dtype=torch.bool)
    G.edata["val_mask"] = torch.tensor(data=val_mask, dtype=torch.bool)
    G.edata["test_mask"] = torch.tensor(data=test_mask, dtype=torch.bool)

    return G


def train_gcn(
    dgl_tuple: Tuple[List[DGLHeteroGraph], dict], params: Dict[str, Any]
) -> Model:
    """
    Trains a GCN model
    """
    epochs: int = params["epochs"]
    hidden_features: int = params["hidden_features"]
    out_features: int = params["out_features"]
    G = dgl_tuple[0][0]
    input_shape: int = G.ndata["nfeat"].shape[2]
    model = Model(input_shape, hidden_features, out_features)
    label: Tensor = G.edata["label"]
    train_mask: Tensor = G.edata["train_mask"]
    val_mask: Tensor = G.edata["val_mask"]

    opt = torch.optim.Adam(model.parameters())
    log.info("Training GCN model")
    model.train()
    for epoch in range(epochs):
        pred = model(G, G.ndata["nfeat"])
        print(f"Shape of final prediction result for this epoch: {pred.shape}")
        print(f"Final prediction result: {pred}")
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(torch.squeeze(pred[train_mask]), label[train_mask])
        acc = evaluate(model, G, G.ndata["nfeat"], label, val_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.info(
            "epoch : {}/{}, loss = {:.6f}, acc = {:.6f}".format(
                epoch + 1, epochs, loss, acc
            )
        )
    log.info("GCN model training complete")

    return model
