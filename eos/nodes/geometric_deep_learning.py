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

from eos.classes.gcn import Model
from eos.utils.dgl_utils import evaluate, sample_mask

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
    G = dgl_tuple[0][0]
    # Squeeze unnecessary dimensions
    G.edata["efeat"] = G.edata["efeat"].squeeze(1)
    G.ndata["nfeat"] = G.ndata["nfeat"].squeeze(1)
    node_in_feats: int = G.ndata["nfeat"].shape[1]
    edge_in_feats: int = G.edata["efeat"].shape[1]
    label: Tensor = G.edata["label"]
    train_mask: Tensor = G.edata["train_mask"]
    val_mask: Tensor = G.edata["val_mask"]
    test_mask: Tensor = G.edata["test_mask"]

    epochs: int = params["epochs"]
    edge_hidden_feats: int = params["edge_hidden_feats"]
    node_out_feats: int = params["node_out_feats"]

    model = Model(node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = BCEWithLogitsLoss()
    log.info("Training GCN model")
    model.train()
    for epoch in range(epochs):
        logits = model(G, G.ndata["nfeat"], G.edata["efeat"])
        loss = loss_fn(
            logits[train_mask].reshape(
                -1,
            ),
            label[train_mask],
        )

        train_acc = evaluate(
            model, G, G.ndata["nfeat"], G.edata["efeat"], label, train_mask
        )
        val_acc = evaluate(
            model, G, G.ndata["nfeat"], G.edata["efeat"], label, val_mask
        )
        test_acc = evaluate(
            model, G, G.ndata["nfeat"], G.edata["efeat"], label, test_mask
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        log.info(
            "epoch : {}/{}, loss = {:.6f}, train_acc = {:.6f}, "
            "val_acc = {:.6f}, test_acc = {:.6f}".format(
                epoch + 1, epochs, loss, train_acc, val_acc, test_acc
            )
        )
    log.info("GCN model training complete")

    return model
