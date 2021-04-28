# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPNN
# pylint: disable= no-member, arguments-differ, invalid-name

"""
Includes the defnition of a Graphical Neural Network (GNN)
"""

import logging

import torch
import torch.nn as nn
from dgl.nn.pytorch import Set2Set

from eos.classes.gcn_modules.mpnn import MPNNGNN

log = logging.getLogger(__name__)


# pylint: disable=W0221
class Model(nn.Module):
    """MPNN for regression and classification on graphs.
    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats=64,
        edge_hidden_feats=128,
        n_tasks=1,
        num_step_message_passing=6,
        num_step_set2set=6,
        num_layer_set2set=3,
    ):
        super(Model, self).__init__()

        self.gnn = MPNNGNN(
            node_in_feats=node_in_feats,
            node_out_feats=node_out_feats,
            edge_in_feats=edge_in_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )
        self.readout = Set2Set(
            input_dim=node_out_feats,
            n_iters=num_step_set2set,
            n_layers=num_layer_set2set,
        )
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks),
        )
        self.pred = MLPPredictor(node_out_feats, 1)

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.
        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        return self.pred(g, node_feats)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(torch.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]
