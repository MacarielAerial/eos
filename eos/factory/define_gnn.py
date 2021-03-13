"""
Includes the defnition of a Graphical Neural Network (GNN)
"""

import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, out_feats: int) -> None:
        super().__init__()
        self.conv1: SAGEConv = dgl.nn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type="mean"
        )
        self.conv2: SAGEConv = dgl.nn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type="mean"
        )

    def forward(self, graph, inputs) -> SAGEConv:
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
