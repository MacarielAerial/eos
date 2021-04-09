"""
Includes the defnition of a Graphical Neural Network (GNN)
"""

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import EdgeConv, GATConv, GraphConv
from torch.nn import Dropout


class Model(nn.Module):
    def __init__(
        self, n_in_features, e_in_features, hidden_features, out_features, **kwargs
    ):
        super().__init__()
        self.gcn = GCN(
            n_in_features, e_in_features, hidden_features, out_features, **kwargs
        )
        self.pred = DotProductPredictor()

    def forward(self, g, x_n, x_e):
        h = self.gcn(g, x_n, x_e)
        return self.pred(g, h)


class GCN(nn.Module):
    def __init__(self, n_in_feats, e_in_feats, hid_feats, out_feats, **kwargs):
        super().__init__()
        self.n_conv1 = GraphConv(in_feats=n_in_feats, out_feats=int(hid_feats/2))
        self.n_conv2 = GraphConv(in_feats=int(hid_feats/2), out_feats=int(out_feats/2))
        self.e_conv1 = EdgeConv(in_feat=e_in_feats, out_feat=int(hid_feats/2))
        self.e_conv2 = EdgeConv(in_feat=int(hid_feats/2), out_feat=int(out_feats/2))

        self.dropout = Dropout(p=kwargs["dropout"])

    def forward(self, graph, n_inputs, e_inputs):
        # inputs are features of nodes and edges
        h_n = self.n_conv1(graph, n_inputs)
        h_n = self.dropout(h_n)
        h_n = F.relu(h_n)
        h_n = self.n_conv2(graph, h_n)

        h_e = self.e_conv1(graph, e_inputs)
        h_e = self.dropout(h_e)
        h_e = F.relu(h_e)
        h_e = self.e_conv2(graph, h_e)

        h = torch.cat((h_n, h_e), 1)
        return h


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return graph.edata["score"]
