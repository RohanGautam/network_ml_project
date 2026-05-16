import torch
from torch import nn
from .basemodel import MLP_gate, MLP
from torch.nn import functional as F
from .cross_hgt import CrossHGT


class HyperGraphTransformer(nn.Module):
    def __init__(self, hidden_size):
        super(HyperGraphTransformer, self).__init__()
        self.hidden_size    = hidden_size
        self.node_aggr_tf   = CrossHGT(in_dim=hidden_size, out_dim=hidden_size, num_types=6, num_relations=1, n_heads=8)
        self.edge_to_node_tf = CrossHGT(in_dim=hidden_size, out_dim=hidden_size, num_types=6, num_relations=1, n_heads=8)
        self.to_edge_feat   = MLP(hidden_size=hidden_size)

    def forward(self, node_feat, node_type_now, edge_pair_now, edge_type_now):
        node_feat_weighted = self.node_aggr_tf(node_feat, node_feat, node_type_now, edge_pair_now, edge_type_now)
        hyper_edge_feat    = self.to_edge_feat(node_feat_weighted)
        node_feat_new      = self.edge_to_node_tf(node_feat, hyper_edge_feat, node_type_now, edge_pair_now[[1, 0]], edge_type_now)
        return node_feat_new
