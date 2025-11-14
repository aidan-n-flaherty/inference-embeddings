import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import conv

from torch_geometric.nn import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MultiAggregation,
    SAGEConv,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
    SetTransformerAggregation,
    PowerMeanAggregation,
    DeepGCNLayer,
    GINEConv,
    DenseSAGEConv
)

class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomGNN, self).__init__()
        self.prelayer1 = nn.Linear(input_dim, hidden_dim)
        self.prelayer2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelayer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer1 = SAGEConv(hidden_dim, hidden_dim, aggr=SoftmaxAggregation())
        self.layerH1 = SAGEConv(hidden_dim, hidden_dim, aggr=SoftmaxAggregation())
        self.layer2 = SAGEConv(hidden_dim, hidden_dim, aggr=SoftmaxAggregation())
        self.postlayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.postlayer2 = nn.Linear(hidden_dim, hidden_dim)
        self.postlayer3 = nn.Linear(hidden_dim, output_dim)

        self.edge_weight = nn.Linear(output_dim * 2, 1)

    def forward(self, feature_data, edge_info):
        x = self.prelayer1(feature_data).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.prelayer2(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.prelayer3(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer1(x, edge_info).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layerH1(x, edge_info).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x, edge_info).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.postlayer1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.postlayer2(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.postlayer3(x).tanh()

        return x

        """N, d = x.shape

        x_i = x.unsqueeze(1).expand(N, N, d)
        x_j = x.unsqueeze(0).expand(N, N, d)

        pairs = torch.cat((x_i, x_j), dim=-1)

        pairs_flat = pairs.reshape(N * N, -1)

        scores = self.edge_weight(pairs_flat).tanh()

        embedding = scores.view(N, N)

        embedding = embedding * (1 - torch.eye(N, device=x.device)) + torch.eye(N, device=x.device)

        return embedding"""
    
def fully_connected_edges(n_nodes, device):
    """Return edge_index for a fully connected directed graph WITHOUT self-loops."""
    src = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
    dst = torch.arange(n_nodes, device=device).repeat(n_nodes)
    
    mask = src != dst
    src, dst = src[mask], dst[mask]
    
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index