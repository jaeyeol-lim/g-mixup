"""Four-layer GINE backbone for DrugOOD G-Mixup."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool


class GMixupGIN(nn.Module):
    def __init__(self, node_dim=39, edge_dim=10, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, batch):
        x = self.node_encoder(batch.x.float())
        edge_attr = batch.edge_attr.float()
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x, batch.edge_index, edge_attr))
            if index + 1 < len(self.convs):
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(global_add_pool(x, batch.batch))
