"""
gat_model.py — Graph Attention Network for Drug-Drug Interaction prediction.

Architecture:
  GATEncoder: 3 GATConv layers → global mean pool → 128-dim graph embedding
  GATDDIClassifier: two GATEncoders (shared weights) → concat → MLP → num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


class GATEncoder(nn.Module):
    """
    Encodes a molecular graph into a fixed-size embedding vector.

    Uses multi-head Graph Attention layers with residual connections
    and batch normalization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=True,
            )
        )
        self.bns.append(BatchNorm(hidden_channels * num_heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                )
            )
            self.bns.append(BatchNorm(hidden_channels * num_heads))

        # Final layer: single head, no concat
        self.convs.append(
            GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=1,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=False,
            )
        )
        self.bns.append(BatchNorm(hidden_channels))

        # Projection (for residual connections when dimensions change)
        self.input_proj = nn.Linear(in_channels, hidden_channels * num_heads)

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                return_attention=False):
        """
        Args:
            x:            node feature matrix          [num_nodes, in_channels]
            edge_index:   edge connectivity             [2, num_edges]
            edge_attr:    edge feature matrix (optional) [num_edges, edge_dim]
            batch:        batch assignment vector        [num_nodes]
            return_attention: if True, also return attention weights from the
                              last layer

        Returns:
            graph_emb: pooled graph embedding [batch_size, hidden_channels]
            (optionally) attention_weights from last layer
        """
        attention_weights = None

        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if return_attention and i == len(self.convs) - 1:
                h_new, (edge_idx_att, alpha) = conv(
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attention_weights = (edge_idx_att, alpha)
            else:
                h_new = conv(h, edge_index, edge_attr=edge_attr)

            h_new = bn(h_new)

            if i < len(self.convs) - 1:
                h_new = F.elu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            h = h_new

        # Global mean pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        graph_emb = global_mean_pool(h, batch)

        if return_attention:
            return graph_emb, attention_weights
        return graph_emb


class GATDDIClassifier(nn.Module):
    """
    Drug-Drug Interaction classifier using a shared GAT encoder.

    Takes two molecular graphs, encodes them with a shared GATEncoder,
    concatenates the embeddings, and classifies through an MLP.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_classes: int = 5,
        dropout: float = 0.2,
        edge_dim: int = None,
    ):
        super().__init__()

        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        # Classification MLP: concat(emb_a, emb_b) → hidden → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data_a, data_b, return_attention=False):
        """
        Args:
            data_a: Batch of molecular graphs for drug A
            data_b: Batch of molecular graphs for drug B
            return_attention: if True, return attention weights

        Returns:
            logits: [batch_size, num_classes]
            (optionally) (att_a, att_b) attention weight tuples
        """
        if return_attention:
            emb_a, att_a = self.encoder(
                data_a.x, data_a.edge_index,
                edge_attr=data_a.edge_attr, batch=data_a.batch,
                return_attention=True,
            )
            emb_b, att_b = self.encoder(
                data_b.x, data_b.edge_index,
                edge_attr=data_b.edge_attr, batch=data_b.batch,
                return_attention=True,
            )
        else:
            emb_a = self.encoder(
                data_a.x, data_a.edge_index,
                edge_attr=data_a.edge_attr, batch=data_a.batch,
            )
            emb_b = self.encoder(
                data_b.x, data_b.edge_index,
                edge_attr=data_b.edge_attr, batch=data_b.batch,
            )

        # Concatenate drug embeddings
        combined = torch.cat([emb_a, emb_b], dim=-1)
        logits = self.classifier(combined)

        if return_attention:
            return logits, (att_a, att_b)
        return logits
