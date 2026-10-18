import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNN(torch.nn.Module):
    """Two-layer GNN. kind in {"gcn", "sage", "gat"}. Outputs 2 class logits.

    in_dropout: dropout on the raw input features (0.5 is a citation-graph
    default and can be too aggressive for dense tabular features).
    dropout: dropout between the two layers.
    """

    def __init__(self, in_dim, hidden=128, out_dim=2, kind="gcn",
                 dropout=0.5, heads=4, in_dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.in_dropout = in_dropout
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hidden), GCNConv(hidden, out_dim)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, out_dim)
        elif kind == "gat":
            self.c1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            self.c2 = GATConv(hidden, out_dim, heads=1, dropout=dropout)
        else:
            raise ValueError(f"unknown kind: {kind}")

    def forward(self, x, edge_index):
        x = F.dropout(x, self.in_dropout, self.training)
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.c2(x, edge_index)
