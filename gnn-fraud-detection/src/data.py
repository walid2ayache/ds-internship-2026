import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

LICIT, ILLICIT, UNKNOWN = 0, 1, 2

# Temporal split on the 49 Elliptic time steps.
TRAIN_END = 29   # train: steps 1-29
VAL_END = 34     # val:   steps 30-34
                 # test:  steps 35-49 (canonical Elliptic boundary)


def load_elliptic(root="data/elliptic"):
    """Load the Elliptic Bitcoin graph as a single PyG Data object."""
    dataset = EllipticBitcoinDataset(root=root)
    return dataset[0]


def load_time_steps(root="data/elliptic"):
    """Read the raw time step per node (PyG drops this column).

    Row order matches PyG's node order (verified: masks line up with it).
    """
    path = f"{root}/raw/elliptic_txs_features.csv"
    return torch.tensor(pd.read_csv(path, header=None, usecols=[1])[1].values)


def get_split(data, root="data/elliptic"):
    """Return (train_idx, val_idx, test_idx), split by time, unknown removed.

    Unknown-label nodes never enter the loss or metrics, but they stay in the
    graph so they can still pass messages to their labeled neighbors.
    """
    t = load_time_steps(root)
    assert len(t) == data.num_nodes, "time steps do not align with nodes"
    labeled = data.y != UNKNOWN
    train_idx = ((t <= TRAIN_END) & labeled).nonzero(as_tuple=True)[0]
    val_idx = ((t > TRAIN_END) & (t <= VAL_END) & labeled).nonzero(as_tuple=True)[0]
    test_idx = ((t > VAL_END) & labeled).nonzero(as_tuple=True)[0]
    return train_idx, val_idx, test_idx


def class_weights(y_train):
    """Inverse-frequency weights for [licit, illicit], for weighted loss."""
    counts = torch.bincount(y_train, minlength=2).float()
    return counts.sum() / (2.0 * counts)
