"""Etape 2: explore the Elliptic dataset.
Run from the repo root:  python scripts/01_explore.py
Paste the printed output back to Claude to confirm label encoding and masks.
"""
import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_elliptic

data = load_elliptic()

print("=== Graph ===")
print(data)
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features per node: {data.num_node_features}")
print(f"Directed: {data.is_directed()}")

print("\n=== Labels (y) ===")
vals, counts = torch.unique(data.y, return_counts=True)
for v, c in zip(vals.tolist(), counts.tolist()):
    print(f"label {v}: {c} ({100 * c / data.num_nodes:.2f}%)")

print("\n=== Masks ===")
for name in ["train_mask", "val_mask", "test_mask"]:
    if hasattr(data, name):
        m = getattr(data, name)
        lab = data.y[m]
        v, c = torch.unique(lab, return_counts=True)
        print(f"{name}: {int(m.sum())} nodes, label counts {dict(zip(v.tolist(), c.tolist()))}")
    else:
        print(f"{name}: not present")

print("\n=== Attributes available ===")
print(list(data.keys()))

os.makedirs("results", exist_ok=True)
plt.bar([str(v) for v in vals.tolist()], counts.tolist())
plt.yscale("log")
plt.title("Label distribution (log scale)")
plt.xlabel("label")
plt.ylabel("nodes")
plt.savefig("results/label_distribution.png", dpi=150, bbox_inches="tight")
print("\nSaved results/label_distribution.png")
