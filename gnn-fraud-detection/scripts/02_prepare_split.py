r"""Etape 3: build the labeled, time-based train/val/test split and verify it.
Run from the repo root:  .venv\Scripts\python.exe scripts\02_prepare_split.py
Paste the printed output back to Claude.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import (ILLICIT, LICIT, UNKNOWN, class_weights, get_split,
                      load_elliptic, load_time_steps)

data = load_elliptic()
t = load_time_steps()
train_idx, val_idx, test_idx = get_split(data)
splits = {"train": train_idx, "val": val_idx, "test": test_idx}

print("=== Split sizes (unknown removed) ===")
for name, idx in splits.items():
    print(f"{name}: {len(idx)} nodes")

print("\n=== Time ranges (must not overlap, must be ordered) ===")
for name, idx in splits.items():
    print(f"{name}: steps {int(t[idx].min())} to {int(t[idx].max())}")

print("\n=== Disjointness checks (all must be 0) ===")
names = list(splits)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = set(splits[names[i]].tolist()), set(splits[names[j]].tolist())
        print(f"{names[i]}/{names[j]} overlap: {len(a & b)}")
for name, idx in splits.items():
    print(f"unknown in {name}: {int((data.y[idx] == UNKNOWN).sum())}")

print("\n=== Consistency with PyG's own masks ===")
pyg_train = set(((data.train_mask) & (data.y != UNKNOWN)).nonzero(as_tuple=True)[0].tolist())
ours = set(train_idx.tolist()) | set(val_idx.tolist())
print(f"our train+val == PyG train: {ours == pyg_train}")
pyg_test = set(((data.test_mask) & (data.y != UNKNOWN)).nonzero(as_tuple=True)[0].tolist())
print(f"our test == PyG test: {set(test_idx.tolist()) == pyg_test}")

print("\n=== Class balance per split ===")
for name, idx in splits.items():
    y = data.y[idx]
    n_licit, n_illicit = int((y == LICIT).sum()), int((y == ILLICIT).sum())
    print(f"{name}: licit={n_licit}  illicit={n_illicit}  "
          f"illicit rate={100 * n_illicit / max(len(y), 1):.2f}%")

print("\n=== Class weights from train only (licit, illicit) ===")
print(class_weights(data.y[train_idx]))
