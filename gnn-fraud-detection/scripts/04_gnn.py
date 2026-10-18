r"""Etape 5: GCN / GraphSAGE / GAT on the same split and metrics as the baseline.
Run from the repo root:  .venv\Scripts\python.exe scripts\04_gnn.py
Paste the printed table back to Claude.

6 configs (3 models x plain/weighted loss) x 3 seeds, so results are mean +- std.
Imbalance methods beyond class weights (focal loss, sampling) come in etape 6.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import class_weights, get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate, per_step_f1
from src.models import GNN
from src.train import train_gnn

SEEDS = [0, 1, 2]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

data = load_elliptic()
t = load_time_steps().numpy()
y = data.y.numpy()
tr, va, te = get_split(data)
tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
w = class_weights(data.y[tr])

# Sanity check: do edges connect different time steps? (matters for leakage)
src, dst = data.edge_index.numpy()
cross = int((t[src] != t[dst]).sum())
print(f"edges connecting different time steps: {cross} of {len(src)}")

results = {}
for kind in ["gcn", "sage", "gat"]:
    for weighted in [False, True]:
        name = f"{kind}_{'weighted' if weighted else 'plain'}"
        runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GNN(data.num_node_features, kind=kind)
            probs, val_pr = train_gnn(model, data, tr, va,
                                      weight=w if weighted else None, device=device)
            thr = best_threshold(y[va_np], probs[va_np])  # threshold from val only
            r = evaluate(y[te_np], probs[te_np], thr)
            r["val_pr_auc"] = val_pr
            r["per_step_f1"] = per_step_f1(y[te_np], probs[te_np], t[te_np], thr)
            runs.append(r)
        results[name] = runs
        print(f"done {name}")

keys = ["precision", "recall", "f1", "pr_auc", "val_pr_auc"]
print("\n=== GNNs: TEST metrics, mean +- std over seeds (illicit class) ===")
print(f"{'model':<16}{'prec':>12}{'recall':>12}{'F1':>12}{'PR-AUC':>12}{'val PR-AUC':>13}")
summary = {}
for name, runs in results.items():
    m = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}
    summary[name] = m
    print(f"{name:<16}" + "".join(f"{m[k][0]:>7.3f}+-{m[k][1]:.3f}" for k in keys))

best = max(summary, key=lambda k: summary[k]["val_pr_auc"][0])  # chosen on VAL
print(f"\n=== Per-time-step test F1, mean over seeds (variant with best VAL PR-AUC: {best}) ===")
steps = sorted(results[best][0]["per_step_f1"])
avg = {}
for s in steps:
    vals = [r["per_step_f1"][s] for r in results[best] if r["per_step_f1"][s] is not None]
    avg[s] = round(float(np.mean(vals)), 2) if vals else None
print(avg)

os.makedirs("results", exist_ok=True)
with open("results/gnn_metrics.json", "w") as f:
    json.dump({"runs": results, "summary": summary}, f, indent=2)
print("\nSaved results/gnn_metrics.json")
