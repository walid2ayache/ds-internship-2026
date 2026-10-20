r"""Build the portfolio demo assets for the Elliptic fraud app.
Run from the repo root:  .venv\Scripts\python.exe scripts\17_build_app.py

Creates:
  app/model/xgb_0.ubj ... xgb_4.ubj   (5-seed final fit, steps 1-34, 332 trees)
  app/data/demo.npz                   (400 labelled test transactions, features/probs)
  app/data/meta.json                  (metrics, feature names, demo metadata)
  app/static/img/per_step_f1.png      (copied from results/)
  app/static/img/pr_curves.png

The demo rows are sampled from test steps 35-42 only, 25 illicit + 25 licit per step.
True labels are kept for the demo UI; the model never sees them at inference.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_elliptic, load_time_steps
from src.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APP = ROOT / "app"
N_LOCAL = 93
SEEDS = [42, 43, 44, 45, 46]
TREES = 332


def features(data):
    """Same 188-feature definition as scripts/10,11,12,13,14,16."""
    X = data.x.numpy()
    src, dst = data.edge_index.numpy()
    n = data.num_nodes
    xl = X[:, :N_LOCAL]
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)
    deg = np.column_stack([np.log1p(out_deg), np.log1p(in_deg)]).astype(np.float32)
    neigh_sum = np.zeros_like(xl)
    np.add.at(neigh_sum, dst, xl[src])
    np.add.at(neigh_sum, src, xl[dst])
    counts = np.bincount(dst, minlength=n) + np.bincount(src, minlength=n)
    neigh_mean = neigh_sum / np.maximum(counts, 1)[:, None]
    contrast = (xl - neigh_mean).astype(np.float32)
    return np.hstack([xl, deg, contrast]).astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    X = features(data)
    print(f"device {device}; X {X.shape}", flush=True)

    (APP / "model").mkdir(parents=True, exist_ok=True)
    (APP / "data").mkdir(parents=True, exist_ok=True)
    (APP / "static" / "img").mkdir(parents=True, exist_ok=True)

    tr_m = (t <= 34) & (y != 2)
    models = []
    for i, seed in enumerate(SEEDS):
        m = xgb.XGBClassifier(
            n_estimators=TREES, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, device=device, tree_method="hist", eval_metric="aucpr")
        m.fit(X[tr_m], y[tr_m], verbose=False)
        path = APP / "model" / f"xgb_{i}.ubj"
        m.get_booster().save_model(path)
        models.append(m.get_booster())
        print(f"saved {path.name} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)

    # Sanity check against the reported final-fit numbers.
    te_m = (t >= 35) & (y != 2)
    dmat = xgb.DMatrix(X[te_m])
    p_test = np.mean([b.predict(dmat) for b in models], axis=0)
    final = json.load(open(RESULTS / "final_fit.json"))
    thr = float(final["threshold"])
    live = evaluate(y[te_m], p_test, thr)
    print(f"live test: PR {live['pr_auc']:.4f} (report {final['final_5seed']['pr_auc']:.4f}) | "
          f"F1 {live['f1']:.4f} (report {final['final_5seed']['f1']:.4f})", flush=True)

    # Balanced demo sample from steps 35-42.
    rng = np.random.default_rng(0)
    rows = []
    for s in range(35, 43):
        for lab in (1, 0):
            idx = np.where((t == s) & (y == lab))[0]
            take = min(25, len(idx))
            rows.extend(rng.choice(idx, size=take, replace=False).tolist())
    rows = np.array(sorted(rows), dtype=np.int64)
    X_demo = X[rows]
    y_demo = y[rows].astype(np.int8)
    step_demo = t[rows].astype(np.int16)
    node_demo = rows.astype(np.int32)
    prob_demo = np.mean([b.predict(xgb.DMatrix(X_demo)) for b in models], axis=0)

    np.savez_compressed(
        APP / "data" / "demo.npz",
        X=X_demo, y=y_demo, step=step_demo, node=node_demo, prob=prob_demo)
    print(f"demo rows: {len(rows)} ({int(y_demo.sum())} illicit)", flush=True)

    # SAGE row from results/RESULTS.md (seed noise is ~0.01).
    sage = {"test_pr_auc": 0.7116, "test_f1": 0.6660,
            "precision": 0.947, "recall": 0.514}
    names = ([f"local feature {i}" for i in range(N_LOCAL)]
             + ["log out-degree", "log in-degree"]
             + [f"contrast feature {i}" for i in range(N_LOCAL)])
    meta = {
        "title": "Elliptic Fraud Detection",
        "model": "XGBoost · 5-seed final fit",
        "training": "steps 1–34, 332 trees per seed, no early stopping",
        "threshold": thr,
        "metrics": {
            "final_fit": {k: final["final_5seed"][k]
                          for k in ("pr_auc", "f1", "precision", "recall")},
            "train_only": {k: final["reference"][k]
                           for k in ("pr_auc", "f1", "precision", "recall")},
            "sage_local": {k: sage[k]
                           for k in ("test_pr_auc", "test_f1", "precision", "recall")},
        },
        "live_check": {"pr_auc": live["pr_auc"], "f1": live["f1"]},
        "dataset": {"nodes": int(data.num_nodes), "edges": int(data.num_edges),
                    "features": 165, "steps": 49,
                    "train": "1–29", "val": "30–34", "test": "35–49"},
        "demo": {"count": int(len(rows)), "steps": [35, 42],
                 "sampling": "25 illicit + 25 licit per step"},
        "feature_count": len(names),
        "features": names,
        "figures": ["/static/img/per_step_f1.png", "/static/img/pr_curves.png"],
    }
    with open(APP / "data" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for name in ("per_step_f1.png", "pr_curves.png"):
        src = RESULTS / name
        if src.exists():
            shutil.copyfile(src, APP / "static" / "img" / name)
    print(f"wrote {APP / 'data' / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
