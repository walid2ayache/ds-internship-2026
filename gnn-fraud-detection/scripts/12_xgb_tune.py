r"""Etape 8c: XGBoost tuning with rolling-origin validation.
Run from the repo root:  .venv\Scripts\python.exe scripts\12_xgb_tune.py

Pre-registered protocol:
  - Features: base 188 (local 93 + degree 2 + contrast 93), optionally plus the
    top-k aggregated features ranked by train-only gain (k = 8, 16, 32).
  - Validation folds inside train (no test, no val 30-34):
      fold A: train steps 1-19, validate 20-24
      fold B: train steps 1-24, validate 25-29
  - Fixed-tree grid (no early stopping in CV): depth 5/6/7, n_estimators 300/600,
    min_child_weight 1/5, lr 0.05.
  - Select by mean fold PR-AUC; tie-break lower std, fewer features, shallower.
  - Final: train on steps 1-29 with the selected config, 5 seeds averaged,
    threshold from val 30-34, test evaluated once.

The validation set 30-34 is used only for the final threshold, as everywhere else.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_LOCAL = 93
FEATSETS = ["base", "agg8", "agg16", "agg32"]
SEEDS = [42, 43, 44, 45, 46]


def make_features(data):
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
    base = np.hstack([xl, deg, contrast]).astype(np.float32)
    return base, X[:, N_LOCAL:].astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    X_base, X_agg = make_features(data)
    X_full = np.hstack([X_base, X_agg]).astype(np.float32)
    print(f"base {X_base.shape}, aggregated {X_agg.shape}", flush=True)

    grid = [dict(depth=d, n_estimators=n, mcw=m, lr=0.05)
            for d in (5, 6, 7) for n in (300, 600) for m in (1, 5)]

    def rank_agg(train_mask):
        model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, device=device, tree_method="hist", eval_metric="aucpr")
        model.fit(X_full[train_mask], y[train_mask], verbose=False)
        gain = model.feature_importances_[-X_agg.shape[1]:]
        return np.argsort(gain)[::-1]

    def features_for(kind, order):
        if kind == "base":
            return X_base
        k = int(kind[3:])
        return np.hstack([X_base, X_agg[:, order[:k]]]).astype(np.float32)

    def fit_cfg(cfg, X, train_mask, seed=42):
        model = xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"], learning_rate=cfg["lr"],
            max_depth=cfg["depth"], min_child_weight=cfg["mcw"],
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, device=device, tree_method="hist", eval_metric="aucpr")
        model.fit(X[train_mask], y[train_mask], verbose=False)
        return model

    folds = [(19, 24), (24, 29)]
    results = []
    for end, val_end in folds:
        tr_m = (t <= end) & (y != 2)
        va_m = (t > end) & (t <= val_end) & (y != 2)
        order = rank_agg(tr_m)
        mats = {kind: features_for(kind, order) for kind in FEATSETS}
        yv = y[va_m]
        for cfg in grid:
            for kind in FEATSETS:
                model = fit_cfg(cfg, mats[kind], tr_m)
                p = model.predict_proba(mats[kind][va_m])[:, 1]
                ap = float(average_precision_score(yv, p))
                results.append({"fold": f"{end + 1}-{val_end}",
                                "featset": kind, "depth": cfg["depth"],
                                "n_estimators": cfg["n_estimators"],
                                "mcw": cfg["mcw"], "ap": ap})
        print(f"fold {end + 1}-{val_end} done", flush=True)
        del mats

    # collapse to one row per (featset, depth, n_estimators, mcw)
    combos = {}
    for r in results:
        key = (r["featset"], r["depth"], r["n_estimators"], r["mcw"])
        combos.setdefault(key, {"fold_ap": [], "fold": []})
        combos[key]["fold_ap"].append(r["ap"])
        combos[key]["fold"].append(r["fold"])

    rows = []
    for (kind, depth, n_est, mcw), v in combos.items():
        rows.append({"featset": kind, "depth": depth, "n_estimators": n_est,
                     "mcw": mcw, "fold_ap": v["fold_ap"],
                     "mean_ap": statistics.mean(v["fold_ap"]),
                     "std_ap": statistics.pstdev(v["fold_ap"]),
                     "n_features": X_base.shape[1] + (0 if kind == "base" else int(kind[3:]))})
    rows.sort(key=lambda r: (r["mean_ap"], -r["std_ap"], -r["n_features"], -r["depth"]),
              reverse=True)

    print("\n=== rolling-origin CV (mean PR-AUC over folds 20-24 and 25-29) ===", flush=True)
    print(f"{'features':<8}{'depth':>6}{'trees':>7}{'mcw':>5}"
          f"{'fold A':>9}{'fold B':>9}{'mean':>9}", flush=True)
    for r in rows[:12]:
        print(f"{r['featset']:<8}{r['depth']:>6}{r['n_estimators']:>7}{r['mcw']:>5}"
              f"{r['fold_ap'][0]:>9.4f}{r['fold_ap'][1]:>9.4f}{r['mean_ap']:>9.4f}",
              flush=True)

    best = rows[0]
    print(f"\nCV-selected: {best['featset']} depth={best['depth']} "
          f"trees={best['n_estimators']} mcw={best['mcw']} "
          f"(AP {best['mean_ap']:.4f})", flush=True)

    order = rank_agg(tr_np)
    X_final = features_for(best["featset"], order)
    cfg = dict(depth=best["depth"], n_estimators=best["n_estimators"],
               mcw=best["mcw"], lr=0.05)

    pv, pt, per_seed = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model = fit_cfg(cfg, X_final, tr_np, seed=seed)
        a = model.predict_proba(X_final[va_np])[:, 1]
        b = model.predict_proba(X_final[te_np])[:, 1]
        ap = float(average_precision_score(y[va_np], a))
        thr = best_threshold(y[va_np], a)
        r = evaluate(y[te_np], b, thr)
        per_seed.append({"seed": seed, "val_ap": ap, "threshold": thr,
                         "test_pr_auc": r["pr_auc"], "test_f1": r["f1"],
                         "precision": r["precision"], "recall": r["recall"],
                         "seconds": time.time() - t0})
        pv.append(a)
        pt.append(b)
        print(f"seed {seed}: val AP {ap:.4f} test PR {r['pr_auc']:.4f} "
              f"F1 {r['f1']:.4f}", flush=True)

    p_val = np.mean(pv, axis=0)
    p_test = np.mean(pt, axis=0)
    thr = best_threshold(y[va_np], p_val)
    r = evaluate(y[te_np], p_test, thr)
    final = {"config": cfg, "featset": best["featset"],
             "n_features": best["n_features"],
             "cv_mean_ap": best["mean_ap"], "cv_fold_ap": best["fold_ap"],
             "val_ap": float(average_precision_score(y[va_np], p_val)),
             "threshold": thr, "test_pr_auc": r["pr_auc"],
             "test_f1": r["f1"], "precision": r["precision"],
             "recall": r["recall"], "per_seed": per_seed}
    print(f"\n5-seed average: val AP {final['val_ap']:.4f} | "
          f"test PR {r['pr_auc']:.4f} | F1 {r['f1']:.4f}", flush=True)

    base = json.load(open(RESULTS / "boost_metrics.json"))
    old = next(x for x in base["xgb"] if x["name"] == "xgb_local_deg_contrast")
    print(f"previous best test PR {old['test_pr_auc']:.4f} -> "
          f"{r['pr_auc']:.4f} ({r['pr_auc'] - old['test_pr_auc']:+.4f})", flush=True)

    payload = {"grid": rows, "selected": best, "final": final,
               "previous_best": old,
               "note": "CV inside train only; test evaluated once for the CV-selected config."}
    with open(RESULTS / "xgb_tune.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {RESULTS / 'xgb_tune.json'}", flush=True)


if __name__ == "__main__":
    main()
