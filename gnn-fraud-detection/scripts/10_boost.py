r"""Etape 8b: Phase 2 - engineered features and XGB + SAGE ensemble.
Run from the repo root:  .venv\Scripts\python.exe scripts\10_boost.py

Candidates (all selected on VAL only):
  - XGBoost: all 165 features, + degree, + neighbour-contrast features,
    weighted / unweighted.
  - SAGE local 93 vs local + degree + contrast (188 features), 3 seeds.
  - Rank-average ensemble of the best XGB and the best SAGE, weight chosen
    on VAL PR-AUC.

The contrast features are x_local - mean(neighbour x_local); they use features
only (no labels), so they are safe for val/test. Degree uses the edge list only.
Test metrics are computed for reporting; the winner is fixed by VAL PR-AUC.
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
from src.models import GNN
from src.train import train_gnn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_LOCAL = 93
SEEDS = [0, 1, 2]
XGB_SEED = 42


def build_features(data, t):
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

    local_eng = np.hstack([xl, deg, contrast]).astype(np.float32)
    step_norm = local_eng.copy()
    for s in np.unique(t):
        m = t == s
        mu = step_norm[m].mean(0)
        sd = step_norm[m].std(0) + 1e-6
        step_norm[m] = (step_norm[m] - mu) / sd

    return {
        "all": X,
        "local": xl,
        "all_deg_contrast": np.hstack([X, deg, contrast]).astype(np.float32),
        "local_deg_contrast": local_eng,
        "local_deg_contrast_sn": step_norm.astype(np.float32),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    yv = y[va_np]
    F = build_features(data, t)
    print({k: v.shape for k, v in F.items()}, flush=True)

    def xgb_run(name, feats, weighted):
        n_neg = int((y[tr_np] == 0).sum())
        n_pos = int((y[tr_np] == 1).sum())
        X = F[feats]
        model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float(n_neg / n_pos) if weighted else 1.0,
            eval_metric="aucpr", early_stopping_rounds=50,
            random_state=XGB_SEED, n_jobs=-1, device=device, tree_method="hist",
        )
        t0 = time.time()
        model.fit(X[tr_np], y[tr_np],
                  eval_set=[(X[va_np], y[va_np])], verbose=False)
        p_val = model.predict_proba(X[va_np])[:, 1]
        p_test = model.predict_proba(X[te_np])[:, 1]
        thr = best_threshold(yv, p_val)
        r = evaluate(y[te_np], p_test, thr)
        val_ap = float(average_precision_score(yv, p_val))
        print(f"{name:<26} val AP {val_ap:.4f} | test PR {r['pr_auc']:.4f} | "
              f"F1 {r['f1']:.4f} | {time.time() - t0:.0f}s", flush=True)
        return {"name": name, "feats": feats, "weighted": weighted,
                "val_ap": val_ap, "val_f1_at_thr": evaluate(yv, p_val, thr)["f1"],
                "threshold": thr, "test_pr_auc": r["pr_auc"], "test_f1": r["f1"],
                "precision": r["precision"], "recall": r["recall"],
                "best_iteration": int(model.best_iteration)}, p_val, p_test

    xgb_runs = []
    xgb_probs = {}
    for name, feats, weighted in [
        ("xgb_all", "all", False),
        ("xgb_all_weighted", "all", True),
        ("xgb_all_deg_contrast", "all_deg_contrast", False),
        ("xgb_all_deg_contrast_w", "all_deg_contrast", True),
        ("xgb_local_deg_contrast", "local_deg_contrast", False),
        ("xgb_local_deg_contrast_w", "local_deg_contrast", True),
        ("xgb_local_deg_contrast_sn", "local_deg_contrast_sn", False),
    ]:
        r, pv, pt = xgb_run(name, feats, weighted)
        xgb_runs.append(r)
        xgb_probs[name] = (pv, pt)

    def sage_run(name, feats):
        d = data.clone()
        d.x = torch.tensor(F[feats], dtype=torch.float32)
        pv, pt, per_seed = [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GNN(d.num_node_features, kind="sage", in_dropout=0.5)
            probs, _ = train_gnn(model, d, tr, va, device=device)
            a, b = probs[va_np], probs[te_np]
            thr = best_threshold(yv, a)
            r = evaluate(y[te_np], b, thr)
            ap = float(average_precision_score(yv, a))
            per_seed.append({"seed": seed, "val_ap": ap, "threshold": thr,
                             "test_pr_auc": r["pr_auc"], "test_f1": r["f1"],
                             "precision": r["precision"],
                             "recall": r["recall"]})
            pv.append(a)
            pt.append(b)
        mean = {k: statistics.mean(x[k] for x in per_seed)
                for k in ("val_ap", "test_pr_auc", "test_f1", "precision", "recall")}
        std = {k: statistics.pstdev(x[k] for x in per_seed)
               for k in ("val_ap", "test_pr_auc", "test_f1")}
        print(f"{name:<26} val AP {mean['val_ap']:.4f}+-{std['val_ap']:.4f} | "
              f"test PR {mean['test_pr_auc']:.4f} | F1 {mean['test_f1']:.4f}",
              flush=True)
        return {"name": name, "feats": feats, "per_seed": per_seed,
                "mean": mean, "std": std}, np.mean(pv, axis=0), np.mean(pt, axis=0)

    sage_runs = []
    sage_probs = {}
    for name, feats in [("sage_local", "local"),
                        ("sage_local_deg_contrast", "local_deg_contrast")]:
        r, pv, pt = sage_run(name, feats)
        sage_runs.append(r)
        sage_probs[name] = (pv, pt)

    def rank01(p):
        order = np.argsort(p, kind="mergesort")
        r = np.empty(len(p), dtype=float)
        r[order] = np.arange(len(p))
        return r / max(len(p) - 1, 1)

    best_xgb = max(xgb_runs, key=lambda r: r["val_ap"])
    best_sage = max(sage_runs, key=lambda r: r["mean"]["val_ap"])
    print(f"\nbest XGB by VAL: {best_xgb['name']} "
          f"({best_xgb['val_ap']:.4f})", flush=True)
    print(f"best SAGE by VAL: {best_sage['name']} "
          f"({best_sage['mean']['val_ap']:.4f})", flush=True)

    xv, xt = xgb_probs[best_xgb["name"]]
    # Use the established SAGE-local for the ensemble (it was the pre-registered
    # GNN partner); sage_local_deg_contrast is reported as its own candidate.
    sv, st = sage_probs["sage_local"]
    rv, rt = rank01(xv), rank01(xt)
    rsv, rst = rank01(sv), rank01(st)
    weights = np.round(np.linspace(0.0, 1.0, 21), 2)
    ens = []
    for w in weights:
        p_val = w * rv + (1 - w) * rsv
        ap = float(average_precision_score(yv, p_val))
        ens.append((ap, float(w)))
    ens.sort(reverse=True)
    best_ap, best_w = ens[0]
    p_val = best_w * rv + (1 - best_w) * rsv
    p_test = best_w * rt + (1 - best_w) * rst
    thr = best_threshold(yv, p_val)
    r = evaluate(y[te_np], p_test, thr)
    ensemble = {"name": "ensemble_rank", "xgb": best_xgb["name"],
                "sage": "sage_local", "w_xgb": best_w,
                "val_ap": best_ap, "threshold": thr,
                "test_pr_auc": r["pr_auc"], "test_f1": r["f1"],
                "precision": r["precision"], "recall": r["recall"]}
    print(f"ensemble_rank              val AP {best_ap:.4f} (w_xgb={best_w:.2f}) | "
          f"test PR {r['pr_auc']:.4f} | F1 {r['f1']:.4f}", flush=True)

    prob_ens = []
    for w in weights:
        ap = float(average_precision_score(yv, w * xv + (1 - w) * sv))
        prob_ens.append((ap, float(w)))
    prob_ens.sort(reverse=True)
    p_ap, p_w = prob_ens[0]
    p_val = p_w * xv + (1 - p_w) * sv
    p_test = p_w * xt + (1 - p_w) * st
    thr_p = best_threshold(yv, p_val)
    rp = evaluate(y[te_np], p_test, thr_p)
    ensemble_prob = {"name": "ensemble_prob", "xgb": best_xgb["name"],
                     "sage": "sage_local", "w_xgb": p_w,
                     "val_ap": p_ap, "threshold": thr_p,
                     "test_pr_auc": rp["pr_auc"], "test_f1": rp["f1"],
                     "precision": rp["precision"], "recall": rp["recall"]}
    print(f"ensemble_prob              val AP {p_ap:.4f} (w_xgb={p_w:.2f}) | "
          f"test PR {rp['pr_auc']:.4f} | F1 {rp['f1']:.4f}", flush=True)

    print("\n=== Phase 2 candidates (sorted by VAL PR-AUC) ===", flush=True)
    rows = []
    for r in xgb_runs:
        rows.append((r["val_ap"], r["name"], r["test_pr_auc"], r["test_f1"],
                     r["precision"], r["recall"]))
    for r in sage_runs:
        rows.append((r["mean"]["val_ap"], r["name"], r["mean"]["test_pr_auc"],
                     r["mean"]["test_f1"], r["mean"]["precision"],
                     r["mean"]["recall"]))
    rows.append((ensemble["val_ap"], ensemble["name"], ensemble["test_pr_auc"],
                 ensemble["test_f1"], ensemble["precision"],
                 ensemble["recall"]))
    rows.append((ensemble_prob["val_ap"], ensemble_prob["name"],
                 ensemble_prob["test_pr_auc"], ensemble_prob["test_f1"],
                 ensemble_prob["precision"], ensemble_prob["recall"]))
    rows.sort(reverse=True)
    print(f"{'model':<28}{'val AP':>9}{'test PR':>9}{'test F1':>9}"
          f"{'P':>8}{'R':>8}", flush=True)
    for ap, name, pr, f1, p, rc in rows:
        print(f"{name:<28}{ap:>9.4f}{pr:>9.4f}{f1:>9.4f}{p:>8.3f}{rc:>8.3f}",
              flush=True)

    winner = rows[0][1]
    print(f"\nVAL-selected winner: {winner}", flush=True)

    def bootstrap_ap_delta(y_true, p_new, p_old, n=1000):
        rng = np.random.default_rng(0)
        n_nodes = len(y_true)
        deltas = []
        for _ in range(n):
            idx = rng.integers(0, n_nodes, n_nodes)
            yy = y_true[idx]
            if yy.sum() < 2 or yy.sum() == len(yy):
                continue
            deltas.append(average_precision_score(yy, p_new[idx]) -
                          average_precision_score(yy, p_old[idx]))
        d = np.array(deltas)
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    p_new = xgb_probs["xgb_local_deg_contrast"][1]
    p_old = xgb_probs["xgb_all"][1]
    d, lo, hi = bootstrap_ap_delta(y[te_np], p_new, p_old)
    print(f"Test PR-AUC delta (local_deg_contrast - xgb_all): {d:+.4f} "
          f"[95% {lo:+.4f}, {hi:+.4f}]", flush=True)

    payload = {"xgb": xgb_runs, "sage": sage_runs,
               "ensemble_rank": ensemble, "ensemble_prob": ensemble_prob,
               "winner": winner,
               "note": "All candidates selected by VAL PR-AUC only; test reported."}
    with open(RESULTS / "boost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {RESULTS / 'boost_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
