r"""Final fit on train+val (steps 1-34), fixed trees, threshold from the train-only model.
Run from the repo root:  .venv\Scripts\python.exe scripts\14_final_fit.py

Protocol:
  - Train the current best model on steps 1-29 with early stopping on val 30-34.
    This gives the reference model, its best iteration, and the VAL threshold 0.39.
  - Refit on steps 1-34 with n_estimators = best_iteration + 1 (no early stopping,
    no validation use for tree count). Apply the train-only threshold unchanged.
  - Report single-seed (42) and 5-seed averaged variants, with paired bootstrap
    CIs versus the train-only model.

This uses val labels for training, so the final-fit numbers are not directly
comparable to the VAL-selected table; the test set is still untouched by selection.
"""
import json
import os
import sys

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
N_LOCAL = 93
SEEDS = [42, 43, 44, 45, 46]


def features(data):
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
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    X = features(data)
    trva = np.concatenate([tr_np, va_np])

    def metrics(p_test, thr):
        r = evaluate(y[te_np], p_test, thr)
        return {"threshold": float(thr), "pr_auc": r["pr_auc"],
                "f1": r["f1"], "precision": r["precision"],
                "recall": r["recall"]}

    # Reference: train on 1-29, early stop on val.
    ref = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        early_stopping_rounds=50, random_state=42, n_jobs=-1,
        device=device, tree_method="hist")
    ref.fit(X[tr_np], y[tr_np],
            eval_set=[(X[va_np], y[va_np])], verbose=False)
    p_ref_val = ref.predict_proba(X[va_np])[:, 1]
    p_ref_test = ref.predict_proba(X[te_np])[:, 1]
    thr = best_threshold(y[va_np], p_ref_val)
    ref_metrics = metrics(p_ref_test, thr)
    ref_metrics["val_ap"] = float(average_precision_score(y[va_np], p_ref_val))
    n_final = int(ref.best_iteration) + 1
    print(f"reference: best_iteration={ref.best_iteration} -> final trees={n_final}, "
          f"thr={thr:.2f}", flush=True)
    print(f"reference test PR {ref_metrics['pr_auc']:.4f} | "
          f"F1 {ref_metrics['f1']:.4f}", flush=True)

    def fit_final(seed):
        m = xgb.XGBClassifier(
            n_estimators=n_final, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            n_jobs=-1, device=device, tree_method="hist", eval_metric="aucpr")
        m.fit(X[trva], y[trva], verbose=False)
        return m.predict_proba(X[te_np])[:, 1]

    p_single = fit_final(42)
    single = metrics(p_single, thr)

    pts, per_seed = [], []
    for seed in SEEDS:
        p = fit_final(seed)
        r = evaluate(y[te_np], p, thr)
        per_seed.append({"seed": seed, "pr_auc": r["pr_auc"], "f1": r["f1"],
                         "precision": r["precision"], "recall": r["recall"]})
        pts.append(p)
        print(f"seed {seed}: test PR {r['pr_auc']:.4f} | F1 {r['f1']:.4f}", flush=True)
    p_avg = np.mean(pts, axis=0)
    avg = metrics(p_avg, thr)

    def f1_at(y_true, p, threshold):
        pred = p >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    def boot(y_true, p_new, p_old, threshold, n=1000):
        rng = np.random.default_rng(0)
        dpr, df1 = [], []
        for _ in range(n):
            idx = rng.integers(0, len(y_true), len(y_true))
            yy = y_true[idx]
            if yy.sum() < 2 or yy.sum() == len(yy):
                continue
            dpr.append(average_precision_score(yy, p_new[idx]) -
                       average_precision_score(yy, p_old[idx]))
            df1.append(f1_at(yy, p_new[idx], threshold) -
                       f1_at(yy, p_old[idx], threshold))
        dpr, df1 = np.array(dpr), np.array(df1)
        return (float(dpr.mean()), float(np.percentile(dpr, 2.5)),
                float(np.percentile(dpr, 97.5)), float(df1.mean()),
                float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5)))

    for name, p in [("final_single", p_single), ("final_5seed", p_avg)]:
        pr_m, pr_lo, pr_hi, f1_m, f1_lo, f1_hi = boot(
            y[te_np], p, p_ref_test, thr)
        print(f"\n{name} vs reference: delta PR {pr_m:+.4f} "
              f"[95% {pr_lo:+.4f}, {pr_hi:+.4f}] | delta F1 {f1_m:+.4f} "
              f"[95% {f1_lo:+.4f}, {f1_hi:+.4f}]", flush=True)

    payload = {"reference": ref_metrics, "best_iteration": int(ref.best_iteration),
               "final_trees": n_final, "threshold": thr,
               "final_single": single, "final_5seed": avg,
               "per_seed": per_seed,
               "note": "Final fit uses steps 1-34; threshold from the train-only model."}
    with open(os.path.join(RESULTS, "final_fit.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {os.path.join(RESULTS, 'final_fit.json')}", flush=True)


if __name__ == "__main__":
    main()
