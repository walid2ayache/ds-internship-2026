r"""Rolling-origin evaluation (corrected 4-fold version).
Run from the repo root:  .venv\Scripts\python.exe scripts\16_rolling.py

Cut points k = 20, 25, 30, 34. For each fold:
  train steps 1..k-3, validate k-2..k, test k+1..min(k+5, 49).
Nothing after k is used for training, early stopping or thresholds.

Models: XGBoost on local 93 + degree 2 + neighbour-contrast 93 (same config as
the train-only best), and SAGE on the 93 local features (in_dropout 0.5), both
with 3 seeds. Threshold rules:
  A: F1-optimal on the fold's validation block.
  B: F1-optimal on pooled predictions of earlier folds whose test window ended
     on or before this fold's k; first fold falls back to A.

k=38 is intentionally excluded: it would train on step 35 and validate on 36-38,
which are old-test steps, violating the no-old-test-for-choices rule. k=34
evaluates 35-39 for reporting only; nothing from 35-49 is used to choose.

Creates only results/rolling.json (or --out). Does not touch src/ or other scripts.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate
from src.models import GNN
from src.train import train_gnn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_LOCAL = 93


def features(data):
    """Same definition as scripts/10,11,12,13,14: local + log-degree + contrast."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", default="20,25,30,34")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", default="rolling.json")
    return p.parse_args()


def main():
    args = parse_args()
    folds = sorted(set(int(x) for x in args.folds.split(",")))
    seeds = sorted(set(int(x) for x in args.seeds.split(",")))
    if not folds or not seeds:
        raise SystemExit("at least one fold and one seed are required")
    for k in folds:
        if k < 4:
            raise SystemExit(f"k={k} leaves no training steps")
        if k > 34:
            raise SystemExit(
                f"k={k} would use steps 35+ for training/validation; "
                "old-test steps must not be used for choices (k <= 34)")
        if k + 1 > 49:
            raise SystemExit(f"k={k} leaves no test steps")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}; folds={folds}; seeds={seeds}", flush=True)

    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    X = features(data)
    d_local = data.clone()
    d_local.x = data.x[:, :N_LOCAL]
    print(f"X {X.shape}; local {d_local.num_node_features} features", flush=True)

    def idx(mask):
        return torch.tensor(np.where(mask)[0], dtype=torch.long)

    def xgb_run(tr_m, va_m, te_m, seed):
        model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            early_stopping_rounds=50, random_state=seed, n_jobs=-1,
            device=device, tree_method="hist")
        model.fit(X[tr_m], y[tr_m],
                  eval_set=[(X[va_m], y[va_m])], verbose=False)
        return (model.predict_proba(X[va_m])[:, 1],
                model.predict_proba(X[te_m])[:, 1],
                int(model.best_iteration))

    def sage_run(tr_i, va_i, te_i, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GNN(N_LOCAL, kind="sage", in_dropout=0.5)
        probs, _ = train_gnn(model, d_local, tr_i, va_i, device=device)
        return probs[va_i.numpy()], probs[te_i.numpy()], None

    def metrics(y_true, p, thr):
        r = evaluate(y_true, p, thr)
        return {"pr_auc": r["pr_auc"], "f1": r["f1"],
                "precision": r["precision"], "recall": r["recall"]}

    history = {"xgb": [], "sage": []}
    folds_out = {}

    for k in folds:
        t0 = time.time()
        tr_m = (t <= k - 3) & (y != 2)
        va_m = (t > k - 3) & (t <= k) & (y != 2)
        te_end = min(k + 5, 49)
        te_m = (t > k) & (t <= te_end) & (y != 2)
        tr_i, va_i, te_i = idx(tr_m), idx(va_m), idx(te_m)
        yv, yt = y[va_m], y[te_m]
        print(f"\n== fold k={k}: train 1-{k - 3}, val {k - 2}-{k}, "
              f"test {k + 1}-{te_end}; test {int(yt.sum())}/{len(yt)} illicit ==",
              flush=True)

        fold = {"k": k, "train_steps": [1, k - 3], "val_steps": [k - 2, k],
                "test_steps": [k + 1, te_end],
                "test_positives": int(yt.sum()), "test_total": int(len(yt)),
                "models": {}}

        for name in ("xgb", "sage"):
            per_seed = []
            for seed in seeds:
                if name == "xgb":
                    p_val, p_test, best_it = xgb_run(tr_m, va_m, te_m, seed)
                else:
                    p_val, p_test, best_it = sage_run(tr_i, va_i, te_i, seed)
                thr_a = best_threshold(yv, p_val)
                r_a = metrics(yt, p_test, thr_a)
                per_seed.append({"seed": seed, "threshold_a": thr_a,
                                 "best_iteration": best_it,
                                 "rule_a": r_a,
                                 "p_val": p_val, "p_test": p_test})
                print(f"  {name} seed {seed}: thrA {thr_a:.2f} | "
                      f"F1 {r_a['f1']:.4f} | PR {r_a['pr_auc']:.4f}", flush=True)

            # Rule B: pool earlier folds whose test window ended <= k.
            pool_p, pool_y = [], []
            for rec in history[name]:
                if rec["end"] <= k:
                    for p in rec["p"]:
                        pool_p.append(p)
                        pool_y.append(rec["y"])
            if pool_p:
                thr_b = best_threshold(np.concatenate(pool_y),
                                       np.concatenate(pool_p))
                source = "pooled"
            else:
                thr_b = None
                source = "fallback_a"

            for rec in per_seed:
                if thr_b is None:
                    rec["rule_b"] = dict(rec["rule_a"])
                    rec["threshold_b"] = rec["threshold_a"]
                else:
                    rec["rule_b"] = metrics(yt, rec["p_test"], thr_b)
                    rec["threshold_b"] = thr_b
                del rec["p_val"]

            history[name].append({"end": te_end, "y": yt.tolist(),
                                  "p": [r["p_test"] for r in per_seed]})
            for rec in per_seed:
                rec.pop("p_test", None)

            fold["models"][name] = {
                "threshold_source": source,
                "threshold_b": (float(thr_b) if thr_b is not None else None),
                "per_seed": per_seed,
            }

        folds_out[str(k)] = fold
        print(f"  fold done in {time.time() - t0:.0f}s", flush=True)

    # Aggregate: mean over seeds per fold, then mean/std across folds.
    def fold_mean(seed_rows, rule, metric):
        return statistics.mean(r[rule][metric] for r in seed_rows)

    summary = {}
    for name in ("xgb", "sage"):
        summary[name] = {}
        for rule in ("rule_a", "rule_b"):
            summary[name][rule] = {}
            for metric in ("f1", "pr_auc", "precision", "recall"):
                per_fold = {k: fold_mean(f["models"][name]["per_seed"], rule, metric)
                            for k, f in folds_out.items()}
                vals = list(per_fold.values())
                summary[name][rule][metric] = {
                    "per_fold": per_fold,
                    "mean": statistics.mean(vals),
                    "std": statistics.pstdev(vals),
                }
            seed_std = {}
            for k, f in folds_out.items():
                rows = f["models"][name]["per_seed"]
                seed_std[k] = statistics.pstdev([r[rule]["f1"] for r in rows])
            summary[name][rule]["seed_std_f1"] = seed_std

    def table(metric, title):
        print(f"\n=== {title} (mean over seeds; mean +- std across folds) ===",
              flush=True)
        ks = [str(k) for k in folds]
        print(f"{'model':<6}{'rule':<6}" + "".join(f"{'k=' + k:>10}" for k in ks)
              + f"{'mean+-std':>16}", flush=True)
        for name in ("xgb", "sage"):
            for rule, rn in (("rule_a", "A"), ("rule_b", "B")):
                s = summary[name][rule][metric]
                cells = "".join(f"{s['per_fold'][k]:>10.4f}" for k in ks)
                print(f"{name:<6}{rn:<6}{cells}"
                      f"{s['mean']:>8.4f}+-{s['std']:.4f}", flush=True)

    table("f1", "F1 (illicit)")
    table("pr_auc", "PR-AUC")

    print("\nTest-window positives:", flush=True)
    for k in folds:
        f = folds_out[str(k)]
        print(f"  k={k}: {f['test_positives']}/{f['test_total']} illicit "
              f"(steps {f['test_steps'][0]}-{f['test_steps'][1]})", flush=True)

    # Head-to-head and rule comparison.
    wins_f1 = wins_pr = 0
    for k in folds:
        a = folds_out[str(k)]["models"]
        x_f1 = statistics.mean(r["rule_a"]["f1"] for r in a["xgb"]["per_seed"])
        s_f1 = statistics.mean(r["rule_a"]["f1"] for r in a["sage"]["per_seed"])
        x_pr = statistics.mean(r["rule_a"]["pr_auc"] for r in a["xgb"]["per_seed"])
        s_pr = statistics.mean(r["rule_a"]["pr_auc"] for r in a["sage"]["per_seed"])
        wins_f1 += x_f1 > s_f1
        wins_pr += x_pr > s_pr
    print(f"\nRule A head-to-head: XGB beats SAGE on F1 in "
          f"{wins_f1}/{len(folds)} folds, PR-AUC in {wins_pr}/{len(folds)}.", flush=True)

    for name in ("xgb", "sage"):
        d = (summary[name]["rule_b"]["f1"]["mean"] -
             summary[name]["rule_a"]["f1"]["mean"])
        better = sum(
            summary[name]["rule_b"]["f1"]["per_fold"][str(k)] >
            summary[name]["rule_a"]["f1"]["per_fold"][str(k)] for k in folds)
        print(f"Rule B vs A ({name}): mean F1 delta {d:+.4f}, "
              f"better in {better}/{len(folds)} folds.", flush=True)

    payload = {"cut_points": folds, "seeds": seeds, "folds": folds_out,
               "summary": summary,
               "notes": ("Corrected 4-fold version; k=38 excluded. "
                         "k=34 evaluates 35-39 for reporting only; no step >k "
                         "used for training, early stopping or thresholds.")}
    with open(RESULTS / args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {RESULTS / args.out}", flush=True)


if __name__ == "__main__":
    main()
