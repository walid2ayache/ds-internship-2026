r"""Etape 8 (screen): can a local-feature GNN close the gap to XGBoost?
Run from the repo root:  .venv\Scripts\python.exe scripts\09_hybrid_sage.py

Phase 0: SAGE with all 165 features vs local 93, seed 0, per-step VAL PR-AUC.
Phase 1: 8 local-feature variants x 3 seeds. Variants test depth (1 vs 2),
aggregation (mean vs max), a residual skip and a concat head, plus GAT-local.

Selection is VAL-only: mean pooled VAL PR-AUC, with late-VAL (steps 33-34,
6.3% illicit, the only pre-test slice close to test prevalence) as a tie-break
within 0.005. Test metrics are computed for reporting and never used to pick.

No existing script or src file is modified; the hybrid model lives here.
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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate
from src.models import GNN
from src.train import train_gnn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_LOCAL = 93
N_SEEDS = 3

CONFIGS = [
    ("sage2_mean", dict(layers=2, in_dropout=0.5)),
    ("sage2_mean_do0", dict(layers=2, in_dropout=0.0)),
    ("sage1_mean", dict(layers=1, in_dropout=0.5)),
    ("sage2_max", dict(layers=2, in_dropout=0.5, aggr="max")),
    ("sage2_skip", dict(layers=2, in_dropout=0.5, skip=True)),
    ("sage2_concat", dict(layers=2, in_dropout=0.5, concat=True)),
    ("sage1_skip", dict(layers=1, in_dropout=0.5, skip=True)),
    ("gat2", dict(kind="gat", in_dropout=0.5)),
]


class HybridSAGE(torch.nn.Module):
    """SAGE with optional residual skip and/or concat of the raw input at the head."""

    def __init__(self, in_dim, hidden=128, out_dim=2, layers=2,
                 in_dropout=0.5, dropout=0.5, aggr="mean",
                 skip=False, concat=False):
        super().__init__()
        self.in_dropout = in_dropout
        self.dropout = dropout
        self.skip = torch.nn.Linear(in_dim, out_dim) if skip else None
        self.concat = concat
        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            c_in = in_dim if i == 0 else hidden
            c_out = hidden if (i < layers - 1 or concat) else out_dim
            self.convs.append(SAGEConv(c_in, c_out, aggr=aggr))
        self.head = (torch.nn.Linear(hidden + in_dim, out_dim)
                     if concat else None)

    def forward(self, x, edge_index):
        x0 = x
        h = F.dropout(x, self.in_dropout, self.training)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.dropout(F.relu(h), self.dropout, self.training)
        if self.concat:
            h = self.head(torch.cat([h, x0], dim=1))
        if self.skip is not None:
            h = h + self.skip(x0)
        return h


def build_model(cfg, in_dim):
    if cfg.get("kind") == "gat":
        return GNN(in_dim, kind="gat", in_dropout=cfg.get("in_dropout", 0.5))
    return HybridSAGE(in_dim, layers=cfg.get("layers", 2),
                      in_dropout=cfg.get("in_dropout", 0.5),
                      aggr=cfg.get("aggr", "mean"),
                      skip=cfg.get("skip", False),
                      concat=cfg.get("concat", False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS,
                   help="use seeds 0..n-1 (default 3)")
    p.add_argument("--only", default=None, help="run a single config by name")
    p.add_argument("--skip-phase0", action="store_true")
    p.add_argument("--out", default="hybrid_screen.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    yv, tv = y[va_np], t[va_np]

    d_local = data.clone()
    d_local.x = data.x[:, :N_LOCAL]

    def val_metrics(p_val):
        per = {}
        for s in range(30, 35):
            m = tv == s
            per[str(s)] = (float(average_precision_score(yv[m], p_val[m]))
                           if yv[m].any() else None)
        late = tv >= 33
        return (float(average_precision_score(yv, p_val)), per,
                float(average_precision_score(yv[late], p_val[late])))

    def run_one(name, cfg, d, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(cfg, d.num_node_features)
        t0 = time.time()
        probs, best_val = train_gnn(model, d, tr, va, device=device)
        p_val, p_test = probs[va_np], probs[te_np]
        val_ap, per_step, late_ap = val_metrics(p_val)
        thr = best_threshold(yv, p_val)  # VAL only
        r = evaluate(y[te_np], p_test, thr)
        secs = time.time() - t0
        print(f"{name:<16} seed {seed} | val AP {val_ap:.4f} | "
              f"late {late_ap:.4f} | test PR {r['pr_auc']:.4f} | "
              f"{secs:.0f}s", flush=True)
        return dict(seed=seed, val_ap=val_ap, late_val_ap=late_ap,
                    per_step_val_ap=per_step, best_epoch_val_ap=float(best_val),
                    threshold=thr, test_pr_auc=r["pr_auc"], test_f1=r["f1"],
                    precision=r["precision"], recall=r["recall"],
                    seconds=secs)

    def summarize(runs):
        keys = ["val_ap", "late_val_ap", "test_pr_auc", "test_f1",
                "precision", "recall"]
        return {k: (statistics.mean([r[k] for r in runs]),
                    statistics.pstdev([r[k] for r in runs])) for k in keys}

    out_path = RESULTS / args.out
    phase0 = None
    runs = {}

    def dump():
        payload = {"phase0": phase0, "runs": runs,
                   "note": "Selection uses VAL only; test is reporting only."}
        if runs:
            payload["summary"] = {n: summarize(rs) for n, rs in runs.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if not args.skip_phase0 and args.only is None:
        print("\n== Phase 0: all features (seed 0) ==", flush=True)
        phase0 = run_one("phase0_all", dict(layers=2, in_dropout=0.5), data, 0)
        dump()

    grids = [c for c in CONFIGS if args.only is None or c[0] == args.only]
    assert grids, f"unknown config: {args.only}"
    seeds = list(range(args.seeds))

    for name, cfg in grids:
        print(f"\n== {name} ({len(seeds)} seeds) ==", flush=True)
        runs[name] = []
        for seed in seeds:
            runs[name].append(run_one(name, cfg, d_local, seed))
        dump()

    summary = {n: summarize(rs) for n, rs in runs.items()}
    ranked = sorted(summary, key=lambda k: summary[k]["val_ap"][0],
                    reverse=True)
    winner = ranked[0]
    if len(ranked) > 1:
        top = summary[winner]["val_ap"][0]
        tied = [k for k in ranked if top - summary[k]["val_ap"][0] <= 0.005]
        winner = max(tied, key=lambda k: summary[k]["late_val_ap"][0])
        print(f"\nVAL tie within 0.005: {tied}", flush=True)

    print("\n=== local-feature configs: mean over seeds ===", flush=True)
    print(f"{'config':<18}{'val AP':>10}{'late AP':>10}"
          f"{'test PR':>10}{'test F1':>10}", flush=True)
    for name in ranked:
        m = summary[name]
        print(f"{name:<18}{m['val_ap'][0]:>10.4f}{m['late_val_ap'][0]:>10.4f}"
              f"{m['test_pr_auc'][0]:>10.4f}{m['test_f1'][0]:>10.4f}",
              flush=True)

    if phase0 and "sage2_mean" in runs:
        local0 = runs["sage2_mean"][0]
        print("\nPhase 0 VAL PR-AUC per step (seed 0):", flush=True)
        print(f"{'step':<6}{'all 165':>10}{'local 93':>10}", flush=True)
        for s in range(30, 35):
            a = phase0["per_step_val_ap"][str(s)]
            b = local0["per_step_val_ap"][str(s)]
            print(f"{s:<6}{a:>10.4f}{b:>10.4f}", flush=True)

    print(f"\nWinner by VAL PR-AUC (late-VAL tie-break): {winner}", flush=True)
    m = summary[winner]
    print(f"  val AP {m['val_ap'][0]:.4f} +- {m['val_ap'][1]:.4f} | "
          f"late {m['late_val_ap'][0]:.4f} | "
          f"test PR {m['test_pr_auc'][0]:.4f} | "
          f"test F1 {m['test_f1'][0]:.4f}", flush=True)

    payload = {"phase0": phase0, "runs": runs, "summary": summary,
               "winner": winner, "seeds": seeds,
               "note": "Selection uses VAL only; test is reporting only."}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
