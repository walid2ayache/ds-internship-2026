# CLAUDE.md: GNN Fraud Detection

## Goal
Detect illicit transactions in the Elliptic Bitcoin graph with GNNs (GCN, GraphSAGE, GAT),
compare against an XGBoost baseline, handle class imbalance, and explain predictions.
Portfolio project: clean code, honest results, a strong README.

## Stack
Python 3.12, PyTorch (CUDA), PyTorch Geometric, scikit-learn, XGBoost, matplotlib. Windows.
Always use `.venv\Scripts\python.exe` from the repo root. RTX 3070 Ti (8 GB).

## Layout
- src/data.py      loading, time steps, get_split (train 1-29 / val 30-34 / test 35-49), class_weights
- src/metrics.py   best_threshold (picked on VAL), evaluate, per_step_f1
- src/models.py    GNN(kind=gcn|sage|gat, in_dropout)
- src/train.py     train_gnn (full-batch, early stop on val PR-AUC, optional weight / focal_gamma)
- scripts/01-06    explore, split, XGB baseline, GNNs, ablation, imbalance
- scripts/07-14    report, plots, hybrid SAGE, boosting, final plots, XGB tuning, compare, final fit
- results/RESULTS.md is the single source of truth for numbers. results/*.json are the raw metrics.

## Hard rules
- Split by TIME. Unknown-label nodes (label 2) are excluded from loss and metrics but stay in the graph.
- Labels: 0 = licit, 1 = illicit, 2 = unknown.
- Report F1 / precision / recall on the illicit class and PR-AUC. Never plain accuracy.
- Same split, metrics, and threshold rule (threshold picked on VAL only) for every model.
- Model selection by VAL metric only. NEVER choose a config by looking at test numbers.
- Do not add dependencies without asking. Do not download anything without asking; give me the command.
- One task at a time. Short plan first, then wait for approval. Keep scripts small.
- Shell may be refused by the auto-mode classifier. If so, ask me to run commands with the `!` prefix.

## STATUS: experiments are DONE. Do NOT start new modelling experiments.
Findings (details and caveats in results/RESULTS.md):
- Best usable model: XGBoost on local features + degree + neighbour-contrast features.
  Train-only (val-selected): test PR-AUC 0.796, F1 0.754. Final fit on steps 1-34: PR-AUC 0.803, F1 0.787.
- Best GNN: SAGE with local features: test PR-AUC 0.712, F1 0.666. GCN/GAT much worse.
- No GNN variant tried beat local SAGE. Class weights hurt GNNs; focal loss did not help.
- The graph helps only as engineered features (small gain), not through message passing.
- VAL is saturated (~0.95) and does not rank configs; state as a limitation.
- Post step-43 F1 drop is partly prior shift and tiny support (2-56 positives per step), not only representation failure.
- 0 of 234,355 edges connect different time steps (no future leakage via edges).
- The final-fit XGB uses val labels, so compare GNN vs XGB using the TRAIN-ONLY numbers.

## Remaining work (in this order)
1. Etape 8: explainability. Explain the SAGE-local GNN with GNNExplainer, plus XGBoost feature importance
   (built in, no new dependency). Save to results/explain/.
2. Etape 9: rewrite README.md (currently an outdated stub) for a portfolio audience, using results/RESULTS.md.
   Lead with the train-only comparison. Keep the Limitations section honest.
3. Housekeeping: delete the stray torch compile-cache files in the repo root; confirm .gitignore; then git commands
   (show me, do not push).
