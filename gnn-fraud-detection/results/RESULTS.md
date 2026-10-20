# Results: Elliptic Bitcoin fraud detection

Last updated 2026-10-20 with the Phase 1/2 experiments (`scripts/09_hybrid_sage.py`,
`scripts/10_boost.py`), the XGB tuning attempt (`scripts/12_xgb_tune.py`,
`scripts/13_compare.py`), the train+val final fit (`scripts/14_final_fit.py`) and the
rolling-origin robustness check (`scripts/16_rolling.py`, `results/rolling.json`). The
original 26-config table comes from `scripts/07_report.py`; Phase 1/2 numbers come from
`results/hybrid_screen.json` and `results/boost_metrics.json`; the final fit is in
`results/final_fit.json` and the tuning attempt in `results/xgb_compare.json`.

Time split: train steps 1–29 / val 30–34 / test 35–49. Unknown-label nodes are excluded
from loss and metrics but stay in the graph. All numbers are illicit-class metrics on
labeled test nodes (steps 35–49); the decision threshold maximises illicit F1 on VAL and is
then applied unchanged to test. Test illicit rate is 6.5% vs 16.8% on val (distribution shift).

## Best model

| Model | Training | VAL PR-AUC | Test PR-AUC | Test F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| **XGB final fit (5 seeds)** | steps 1–34, 332 trees, threshold from train-only | n/a¹ | **0.8028** | **0.7870** | 0.856 | 0.729 |
| XGB train-only (VAL-selected best) | steps 1–29, early stopping | 0.9912 | 0.7959 | 0.7537 | 0.782 | 0.728 |
| SAGE local (best GNN) | steps 1–29 | 0.9301 | 0.7116 | 0.6660 | 0.947 | 0.514 |

¹ VAL labels are used in the final fit, so its VAL PR-AUC is not a held-out number.

**Final fit (adopted for the deployment-style headline).** The 188-feature XGB is retrained
on steps 1–34 with trees fixed at 332 (the train-only early-stopping count), no early
stopping, then 5 seeds are averaged. The threshold stays 0.39 from the train-only model.
Test PR-AUC rises from 0.7959 to 0.8028 (Δ **+0.0069**, 95% CI **[+0.0039, +0.0099]**) and
F1 from 0.7537 to 0.7870 (Δ **+0.0333**, **[+0.0240, +0.0426]**); precision improves from
0.782 to 0.856 at nearly the same recall. This is the first change that improves both
metrics significantly. Because it uses val labels for training it is not directly
comparable to the VAL-selected table; the test set was never used to choose the config or
the threshold.

**VAL-selected best (train-only, comparable).** Without val labels, the best model is the
same 188-feature XGB trained on 1–29: test PR 0.7959 / F1 0.7537. It drops the 72 aggregated
features and adds two log-degree features plus 93 neighbour-contrast features
(`x_local - mean(neighbour x_local)`), all label-free. Against the unweighted all-feature
XGB (0.7900) the gain is +0.0058 [+0.0003, +0.0111]; against the weighted old best (0.7895)
it is +0.0064. This is the number to quote when comparing VAL-selected models.

(`ensemble_rank` has a marginally higher VAL AP, 0.9919, but is unusable under a single
threshold; see Phase 2.)

## Phase 1: architecture screen (local 93, 3 seeds)

Goal: test whether depth, aggregation, a residual skip, a concat head, or GAT-local can
close the GNN gap. Selection was mean VAL PR-AUC, late-VAL (steps 33–34) as tie-break.

| Config | VAL AP | Late-VAL 33–34 | Test PR | Test F1 |
|---|---:|---:|---:|---:|
| SAGE 2L, in-drop 0.0 | 0.9319 | **0.8126** | 0.6053 | 0.7098 |
| SAGE 2L + concat | 0.9303 | 0.7602 | 0.6547 | 0.6765 |
| SAGE 2L baseline | 0.9299 | 0.7587 | **0.7115** | 0.6777 |
| SAGE 2L max-aggr | 0.9250 | 0.7282 | 0.7118 | 0.6626 |
| SAGE 2L + residual skip | 0.9139 | 0.7543 | 0.6650 | 0.6758 |
| GAT 2L local | 0.8438 | 0.5222 | 0.4856 | 0.4764 |
| SAGE 1L + skip | 0.4792 | 0.3871 | 0.1753 | 0.3265 |
| SAGE 1L | 0.4440 | 0.4372 | 0.1777 | 0.3131 |

No architecture change meaningfully beat the local SAGE baseline on test PR-AUC
(max aggregation ties it at 0.7118 vs 0.7115, inside seed noise). Max aggregation matches
mean; skip/concat hurt; GAT-local is poor; 1-layer SAGE collapses to a near-linear model with
no hidden nonlinearity.

Phase 0 made the validation problem explicit. SAGE with all 165 features won **every** val
step (per-step VAL AP 0.9495/0.9388/0.9739/0.8384/0.7727 for steps 30–34 vs
0.9193/0.9213/0.9561/0.8267/0.7157 for local), yet tested at 0.5918 vs 0.7099. The val-selected
winner from Phase 1 (`in-drop 0.0`, late-VAL 0.813) also lost on test (0.605 vs 0.712).
VAL does not rank these models.

## Phase 2: engineered features and ensembles

| Model | VAL AP | Test PR | Test F1 |
|---|---:|---:|---:|
| ensemble_rank (XGB + SAGE, ranks) | **0.9919** | 0.7959 | 0.4628 |
| **XGB local + degree + contrast (unweighted)** | **0.9912** | **0.7959** | **0.7537** |
| ensemble_prob (chose w_xgb = 1.0) | 0.9912 | 0.7959 | 0.7537 |
| XGB all 165, class weights (old best) | 0.9886 | 0.7895 | 0.7417 |
| XGB all 165 + degree + contrast, weighted | 0.9876 | 0.7960 | 0.7439 |
| XGB local + degree + contrast, weighted | 0.9851 | 0.7811 | 0.7601 |
| XGB all 165, unweighted | 0.9816 | 0.7900 | 0.6781 |
| XGB all 165 + degree + contrast, unweighted | 0.9789 | 0.7956 | 0.6869 |
| SAGE local + degree + contrast | 0.9426 | 0.6718 | 0.6694 |
| SAGE local (established GNN) | 0.9301 | 0.7116 | 0.6660 |
| XGB local + degree + contrast, per-step z-score | 0.7002 | 0.7190 | 0.5260 |

Notes:
- The rank ensemble wins VAL by 0.0007, but rank scores are uniform by construction and the
  single VAL threshold does not transfer: test F1 collapses to 0.4628 (precision 0.325).
  It is not a usable model and is not recommended.
- The probability-scale ensemble selects `w_xgb = 1.0`, i.e. XGBoost alone; the GNN adds
  nothing on VAL.
- Among valid probability-scale candidates the VAL winner is XGB local + degree + contrast
  (0.9912), tied with the degenerate ensemble.
- Per-step z-scoring destroys VAL (0.700); feature drift is real but normalising it away
  removes signal.
- SAGE with the new features improves VAL (0.930→0.943) but drops test (0.712→0.672), the
  same val/test inversion seen throughout.

Provenance: the first Phase 2 batch contained the two plain XGB baselines, both
degree+contrast XGB variants, the two SAGE variants, and the rank ensemble. The weighted
local variant, the probability-scale ensemble, and the per-step z-score variant were added
afterwards as exploratory checks; none beat the first-batch winner on VAL.

### Tuning attempt (not adopted)

A pre-registered XGB grid (12 configs × 4 feature sets × 2 rolling-origin folds inside train)
selected base + top-32 aggregated features, depth 7, 300 trees, `min_child_weight=1`, with
CV mean PR-AUC 0.9371. Five seeds averaged gave test PR-AUC 0.7997 but F1 0.7066.

Against the current model, paired node bootstrap gives ΔPR **+0.0039 [95% −0.0007, +0.0083]**
(not distinguishable from zero) and ΔF1 **−0.0469 [−0.0609, −0.0336]** (significantly worse).
The tuned model's VAL threshold is 0.22 vs 0.39, so it over-predicts at the 6.5% test
prevalence. It is **not adopted**; the current model stays the best usable one.

## Original screen (before Phase 1/2)

| # | Family | Config | Features | Loss / imbalance | VAL PR-AUC | Test PR-AUC | Test F1 | Precision | Recall | F1 steps 35–42 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | XGBoost (no graph) | XGBoost ★ | all | class weights | — | 0.789 | 0.742 | 0.744 | 0.740 | 0.863 |
| 2 | XGBoost (no graph) | XGBoost | local | class weights | — | 0.775 | 0.730 | 0.753 | 0.709 | 0.834 |
| 3 | XGBoost (no graph) | XGBoost | local | plain | — | 0.770 | 0.736 | 0.799 | 0.682 | 0.836 |
| 4 | XGBoost (no graph) | XGBoost | all | plain | — | 0.790 | 0.678 | 0.621 | 0.747 | 0.821 |
| 5 | GNN (main) | SAGE ★ | all | plain | 0.951 ± 0.003 | 0.564 ± 0.023 | 0.563 ± 0.008 | 0.542 ± 0.014 | 0.585 ± 0.009 | — |
| 6 | GNN (main) | SAGE | all | class weights | 0.938 ± 0.003 | 0.486 ± 0.043 | 0.477 ± 0.009 | 0.395 ± 0.009 | 0.604 ± 0.014 | — |
| 7 | GNN (main) | GAT | all | plain | 0.892 ± 0.003 | 0.392 ± 0.018 | 0.416 ± 0.012 | 0.334 ± 0.021 | 0.555 ± 0.016 | — |
| 8 | GNN (main) | GAT | all | class weights | 0.884 ± 0.002 | 0.358 ± 0.012 | 0.387 ± 0.017 | 0.309 ± 0.029 | 0.522 ± 0.021 | — |
| 9 | GNN (main) | GCN | all | plain | 0.856 ± 0.006 | 0.354 ± 0.009 | 0.384 ± 0.005 | 0.358 ± 0.029 | 0.421 ± 0.032 | — |
| 10 | GNN (main) | GCN | all | class weights | 0.809 ± 0.016 | 0.333 ± 0.010 | 0.377 ± 0.015 | 0.342 ± 0.042 | 0.430 ± 0.035 | — |
| 11 | Ablation | SAGE (in-drop 0.5) ★ | all | plain | 0.951 ± 0.003 | 0.563 ± 0.023 | 0.557 ± 0.001 | 0.528 ± 0.011 | 0.590 ± 0.015 | 0.643 ± 0.006 |
| 12 | Ablation | SAGE (in-drop 0.0) | local | plain | 0.932 ± 0.003 | 0.620 ± 0.037 | 0.705 ± 0.005 | 0.833 ± 0.031 | 0.612 ± 0.019 | 0.817 ± 0.009 |
| 13 | Ablation | SAGE (in-drop 0.5) | local | plain | 0.930 ± 0.003 | 0.713 ± 0.003 | 0.670 ± 0.012 | 0.940 ± 0.022 | 0.521 ± 0.020 | 0.749 ± 0.013 |
| 14 | Ablation | SAGE (in-drop 0.0) | all | plain | 0.909 ± 0.006 | 0.617 ± 0.012 | 0.540 ± 0.007 | 0.461 ± 0.015 | 0.654 ± 0.011 | 0.633 ± 0.005 |
| 15 | Ablation | GCN (in-drop 0.5) | all | plain | 0.856 ± 0.006 | 0.352 ± 0.007 | 0.379 ± 0.002 | 0.342 ± 0.024 | 0.431 ± 0.032 | 0.466 ± 0.009 |
| 16 | Ablation | GCN (in-drop 0.0) | all | plain | 0.819 ± 0.006 | 0.370 ± 0.007 | 0.437 ± 0.025 | 0.414 ± 0.048 | 0.469 ± 0.040 | 0.531 ± 0.034 |
| 17 | Ablation | GCN (in-drop 0.5) | local | plain | 0.816 ± 0.003 | 0.384 ± 0.002 | 0.433 ± 0.002 | 0.446 ± 0.006 | 0.420 ± 0.003 | 0.541 ± 0.001 |
| 18 | Ablation | GCN (in-drop 0.0) | local | plain | 0.772 ± 0.016 | 0.299 ± 0.049 | 0.418 ± 0.032 | 0.376 ± 0.062 | 0.483 ± 0.031 | 0.555 ± 0.025 |
| 19 | Imbalance | SAGE ★ | all | plain | 0.950 ± 0.004 | 0.599 ± 0.032 | 0.569 ± 0.014 | 0.548 ± 0.035 | 0.594 ± 0.019 | 0.657 ± 0.012 |
| 20 | Imbalance | SAGE | all | sqrt weights | 0.947 ± 0.002 | 0.545 ± 0.024 | 0.528 ± 0.012 | 0.481 ± 0.027 | 0.588 ± 0.012 | 0.631 ± 0.011 |
| 21 | Imbalance | SAGE | all | focal γ=2 | 0.944 ± 0.006 | 0.606 ± 0.021 | 0.568 ± 0.024 | 0.559 ± 0.063 | 0.584 ± 0.023 | 0.654 ± 0.024 |
| 22 | Imbalance | SAGE | all | class weights | 0.940 ± 0.002 | 0.497 ± 0.038 | 0.478 ± 0.019 | 0.392 ± 0.029 | 0.614 ± 0.016 | 0.605 ± 0.014 |
| 23 | Imbalance | SAGE | local | sqrt weights | 0.930 ± 0.004 | 0.700 ± 0.006 | 0.671 ± 0.010 | 0.852 ± 0.078 | 0.561 ± 0.050 | 0.760 ± 0.018 |
| 24 | Imbalance | SAGE | local | plain | 0.929 ± 0.003 | 0.712 ± 0.003 | 0.679 ± 0.015 | 0.896 ± 0.059 | 0.551 ± 0.043 | 0.762 ± 0.020 |
| 25 | Imbalance | SAGE | local | class weights | 0.928 ± 0.004 | 0.675 ± 0.004 | 0.655 ± 0.006 | 0.832 ± 0.027 | 0.541 ± 0.015 | 0.746 ± 0.007 |
| 26 | Imbalance | SAGE | local | focal γ=2 | 0.926 ± 0.003 | 0.700 ± 0.006 | 0.658 ± 0.006 | 0.920 ± 0.024 | 0.512 ± 0.014 | 0.737 ± 0.007 |

Table notes: XGBoost rows show `—` under VAL PR-AUC because `03_baseline.py` records VAL F1
only. The six GNN (main) rows show `—` under F1 steps 35–42 because `04_gnn.py` stored
per-step F1, not the pooled number.

Family picks by VALIDATION only (original screen):

| Family | Runs | Selected config | Rule | VAL score | Test PR-AUC* | Test F1* |
|---|---|---|---:|---:|---:|---:|
| XGBoost (no graph) | 1 run | XGBoost · all · class weights | VAL F1 | 0.961 | 0.789 | 0.742 |
| GNN (main) | 3 seeds | SAGE · all · plain | VAL PR-AUC | 0.951 | 0.564 ± 0.023 | 0.563 ± 0.008 |
| Ablation | 3 seeds | SAGE (in-drop 0.5) · all · plain | VAL PR-AUC | 0.951 | 0.563 ± 0.023 | 0.557 ± 0.001 |
| Imbalance | 5 seeds | SAGE · all · plain | VAL PR-AUC | 0.950 | 0.599 ± 0.032 | 0.569 ± 0.014 |

`*` Test columns are reference only and were never used to choose a config.

## Rolling-origin evaluation

Corrected 4-fold version (`scripts/16_rolling.py`, `results/rolling.json`). Cut points
k = 20, 25, 30, 34; for each fold, train on steps 1..k-3, validate on k-2..k, test on
k+1..min(k+5, 49). XGBoost uses the 188-feature train-only config; SAGE uses the 93 local
features with in-dropout 0.5; 3 seeds each. Rule A is the F1-optimal threshold on the fold's
validation block. Rule B pools the test predictions of earlier folds whose window ended on
or before k and falls back to A for the first fold. Nothing after k is used for training,
early stopping or thresholds. k=38 was excluded: it would train on step 35 and validate on
36–38, which are old-test steps.

F1 (illicit), mean over 3 seeds:

| Model | Rule | k=20 | k=25 | k=30 | k=34 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | A | 0.8169 | 0.7338 | 0.9404 | 0.8697 | 0.8402 ± 0.0755 |
| XGBoost | B | 0.8169 | 0.7365 | 0.9236 | 0.8257 | 0.8257 ± 0.0664 |
| SAGE local | A | 0.6881 | 0.6702 | 0.8798 | 0.8057 | 0.7609 ± 0.0861 |
| SAGE local | B | 0.6881 | 0.6732 | 0.8433 | 0.7790 | 0.7459 ± 0.0693 |

PR-AUC (threshold-free, so A and B are identical):

| Model | Rule | k=20 | k=25 | k=30 | k=34 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | A/B | 0.8341 | 0.7622 | 0.9876 | 0.9552 | 0.8848 ± 0.0910 |
| SAGE local | A/B | 0.7230 | 0.7213 | 0.9235 | 0.8783 | 0.8115 ± 0.0908 |

| Fold | Test steps | Illicit / labeled |
|---|---:|---:|
| k=20 | 21–25 | 566 / 5,311 |
| k=25 | 26–30 | 617 / 2,705 |
| k=30 | 31–35 | 690 / 4,330 |
| k=34 | 35–39 | 447 / 5,486 |

Interpretation:

- **XGBoost wins 4/4 folds** on both F1 and PR-AUC; SAGE never wins. The single-split
  ordering is stable across these cut points.
- **Rule B does not help F1.** Mean F1 falls by 0.0145 for XGBoost and 0.0150 for SAGE;
  it improves only at k=25 (by +0.0027 and +0.0030) and is worse at k=30 and k=34.
  PR-AUC is unchanged by construction. Pooled thresholds from earlier folds/models do not
  transfer; keep the single-fold VAL threshold.
- **Scores vary strongly by fold** (XGBoost F1 0.734–0.940, PR-AUC 0.762–0.988). This
  tracks the shorter training windows at k=20/25 and the higher-prevalence windows at
  k=30/34. The mean ± std is descriptive across four correlated folds, not an estimate of
  one deployment performance; do not compare it with the 0.796/0.803 headline. Std is
  population std (ddof=0) across folds; per-seed values are in `rolling.json`.
- **Caveats.** k=34 evaluates steps 35–39 and k=30 evaluates step 35; both are inside the
  old test range but are used for reporting only — no step from 35–49 was used for
  training, early stopping or thresholds. k=30 and k=34 share step 35, so the folds are
  not independent and the std understates uncertainty. Fold windows have 447–690
  positives, but per-step support inside them is uneven (e.g. 33 illicit at step 36), so
  fold F1 is noisy. Four folds and three seeds is a small sample.
- **What it does not show.** It does not establish that XGBoost is better at every future
  cut point, and it does not improve any model. It shows the ordering is stable across
  these four cut points and that pooled thresholds are worse than the current rule.

## Limitations

1. **VAL is saturated and anti-correlated with test.** SAGE reaches ~0.95 VAL PR-AUC with
   seed std up to 0.006, and the original screen's VAL winner (SAGE all) tests at 0.564 vs
   0.712 for the lower-VAL local model. Phase 1/2 reproduced this repeatedly: all-features
   wins every val step but loses test; in-drop 0.0 wins the late-VAL tie-break but loses test;
   SAGE + degree/contrast improves VAL but lowers test. VAL cannot rank these models.
2. **GNN ceiling is ~0.71 PR-AUC.** Residual skips, concat heads, max aggregation, GAT-local,
   1-layer models, and graph-derived features all failed to beat the local SAGE baseline.
   The graph is 49 disconnected snapshots with mean degree 2.3 and illicit-neighbour purity
   of 37%, so mean smoothing dilutes the positive class. This is a negative result to state,
   not hide.
3. **Rank ensembles are invalid under the single-threshold rule.** Rank scores are uniform by
   construction and the VAL threshold does not transfer; test F1 collapsed to 0.463.
4. **Class weighting hurts the GNN** (SAGE all: 0.564→0.486 PR-AUC). Focal loss raises
   all-feature PR-AUC to 0.606 but leaves F1 flat (0.568 vs 0.563) and stays below local
   SAGE; imbalance is not the bottleneck.
5. **Threshold/prior shift.** VAL is 16.8% illicit, test 6.5%; the pooled VAL threshold is
   dominated by step 32 (25.9% illicit, 342 of 591 val positives). F1 after step 43 is based
   on 2–56 positives per step, so the "collapse" is partly prior shift and tiny support, not
   only representation failure. Report steps 35–42 and 43–49 separately.
6. **The XGB gain is protocol-dependent.** Train-only, the +0.0058 PR gain is small and the
   CI only just excludes zero. The final-fit variant adds +0.0069 PR and +0.0333 F1 but
   trains on val labels. Node bootstrap ignores graph/time dependence: treat the CIs as
   indicative, not causal.
7. **The final fit uses val labels.** Steps 1–34 are used for training, so val is no longer a
   held-out selection set and its PR-AUC is not reported. The tree count (332) and threshold
   (0.39) come from the train-only model, and test was untouched. The 0.8028/0.7870 headline
   is deployment-style, not a VAL-selected comparison; quote 0.7959/0.7537 for that.
8. **Test discipline.** The Phase 1 grid and the final-fit protocol were pre-registered.
   Phase 2 was exploratory: after the first batch, the weighted local variant, probability
   ensemble, and per-step z-score variant were added and are reported for completeness. The
   recommended model was in the first batch and none of the additions beat it on VAL. Test
   numbers are reported for every candidate and were not used to switch the final choice.
9. **Figures** (`per_step_f1.png`, `pr_curves.png`) show the final-fit XGB, the train-only
   XGB, and SAGE-local, generated by `11_final_plots.py`.
