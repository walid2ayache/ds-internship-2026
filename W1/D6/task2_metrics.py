"""
=============================================================================
WEEK 1 - DAY 6: TASK 2 - EVALUATION METRICS FROM SCRATCH
=============================================================================
Topics Covered:
  1. Classification Metrics:
     - Confusion Matrix (TP, TN, FP, FN)
     - Accuracy
     - Precision
     - Recall (Sensitivity)
     - F1-Score (Harmonic Mean)
  2. Regression Metrics:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - Coefficient of Determination (R²)
  3. Complete validation suite comparing against sklearn.metrics
=============================================================================
"""

import numpy as np
from sklearn import metrics as sk_metrics


# =============================================================================
# 1. CLASSIFICATION METRICS
# =============================================================================
def confusion_matrix(y_true, y_pred, labels=None):
    """
    Computes confusion matrix where:
    rows = true labels, columns = predicted labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    label_to_idx = {val: i for i, val in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    return cm


def accuracy_score(y_true, y_pred):
    """
    Accuracy = (TP + TN) / Total
    Fraction of correct classifications.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "Lengths must match"
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, pos_label=1):
    """
    Precision = TP / (TP + FP)
    Answers: 'Of all predicted positives, how many were right?'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(y_true, y_pred, pos_label=1):
    """
    Recall = TP / (TP + FN)
    Answers: 'Of all actual positives, how many were detected?'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Harmonic mean balancing Precision and Recall.
    """
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)

    denom = prec + rec
    return float(2 * (prec * rec) / denom) if denom > 0 else 0.0


# =============================================================================
# 2. REGRESSION METRICS
# =============================================================================
def mean_squared_error(y_true, y_pred):
    """MSE = (1/n) * sum((y_true - y_pred)^2)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "Lengths must match"
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """RMSE = sqrt(MSE)"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    """MAE = (1/n) * sum(|y_true - y_pred|)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "Lengths must match"
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """
    R² = 1 - (SS_res / SS_tot)
    SS_res = sum((y_true - y_pred)^2)
    SS_tot = sum((y_true - mean(y_true))^2)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "Lengths must match"

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


# =============================================================================
# 3. VERIFICATION AGAINST SCIKIT-LEARN
# =============================================================================
def run_metrics_verification():
    print("=" * 60)
    print("RUNNING METRICS VALIDATION AGAINST SCIKIT-LEARN")
    print("=" * 60)

    # 1. Classification Data
    y_true_c = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1])
    y_pred_c = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1])

    # Confusion matrix check
    custom_cm = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1])
    sk_cm = sk_metrics.confusion_matrix(y_true_c, y_pred_c, labels=[0, 1])
    assert np.array_equal(custom_cm, sk_cm), "Confusion matrix mismatch"
    print("[OK] Confusion Matrix matches!")

    # Classification score checks
    assert np.isclose(accuracy_score(y_true_c, y_pred_c), sk_metrics.accuracy_score(y_true_c, y_pred_c))
    print(f"[OK] Accuracy matches:  {accuracy_score(y_true_c, y_pred_c):.4f}")

    assert np.isclose(precision_score(y_true_c, y_pred_c), sk_metrics.precision_score(y_true_c, y_pred_c))
    print(f"[OK] Precision matches: {precision_score(y_true_c, y_pred_c):.4f}")

    assert np.isclose(recall_score(y_true_c, y_pred_c), sk_metrics.recall_score(y_true_c, y_pred_c))
    print(f"[OK] Recall matches:    {recall_score(y_true_c, y_pred_c):.4f}")

    assert np.isclose(f1_score(y_true_c, y_pred_c), sk_metrics.f1_score(y_true_c, y_pred_c))
    print(f"[OK] F1-score matches:  {f1_score(y_true_c, y_pred_c):.4f}")

    # 2. Regression Data
    y_true_r = np.array([12.5, 15.0, 18.2, 22.0, 30.5, 35.1, 40.0])
    y_pred_r = np.array([13.1, 14.8, 17.5, 23.1, 29.8, 36.0, 38.9])

    assert np.isclose(mean_squared_error(y_true_r, y_pred_r), sk_metrics.mean_squared_error(y_true_r, y_pred_r))
    print(f"[OK] MSE matches:       {mean_squared_error(y_true_r, y_pred_r):.4f}")

    assert np.isclose(mean_absolute_error(y_true_r, y_pred_r), sk_metrics.mean_absolute_error(y_true_r, y_pred_r))
    print(f"[OK] MAE matches:       {mean_absolute_error(y_true_r, y_pred_r):.4f}")

    assert np.isclose(r2_score(y_true_r, y_pred_r), sk_metrics.r2_score(y_true_r, y_pred_r))
    print(f"[OK] R² matches:        {r2_score(y_true_r, y_pred_r):.4f}")

    print("[SUCCESS] All classification and regression metrics validated successfully!\n")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run_metrics_verification()
