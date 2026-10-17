"""
=============================================================================
WEEK 1 - DAY 6: BLANK-FILE TEST (TIMED 60 MINUTES)
=============================================================================
Instructions:
  1. This test assesses your ability to write core Python, NumPy, and ML
     algorithms from memory with NO external documentation or AI assistance.
  2. Total Time Allowed: 60 Minutes.
  3. Total Points: 60 Points.
  4. Fill in the empty function/class templates below and run this file:
     python blank_file_test.py
=============================================================================
"""

import string
import numpy as np


# =============================================================================
# PROBLEM 1: PURE PYTHON STATISTICS & STRING PROCESSING (15 POINTS)
# =============================================================================
# Rule: Do NOT import numpy or pandas for Problem 1. Use pure standard Python.

def mean(numbers):
    """
    Returns the arithmetic mean of a list of numbers.
    Formula: sum(x) / n
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def variance(numbers, sample=True):
    """
    Returns variance of a list of numbers.
    sample=True  -> divide by (n - 1) (Bessel's correction)
    sample=False -> divide by n (population variance)
    """
    n = len(numbers)
    if n <= 1 and sample:
        return 0.0
    m = mean(numbers)
    squared_diffs = sum((x - m) ** 2 for x in numbers)
    return squared_diffs / (n - 1 if sample else n)


def std_dev(numbers, sample=True):
    """Returns standard deviation: sqrt(variance)."""
    return variance(numbers, sample=sample) ** 0.5


def top_k_words(text, k=3):
    """
    Takes a raw string, converts to lowercase, strips punctuation,
    and returns a list of (word, count) tuples for the top k most frequent words,
    sorted by frequency descending.
    """
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(translator).lower()
    words = clean_text.split()

    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_words[:k]


# =============================================================================
# PROBLEM 2: NUMPY VECTORIZATION & PREPROCESSING (20 POINTS)
# =============================================================================

class StandardScaler:
    """
    Feature standardizer: z = (x - mu) / sigma.
    Fit on training data; transform both train and test.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Handle zero variance columns gracefully
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X):
        assert self.mean_ is not None, "Scaler must be fit() before transform()"
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def pairwise_euclidean_matrix(A, B):
    """
    Computes pairwise Euclidean distance matrix between rows of A and rows of B.
    Shape of A: (M, d)
    Shape of B: (N, d)
    Shape of output: (M, N)

    CRITICAL RULE: MUST NOT use any Python for-loops!
    Use broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T * b
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    A_sq = np.sum(A ** 2, axis=1, keepdims=True)      # (M, 1)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True).T  # (1, N)
    dists_sq = A_sq + B_sq - 2.0 * (A @ B.T)
    dists_sq = np.maximum(dists_sq, 0.0)
    return np.sqrt(dists_sq)


# =============================================================================
# PROBLEM 3: DATA SPLIT & LINEAR REGRESSION NORMAL EQUATION (20 POINTS)
# =============================================================================

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits feature matrix X and target y into train/test subsets.
    Returns: X_train, X_test, y_train, y_test
    """
    assert len(X) == len(y), "X and y must be same length"
    n = len(X)
    rng = np.random.default_rng(random_state)
    shuffled_idx = rng.permutation(n)

    n_test = int(n * test_size)
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class LinearRegressionNormal:
    """
    Linear Regression using the closed-form Normal Equation:
    w = (X_b^T @ X_b)^(-1) @ X_b^T @ y
    """
    def __init__(self):
        self.weights_ = None  # Includes weights and intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]

        # Add column of 1s for bias term (intercept)
        X_b = np.column_stack([X, np.ones(n)])

        # Solve normal equation via pseudo-inverse
        self.weights_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        assert self.weights_ is not None, "Model not fitted yet"
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        X_b = np.column_stack([X, np.ones(n)])
        return X_b @ self.weights_


# =============================================================================
# AUTOMATED TEST & SCORING RUNNER (5 POINTS CLEAN CODE)
# =============================================================================
def test_all():
    print("=" * 60)
    print("RUNNING 60-MINUTE BLANK-FILE TEST SUITE")
    print("=" * 60)
    score = 0

    # 1. Test Problem 1
    data = [10, 20, 30, 40, 50]
    assert mean(data) == 30.0, "Mean incorrect"
    assert variance(data, sample=True) == 250.0, "Sample variance incorrect"
    assert abs(std_dev(data, sample=True) - (250 ** 0.5)) < 1e-6, "Std dev incorrect"

    txt = "Data science is great. Machine learning is great and science is fun!"
    top2 = top_k_words(txt, k=2)
    assert top2[0][0] == "is" and top2[0][1] == 3, "top_k_words incorrect"
    score += 15
    print("[PASSED] Problem 1 Passed! (+15 pts)")

    # 2. Test Problem 2
    X_sample = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0, 0.0]), "Mean not 0"
    assert np.allclose(np.std(X_scaled, axis=0), [1.0, 1.0]), "Std not 1"

    A = np.array([[0.0, 0.0], [3.0, 4.0]])
    B = np.array([[0.0, 0.0], [1.0, 0.0]])
    dists = pairwise_euclidean_matrix(A, B)
    expected_dists = np.array([[0.0, 1.0], [5.0, ((2**2 + 4**2)**0.5)]])
    assert np.allclose(dists, expected_dists), "Pairwise distance matrix incorrect"
    score += 20
    print("[PASSED] Problem 2 Passed! (+20 pts)")

    # 3. Test Problem 3
    np.random.seed(42)
    X_reg = np.linspace(0, 10, 50).reshape(-1, 1)
    y_reg = 2.5 * X_reg.ravel() + 4.0

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    assert len(X_train) == 40 and len(X_test) == 10, "Split lengths incorrect"

    model = LinearRegressionNormal()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert np.allclose(preds, y_test, atol=1e-5), "Linear regression normal fit incorrect"
    score += 20
    print("[PASSED] Problem 3 Passed! (+20 pts)")

    # Code cleanliness bonus
    score += 5
    print("[PASSED] Code Cleanliness & Structure (+5 pts)")

    print("=" * 60)
    print(f"[FINAL] SCORE: {score} / 60 POINTS (100%) - EXCELLENT WORK!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    test_all()
