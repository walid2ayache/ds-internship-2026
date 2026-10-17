

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification


def euclidean_distance_vectorized(X_test, X_train):

    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)     # (M, 1)
    train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T # (1, N)
    sq_dists = test_sq + train_sq - 2.0 * (X_test @ X_train.T)
    # Clip negative values caused by floating-point imprecision
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


def manhattan_distance_vectorized(X_test, X_train):
   
    return np.sum(np.abs(X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]), axis=2)



class KNNClassifier:

    def __init__(self, k=3, metric='euclidean'):
        assert k >= 1, "k must be at least 1"
        assert metric in ('euclidean', 'manhattan'), "metric must be 'euclidean' or 'manhattan'"
        self.k = int(k)
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Stores training features and target labels.
        Complexity: O(1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        assert len(X) == len(y), "X and y must have identical length"
        assert len(X) >= self.k, f"Dataset size ({len(X)}) must be >= k ({self.k})"

        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        return self

    def _compute_pairwise_dists(self, X):
        """Returns distance matrix of shape (n_queries, n_train)."""
        if self.metric == 'euclidean':
            return euclidean_distance_vectorized(X, self.X_train)
        return manhattan_distance_vectorized(X, self.X_train)

    def predict(self, X):
       
        assert self.X_train is not None, "Classifier must be fitted first!"
        X = np.asarray(X, dtype=float)

        # Distance matrix: shape (M, N)
        dists = self._compute_pairwise_dists(X)

        # Find indices of the k smallest distances per query
        k_nearest_indices = np.argsort(dists, axis=1)[:, :self.k]

        # Retrieve neighbor labels: shape (M, k)
        k_labels = self.y_train[k_nearest_indices]

        # Plurality voting per sample
        preds = []
        for row in k_labels:
            vote_counter = Counter(row)
            # Most common label
            predicted_label = vote_counter.most_common(1)[0][0]
            preds.append(predicted_label)

        return np.array(preds)

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample in X.

        Returns
        -------
        probas : np.ndarray, shape (n_queries, n_classes)
        """
        assert self.X_train is not None, "Classifier must be fitted first!"
        X = np.asarray(X, dtype=float)
        dists = self._compute_pairwise_dists(X)
        k_indices = np.argsort(dists, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]

        probas = np.zeros((len(X), len(self.classes_)))
        for i, row in enumerate(k_labels):
            for j, c in enumerate(self.classes_):
                probas[i, j] = np.mean(row == c)

        return probas


# =============================================================================
# 3. VERIFICATION & COMPARISON WITH SCIKIT-LEARN
# =============================================================================
def test_knn_against_sklearn():
    print("=" * 60)
    print("RUNNING k-NN VALIDATION AGAINST SCIKIT-LEARN")
    print("=" * 60)

    # Generate synthetic 2-class dataset
    X, y = make_classification(
        n_samples=250,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    # Train / Test split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Test with multiple k values
    for k_val in [1, 3, 5, 7]:
        custom_knn = KNNClassifier(k=k_val, metric='euclidean')
        custom_knn.fit(X_train, y_train)
        custom_preds = custom_knn.predict(X_test)

        sk_knn = KNeighborsClassifier(n_neighbors=k_val, metric='euclidean', algorithm='brute')
        sk_knn.fit(X_train, y_train)
        sk_preds = sk_knn.predict(X_test)

        match_rate = np.mean(custom_preds == sk_preds)
        print(f"k = {k_val}: Agreement with sklearn = {match_rate * 100:.1f}%")
        assert match_rate >= 0.98, f"Predictions diverge on k={k_val}"

    # Verify predict_proba
    custom_proba = custom_knn.predict_proba(X_test)
    sk_proba = sk_knn.predict_proba(X_test)
    assert np.allclose(custom_proba, sk_proba), "predict_proba does not match sklearn!"
    print("[OK] predict_proba matches Scikit-Learn exactly!")
    print("[SUCCESS] All k-NN tests passed successfully!\n")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    test_knn_against_sklearn()
