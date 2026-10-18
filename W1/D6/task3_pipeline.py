

# =============================================================================
# 1. DAY 5 PREPROCESSING BUILDING BLOCKS
# =============================================================================
def train_test_split(X, y, test_size=0.25, random_state=42):
    """Splits data into train and test partitions using shuffled indices."""
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n = len(X)
    shuffled_idx = rng.permutation(n)
    n_test = int(n * test_size)

    test_indices = shuffled_idx[:n_test]
    train_indices = shuffled_idx[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid zero division if any feature has variance 0
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X):
        assert self.mean_ is not None, "StandardScaler must be fit() first"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# =============================================================================
# 2. RUN FULL PIPELINE
# =============================================================================
def run_pipeline():
    print("=" * 60)
    print("RUNNING END-TO-END CLASSIFICATION PIPELINE")
    print("=" * 60)

    # 1. Synthesize non-linear dataset (make_moons)
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Train-Test Split (Day 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Standardize Features (Day 5)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Never fit on test!

    # 4. Train k-NN Classifier (Day 6)
    k = 5
    knn = KNNClassifier(k=k, metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    # 5. Predict & Evaluate (Day 6)
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\n--- Model Evaluation Report ---")
    print(f"k-NN (k={k}) Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")

    # 6. Plot Decision Boundary
    try:
        x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
        y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = knn.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=35, label='Train')
        plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='white', marker='^', s=60, label='Test')
        plt.title(f"Pure NumPy k-NN (k={k}) Decision Boundary on make_moons")
        plt.xlabel("Scaled Feature 1")
        plt.ylabel("Scaled Feature 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig("knn_decision_boundary.png", dpi=150)
        print("\n[OK] Decision boundary saved to 'knn_decision_boundary.png'")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\n[SUCCESS] End-to-end pipeline finished successfully!")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    run_pipeline()
