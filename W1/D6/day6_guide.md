# 🗓️ Day 6 — ML from Scratch (Part 2) + Blank-File Test — Full Learning Guide

---

## 🎯 Today's Roadmap

1. **k-Nearest Neighbors (k-NN) from Scratch:**
   - The intuition of instance-based learning & non-parametric methods
   - Distance metrics: Euclidean ($L_2$) and Manhattan ($L_1$)
   - Vectorized pairwise distance calculation in NumPy (broadcast vs dot product trick)
   - Voting logic, tie-breaking, and finding optimal $k$
   - Full OOP implementation (`KNNClassifier`) matching Scikit-Learn API
2. **Evaluation Metrics from Scratch:**
   - The Confusion Matrix (TP, TN, FP, FN)
   - Classification metrics: Accuracy, Precision, Recall, F1-Score
   - Regression metrics: Mean Squared Error (MSE), RMSE, MAE, and $R^2$ Score
   - Edge cases & numeric stability (zero division)
3. **End-to-End Pipeline Integration:**
   - Combining Day 5 (`train_test_split`, `StandardScaler`) with Day 6 (`KNNClassifier`, metrics)
   - Benchmark validation against `sklearn`
4. **The 60-Minute Blank-File Test:**
   - What it is and why it guarantees mastery
   - 3 rigorous timed challenges from Days 1–5
   - Self-scoring rubric & solution walkthroughs
5. **Clean Notebook Guide:**
   - Structuring `from_scratch.ipynb` for your portfolio & git push

---

## Part 1: k-Nearest Neighbors (k-NN) — The Theory

### 1.1 · What is k-Nearest Neighbors?

k-Nearest Neighbors (k-NN) is one of the simplest yet most intuitive machine learning algorithms. Its core philosophy is summarized by the proverb:

> *"Birds of a feather flock together."* — Or in ML terms: **Query samples will share properties with the training samples closest to them in feature space.**

Unlike Linear Regression (Day 5), k-NN has two distinct architectural properties:

1. **Non-parametric:** It makes **zero assumptions** about the underlying data distribution or functional form (unlike linear regression, which assumes a linear plane $y = Xw$).
2. **Instance-based ("Lazy Learning"):** It does **not** learn an explicit set of weights during training.
   - **Training phase (`fit`):** Simply memorizes/stores the training dataset $X_{train}$ and $y_{train}$. Computational complexity is $O(1)$.
   - **Prediction phase (`predict`):** Does all the heavy lifting! For every query point, it calculates distances to **all** stored training samples, sorts them, selects the $k$ closest neighbors, and aggregates their labels. Computational complexity is $O(N_{train} \cdot d)$.

```
      Feature 2 ^
                |       Class A (●)
                |     ●       ●
                |        ●  ? (Query Point)
                |          ▲
                |          │ Euclidean Distance
                |     ■    ▼   ■
                |        ■       Class B (■)
                +--------------------------> Feature 1
```

---

### 1.2 · The Algorithm Step-by-Step

When given a new test point $\mathbf{x}_{test}$:

```
1. Calculate the distance between x_test and every training point x_train[i].
2. Sort distances in ascending order and get the indices of the k smallest distances.
3. Retrieve the corresponding target labels y_train for those k indices.
4. Classification: Take the majority vote (mode) of the k neighbor labels.
   Regression (k-NN Regressor): Take the average (mean) of the k neighbor values.
5. Return the predicted label.
```

---

### 1.3 · Distance Metrics: Euclidean vs Manhattan

To determine who is "closest", we need a mathematical definition of distance between two vectors $\mathbf{p} = [p_1, p_2, \dots, p_d]$ and $\mathbf{q} = [q_1, q_2, \dots, q_d]$.

#### 1. Euclidean Distance ($L_2$ Norm)
The standard straight-line distance ("as the crow flies"):

$$d_2(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{j=1}^{d} (p_j - q_j)^2} = \|\mathbf{p} - \mathbf{q}\|_2$$

In pure NumPy for two 1D vectors:
```python
dist = np.sqrt(np.sum((p - q) ** 2))
# or using np.linalg.norm:
dist = np.linalg.norm(p - q)
```

#### 2. Manhattan Distance ($L_1$ Norm / City Block Distance)
The grid-like distance traveled when moving along orthogonal axes (like navigating Manhattan city streets):

$$d_1(\mathbf{p}, \mathbf{q}) = \sum_{j=1}^{d} |p_j - q_j| = \|\mathbf{p} - \mathbf{q}\|_1$$

In pure NumPy:
```python
dist = np.sum(np.abs(p - q))
```

#### Comparison:
| Metric | Formula | When to Use | Sensitivity to Outliers |
|---|---|---|---|
| **Euclidean ($L_2$)** | $\sqrt{\sum (p_j - q_j)^2}$ | Continuous features, isotropic space, standard default | High (squares differences) |
| **Manhattan ($L_1$)** | $\sum \|p_j - q_j\|$ | Grid layouts, high-dimensional spaces, sparse features | Lower / More robust |

---

### 1.4 · The Critical Importance of Feature Scaling

> [!CAUTION]
> **Never run k-NN on unscaled data!**
> Distance calculations treat all feature axes equally. If one feature has values from $10,000$ to $50,000$ (e.g. Salary in \$) and another ranges from $1$ to $5$ (e.g. Years of Experience), the distance will be **99.99% dominated by Salary**, completely ignoring Experience.
>
> **Always apply `StandardScaler` (Day 5) before training k-NN.**

---

### 1.5 · Vectorized Distance Computations in NumPy

A naive implementation uses nested Python loops:
- Loop over every test point $i$ in $N_{test}$
- Loop over every training point $j$ in $N_{train}$
- Loop over every feature $k$ in $d$

In Python, nested loops are **extremely slow** ($100\times - 1000\times$ slower than C/NumPy). Let's look at how to vectorize this properly:

#### Method A: Semi-Vectorized (Loop over test queries, vectorize across all train points)
For a single test vector `x` of shape `(d,)` and training matrix `X_train` of shape `(N_train, d)`:

```python
# NumPy automatically broadcasts x across all rows of X_train!
diffs = X_train - x                # shape: (N_train, d)
sq_dists = np.sum(diffs ** 2, axis=1) # shape: (N_train,)
distances = np.sqrt(sq_dists)       # shape: (N_train,)
```

#### Method B: Fully Vectorized Pairwise Matrix (Zero Python Loops!)
If `X_test` has shape `(M, d)` and `X_train` has shape `(N, d)`, we can compute the entire `(M, N)` distance matrix simultaneously using NumPy broadcasting or the binomial expansion trick:

**Algebraic Identity:**
$$\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2 \mathbf{a}^T \mathbf{b}$$

In matrix form:
- $\mathbf{X}_{test}$ squared norms: vector of shape `(M, 1)`
- $\mathbf{X}_{train}$ squared norms: vector of shape `(1, N)`
- Dot products: $\mathbf{X}_{test} \mathbf{X}_{train}^T$ of shape `(M, N)`

```python
def compute_pairwise_distances(X_test, X_train):
    """
    Computes Euclidean distance between every row of X_test and X_train.
    
    Parameters
    ----------
    X_test : np.ndarray, shape (M, d)
    X_train : np.ndarray, shape (N, d)
    
    Returns
    -------
    dists : np.ndarray, shape (M, N)
        dists[i, j] is the distance between X_test[i] and X_train[j].
    """
    # Sum of squares along rows
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)  # (M, 1)
    train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T # (1, N)
    
    # 2 * X_test @ X_train.T has shape (M, N)
    dists_sq = test_sq + train_sq - 2 * (X_test @ X_train.T)
    
    # Due to small floating point errors, dists_sq might have tiny negative values (-1e-16)
    dists_sq = np.maximum(dists_sq, 0.0)
    return np.sqrt(dists_sq)
```

> [!TIP]
> Fully vectorized matrix multiplication is blazingly fast because NumPy calls highly optimized BLAS/LAPACK routines (like Intel MKL or OpenBLAS).

---

### 1.6 · Choosing $k$ & The Bias-Variance Tradeoff

| $k$ Value | Model Complexity | Decision Boundary | Risk |
|---|---|---|---|
| **$k = 1$** | Maximum complexity | Extremely jagged, wraps around individual noise points | **High Variance (Overfitting)** |
| **Small $k$ ($3, 5$)** | Balanced | Captures local nuances while smoothing random noise | Optimal balance |
| **Large $k$ ($k \approx N$)** | High simplicity | Overly smooth; predicts the global majority class everywhere | **High Bias (Underfitting)** |

```
k = 1 (Overfit):           k = 5 (Well-fit):          k = N (Underfit):
+-----------------+        +-----------------+        +-----------------+
| ●   /---\       |        | ● ●       /     |        |                 |
|    |  ■  |  ●   |        |   ●      /   ■  |        |  Predicts ●     |
|     \---/       |        |         /   ■ ■ |        |  everywhere!    |
|   ■        ■    |        |        /        |        |                 |
+-----------------+        +-----------------+        +-----------------+
```

> [!IMPORTANT]
> **The Tie-Breaking Rule:** In binary classification (2 classes), **always pick an odd $k$** ($k = 3, 5, 7, \dots$) to prevent 50/50 ties!
> If a tie does occur (e.g. with multiple classes or even $k$), a common tie-breaker is to pick the class whose nearest neighbors have the smaller average distance.

---

## Part 2: Building `KNNClassifier` in Pure NumPy

Let's write a complete, robust, Scikit-Learn style class:

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    """
    k-Nearest Neighbors classifier implemented from scratch in NumPy.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to query.
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan').
    """

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
        Store training data and labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X) >= self.k, f"Training set size ({len(X)}) must be >= k ({self.k})"

        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        return self

    def _compute_distances(self, X_test):
        """Vectorized distance matrix between X_test and self.X_train."""
        if self.metric == 'euclidean':
            # Pairwise squared expansion trick
            test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
            train_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
            sq_dists = test_sq + train_sq - 2 * (X_test @ self.X_train.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            return np.sqrt(sq_dists)
        else: # manhattan
            # |X_test[:, None, :] - X_train[None, :, :]sum over axis 2
            return np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (n_queries, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_queries,)
        """
        assert self.X_train is not None, "Classifier must be fit() before predicting!"
        X = np.asarray(X, dtype=float)

        # Distances shape: (n_queries, n_train)
        dists = self._compute_distances(X)

        # Get indices of the k smallest distances per query row
        # np.argpartition or np.argsort
        k_nearest_indices = np.argsort(dists, axis=1)[:, :self.k]

        # Retrieve neighbor labels: shape (n_queries, k)
        k_neighbor_labels = self.y_train[k_nearest_indices]

        # Majority vote per query
        y_pred = []
        for labels in k_neighbor_labels:
            # Counter finds frequency of each label
            vote_counts = Counter(labels)
            # most_common(1)[0][0] gives label with highest frequency
            winner = vote_counts.most_common(1)[0][0]
            y_pred.append(winner)

        return np.array(y_pred)

    def predict_proba(self, X):
        """
        Return probability estimates for each class for test samples in X.

        Returns
        -------
        probas : np.ndarray, shape (n_queries, n_classes)
        """
        assert self.X_train is not None, "Classifier must be fit() before predicting!"
        X = np.asarray(X, dtype=float)
        dists = self._compute_distances(X)
        k_nearest_indices = np.argsort(dists, axis=1)[:, :self.k]
        k_neighbor_labels = self.y_train[k_nearest_indices]

        probas = np.zeros((len(X), len(self.classes_)))
        for i, labels in enumerate(k_neighbor_labels):
            for j, cls in enumerate(self.classes_):
                probas[i, j] = np.mean(labels == cls)
        return probas
```

---

## Part 3: Evaluation Metrics from Scratch

Evaluating a model is just as crucial as building it. If your evaluation metric is misleading, you might deploy a model that fails silently.

```
                     Actual Positive      Actual Negative
Predicted Positive   True Positive (TP)   False Positive (FP)  <- Type I Error
Predicted Negative   False Negative (FN)  True Negative (TN)   <- Type II Error
                     ^ Type II Error
```

### 3.1 · Confusion Matrix

A confusion matrix tallies model predictions against the true ground truth.

```python
def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix to evaluate classification accuracy.

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes)
        Row index = true label, Column index = predicted label.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    return cm
```

---

### 3.2 · Classification Metrics

#### 1. Accuracy
The fraction of predictions the model got right:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\sum [y_{true} == y_{pred}]}{N}$$

```python
def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)
```

> [!WARNING]
> **The Accuracy Paradox:** In imbalanced datasets (e.g. 99% healthy patients, 1% cancer), a dummy model predicting "healthy" for every patient gets **99% accuracy**, but catches **0% of cancer cases**! Never rely solely on accuracy for imbalanced data.

#### 2. Precision
*"Of all instances predicted as Positive, how many are actually Positive?"*

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **High Precision Priority:** Minimizing False Positives (e.g., spam filter — you never want an important job offer marked as spam).

```python
def precision_score(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))
```

#### 3. Recall (Sensitivity)
*"Of all actual Positive instances, how many did the model detect?"*

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **High Recall Priority:** Minimizing False Negatives (e.g., cancer diagnosis or fraud detection — missing a positive case has disastrous consequences).

```python
def recall_score(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))
```

#### 4. F1-Score
The **harmonic mean** of Precision and Recall. It balances both metrics and punishes extreme imbalances between them:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

> **Why harmonic mean instead of arithmetic mean?**
> If Precision = $1.0$ and Recall = $0.0$:
> - Arithmetic Mean: $(1.0 + 0.0) / 2 = 0.50$ (misleadingly acceptable)
> - Harmonic Mean: $2 \cdot (1.0 \cdot 0.0) / (1.0 + 0.0) = \mathbf{0.0}$ (correctly penalizes total failure)

```python
def f1_score(y_true, y_pred, pos_label=1):
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    if prec + rec == 0.0:
        return 0.0
    return float(2 * (prec * rec) / (prec + rec))
```

---

### 3.3 · Regression Metrics

When predicting continuous numbers (like prices or temperatures):

#### 1. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```python
def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))
```

#### 2. Root Mean Squared Error (RMSE)
Standard deviation of prediction residuals; expressed in the **exact same units** as the target variable:

$$\text{RMSE} = \sqrt{\text{MSE}}$$

```python
def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
```

#### 3. Mean Absolute Error (MAE)
Average absolute magnitude of errors; less sensitive to extreme outliers than MSE:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

```python
def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
```

#### 4. $R^2$ Score (Coefficient of Determination)
Quantifies the proportion of target variance explained by the model:

$$R^2 = 1 - \frac{\text{Residual Sum of Squares (RSS)}}{\text{Total Sum of Squares (TSS)}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where $\bar{y} = \frac{1}{n}\sum y_i$ is the mean of true values.

```python
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0  # avoid division by zero if target is constant
    return float(1.0 - (ss_res / ss_tot))
```

#### Interpreting $R^2$:
- $R^2 = 1.0$: Perfect predictions. Zero error.
- $R^2 = 0.0$: The model performs exactly as well as a baseline that constantly guesses $\bar{y}$ (the mean).
- $R^2 < 0.0$: The model is **worse** than simply predicting the mean!

---

## Part 4: Verification Against Scikit-Learn

To ensure our from-scratch implementations are 100% bug-free, we test them against `scikit-learn` using random and synthetic datasets:

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score as sk_acc,
    precision_score as sk_prec,
    recall_score as sk_rec,
    f1_score as sk_f1,
    mean_squared_error as sk_mse,
    r2_score as sk_r2
)

# 1. Test Classification Metrics
y_true_clf = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred_clf = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])

assert np.isclose(accuracy_score(y_true_clf, y_pred_clf), sk_acc(y_true_clf, y_pred_clf))
assert np.isclose(precision_score(y_true_clf, y_pred_clf), sk_prec(y_true_clf, y_pred_clf))
assert np.isclose(recall_score(y_true_clf, y_pred_clf), sk_rec(y_true_clf, y_pred_clf))
assert np.isclose(f1_score(y_true_clf, y_pred_clf), sk_f1(y_true_clf, y_pred_clf))
print("✅ All Classification Metrics match sklearn perfectly!")

# 2. Test Regression Metrics
y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 3.9])

assert np.isclose(mean_squared_error(y_true_reg, y_pred_reg), sk_mse(y_true_reg, y_pred_reg))
assert np.isclose(r2_score(y_true_reg, y_pred_reg), sk_r2(y_true_reg, y_pred_reg))
print("✅ All Regression Metrics match sklearn perfectly!")

# 3. Test k-NN Classifier
np.random.seed(42)
X_train_dummy = np.random.randn(80, 4)
y_train_dummy = np.random.choice([0, 1], size=80)
X_test_dummy = np.random.randn(20, 4)

my_knn = KNNClassifier(k=5).fit(X_train_dummy, y_train_dummy)
sk_knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute').fit(X_train_dummy, y_train_dummy)

my_preds = my_knn.predict(X_test_dummy)
sk_preds = sk_knn.predict(X_test_dummy)

assert np.array_equal(my_preds, sk_preds), "k-NN predictions differ from sklearn!"
print("✅ Custom k-NN matches Scikit-Learn predictions 100%!")
```

---

## Part 5: The 60-Minute Blank-File Test

### 5.1 · What is the Blank-File Test?

The **Blank-File Test** is the gold standard used by senior ML practitioners to verify genuine conceptual internalization.

- **The Rules:**
  1. Open a fresh, completely blank file: `blank_test.py`.
  2. Set a timer for **60 minutes**.
  3. Close all ChatGPT/LLM tabs, documentation, and previous notes.
  4. Write the solutions from memory using only standard Python and NumPy.
  5. Run the verification assertions at the bottom.

If you can pass this test, you have permanently internalized the mathematical and engineering foundations of Week 1!

---

### 5.2 · The 3 Selected Problems (Days 1–5 Review)

#### 📝 Problem 1 (Days 1 & 3): Pure Python Statistics & Frequency
**Requirement:** Without importing `numpy` or `pandas`, implement:
1. `mean(nums)`: returns the arithmetic mean.
2. `variance(nums, sample=True)`: returns sample variance (divide by $n-1$) or population variance ($n$).
3. `std_dev(nums, sample=True)`: returns standard deviation.
4. `top_k_words(text, k=3)`: takes a raw text string, normalizes case, strips punctuation, and returns the top $k$ most frequent words as a list of `(word, count)` tuples.

#### 📝 Problem 2 (Days 4 & 5): NumPy Vectorization & StandardScaler
**Requirement:**
1. Given a 2D array $X$ of shape $(N, d)$, write `StandardScaler` from scratch:
   - `fit(X)`: stores feature-wise mean and std (handle division by zero if std == 0).
   - `transform(X)`: returns $Z = (X - \mu) / \sigma$.
   - `fit_transform(X)`: chains both.
2. Implement `pairwise_euclidean_matrix(A, B)` without any Python loops where $A$ is $(M, d)$ and $B$ is $(N, d)$, returning distance matrix of shape $(M, N)$.

#### 📝 Problem 3 (Day 5): Train-Test Split & Linear Regression (Normal Equation)
**Requirement:**
1. `train_test_split(X, y, test_size=0.2, random_state=None)`:
   - Shuffles indices and returns `X_train, X_test, y_train, y_test`.
2. `LinearRegressionNormal`:
   - `fit(X, y)`: computes weights including intercept using the closed-form normal equation:
     $$\mathbf{w} = (X_b^T X_b)^{-1} X_b^T y$$
   - `predict(X)`: returns $\hat{y} = X_b \mathbf{w}$.

---

### 5.3 · Self-Scoring Rubric (60 Points Total)

| Problem | Time Target | Points | Criteria |
|---|:-:|:-:|---|
| **Problem 1: Pure Python Stats** | 12 min | 15 pts | Correct formula, handles division by $n-1$, cleans punctuation |
| **Problem 2: Vectorization & Scaler** | 18 min | 20 pts | `fit/transform` paradigm clean, zero Python loops for pairwise distances |
| **Problem 3: Split & Normal Equation** | 20 min | 20 pts | Proper index permutation, bias column correctly prepended/appended, matrix operations `@` and `np.linalg.pinv` |
| **Code Cleanliness & Docstrings** | 10 min | 5 pts | Clear naming, no bugs, tests pass |

**Score Guide:**
- **50–60 pts:** 🎉 Mastery! Ready for advanced optimization and neural architectures.
- **40–49 pts:** Solid foundation; review vectorized matrix operations.
- **< 40 pts:** Retake the test tomorrow after reviewing Day 4 and Day 5 guides.

---

## Part 6: Clean Notebook & Git Push (`from_scratch.ipynb`)

To showcase your work in your portfolio or repository:

### 6.1 · Notebook Checklist
- [ ] **Title & Intro:** Clear H1 heading with an overview of all algorithms implemented from scratch.
- [ ] **Organized Sections:**
  - Section 1: Preprocessing & Splitting (`train_test_split`, `StandardScaler`)
  - Section 2: Linear Models (`LinearRegressionNormal`, `LinearRegressionGD`)
  - Section 3: Instance Models (`KNNClassifier`)
  - Section 4: Metrics Suite (`accuracy`, `precision`, `recall`, `f1`, `mse`, `r2`)
  - Section 5: Benchmark & Validation against Scikit-Learn
- [ ] **Visualizations:** Clean matplotlib plots:
  - 1D Linear regression fit line with actual scatter points
  - k-NN Decision Boundary plot on 2D synthetic features
  - Gradient descent loss curve ($J$ vs iterations)
- [ ] **Verification Assertions:** Every custom function has an `assert np.allclose(...)` comparing it to `sklearn`.
- [ ] **Clean Output:** Clear excessive verbose output; run "Restart Kernel and Run All Cells" to ensure no execution errors.

### 6.2 · Git Push Workflow
```bash
# Check status
git status

# Add day 6 work
git add W1/D6/

# Commit with a clean conventional commit message
git commit -m "feat(W1D6): complete k-NN, custom metrics suite, blank-file test, and from_scratch notebook"

# Push to your remote repository
git push origin main
```

---

## 🗺️ Quick Reference Cheat Sheet

| Topic | Equation / Key Syntax | Notes |
|---|---|---|
| **Euclidean Distance** | $\sqrt{\sum (p_i - q_i)^2}$ | Standard distance; sensitive to scaling |
| **Manhattan Distance** | $\sum \|p_i - q_i\|$ | Grid distance; more robust in high dimensions |
| **Pairwise Distances** | $\sqrt{\mathbf{a}^2 + \mathbf{b}^2 - 2\mathbf{a}^T\mathbf{b}}$ | Vectorized distance matrix without loops |
| **k-NN Complexity** | Train: $O(1)$, Test: $O(N \cdot d)$ | Lazy learner; slow inference on big datasets |
| **Tie-Breaking** | Use odd $k$ for binary classes | Prevents split votes |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Misleading on imbalanced datasets |
| **Precision** | $\frac{TP}{TP + FP}$ | Focus on minimizing False Positives (Spam filter) |
| **Recall** | $\frac{TP}{TP + FN}$ | Focus on minimizing False Negatives (Medical screening) |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean; balances Precision and Recall |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| **$R^2$ Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | $1.0$ = perfect, $0$ = baseline mean predictor, $<0$ = worse than mean |
