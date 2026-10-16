# 🗓️ Day 5 — ML from Scratch (Part 1) — Full Learning Guide

---

## Part 1: `train_test_split` — Build It Yourself

### 1.1 · Why Split Data?

When building a machine learning model, you need to **evaluate** it on data it has **never seen**. If you train and test on the same data, the model can just memorize the answers — that's **overfitting**, not learning.

```
Your Dataset
├── Training Set (80%)  → model learns from this
└── Test Set (20%)      → model is evaluated on this (never touched during training)
```

**The golden rule:** The test set is a **sealed envelope**. You only open it at the very end.

---

### 1.2 · How `train_test_split` Works (The Algorithm)

```
1. Get the total number of samples (n)
2. Compute how many go to test: n_test = int(n * test_size)
3. Create an array of indices [0, 1, 2, ..., n-1]
4. Shuffle the indices randomly
5. Split: first n_test indices → test, rest → train
6. Use the indices to select rows from X and y
```

> [!IMPORTANT]
> Both `X` (features) and `y` (labels) must be shuffled **together** — the same permutation applied to both. Otherwise row 5 of X might end up paired with row 12 of y!

---

### 1.3 · Implementation

```python
import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into train and test subsets.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.
    test_size : float, default=0.2
        Fraction of data to use as test set (0.0 to 1.0).
    random_state : int or None, default=None
        Seed for reproducibility. Same seed → same split every time.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    n = len(X)
    assert len(X) == len(y), "X and y must have the same number of rows"
    assert 0.0 < test_size < 1.0, "test_size must be between 0 and 1"

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_state)

    # Shuffle indices (not the data itself — cleaner)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Split point
    n_test = int(n * test_size)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

**Usage:**
```python
# Example data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
              [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)} samples")   # Train: 8 samples
print(f"Test:  {len(X_test)} samples")    # Test:  2 samples
```

> [!TIP]
> **`np.random.default_rng(seed)`** is the modern way to create a random number generator in NumPy. It's preferred over the old `np.random.seed()` because each `rng` instance is independent — you won't accidentally affect other random operations elsewhere in your code.

---

### 1.4 · Why `random_state` Matters

```python
# Without random_state — different split every time!
X_train, X_test, _, _ = train_test_split(X, y)
X_train2, X_test2, _, _ = train_test_split(X, y)
# X_train ≠ X_train2 (probably)

# With random_state — reproducible!
X_train, X_test, _, _ = train_test_split(X, y, random_state=42)
X_train2, X_test2, _, _ = train_test_split(X, y, random_state=42)
# X_train == X_train2  ✅ (guaranteed)
```

Always use `random_state` when debugging or comparing models — otherwise your results change every run and you can't tell what actually improved.

---

### ✏️ Exercise 1 — `train_test_split`

Create `task1.py`:

1. Implement `train_test_split` from scratch (no sklearn!)
2. Create a dummy dataset: `X = np.random.randn(100, 3)` and `y = np.random.randint(0, 2, 100)`
3. Split with `test_size=0.25` and `random_state=42`
4. Print the shapes of all 4 outputs
5. Verify: call it twice with the same `random_state` — do you get identical splits?
6. Verify: `len(X_train) + len(X_test) == len(X)`

---

## Part 2: `StandardScaler` — Build It Yourself

### 2.1 · Why Scale Features?

Imagine predicting house prices with two features:
- **Area:** 500 – 5000 sq ft
- **Bedrooms:** 1 – 6

Area values are ~1000× larger than bedroom values. Without scaling, algorithms that use distances or gradients (like gradient descent) will be **dominated by area** and practically ignore bedrooms.

**StandardScaler** transforms each feature to have **mean = 0** and **standard deviation = 1**:

$$z = \frac{x - \mu}{\sigma}$$

```
Before scaling:          After scaling:
Area:     [500, ..., 5000]    → [-1.7, ..., 1.7]
Bedrooms: [1, ..., 6]         → [-1.5, ..., 1.5]
```

Now both features are on the same scale and contribute equally.

---

### 2.2 · The `fit` / `transform` Pattern

This is a **fundamental pattern** in ML. Every preprocessing step follows it:

```
scaler.fit(X_train)         # Learn the parameters (mean, std) from training data
X_train_scaled = scaler.transform(X_train)   # Apply to training data
X_test_scaled  = scaler.transform(X_test)    # Apply SAME parameters to test data
```

> [!CAUTION]
> **Never fit on test data!** The scaler must learn mean/std from **training data only**. If you fit on the full dataset (including test), you're leaking future information into your model — this is called **data leakage** and gives you falsely optimistic results.
>
> ```python
> # ❌ WRONG — data leakage!
> scaler.fit(X_all)
> X_train_scaled = scaler.transform(X_train)
> X_test_scaled = scaler.transform(X_test)
>
> # ✅ CORRECT
> scaler.fit(X_train)          # learn from train ONLY
> X_train_scaled = scaler.transform(X_train)
> X_test_scaled = scaler.transform(X_test)   # uses train's mean/std
> ```

---

### 2.3 · Implementation

```python
import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    z = (x - mean) / std

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
        Per-feature mean, computed during fit().
    std_ : np.ndarray, shape (n_features,)
        Per-feature standard deviation, computed during fit().
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Compute the mean and std of each feature (column) from the training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self : StandardScaler
            Returns self so you can chain: scaler.fit(X).transform(X)
        """
        self.mean_ = np.mean(X, axis=0)    # mean of each column
        self.std_ = np.std(X, axis=0)      # std of each column
        return self

    def transform(self, X):
        """
        Apply the standardization using the stored mean and std.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_scaled : np.ndarray, same shape as X
        """
        assert self.mean_ is not None, "Must call fit() before transform()"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Convenience: fit and transform in one step."""
        return self.fit(X).transform(X)
```

**Usage:**
```python
X_train = np.array([[500, 3], [1500, 4], [3000, 2], [4500, 5]])
X_test = np.array([[2000, 3], [1000, 6]])

scaler = StandardScaler()
scaler.fit(X_train)                     # learn from train

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # same params!

print(X_train_scaled.mean(axis=0))   # ≈ [0, 0]  (close to zero)
print(X_train_scaled.std(axis=0))    # ≈ [1, 1]  (close to one)
```

> [!NOTE]
> **Why return `self` from `fit()`?** It enables **method chaining**: `scaler.fit(X).transform(X)` — a common pattern you'll see in sklearn. It's just a convenience, not magic.

---

### 2.4 · Edge Case: Zero Variance

What if a feature is constant (e.g., all values are 5)? Its std = 0, and you'd divide by zero!

```python
# Robust version: replace 0 std with 1 (no scaling for constant features)
def transform(self, X):
    std = np.where(self.std_ == 0, 1, self.std_)   # avoid division by zero
    return (X - self.mean_) / std
```

---

### ✏️ Exercise 2 — `StandardScaler`

Add to `task1.py` or create `task2.py`:

1. Implement `StandardScaler` from scratch
2. Create training data: `X_train = np.array([[1, 200], [2, 400], [3, 600], [4, 800], [5, 1000]])`
3. Create test data: `X_test = np.array([[3, 500], [6, 1200]])`
4. Fit on `X_train`, transform both `X_train` and `X_test`
5. Print the scaler's `mean_` and `std_`
6. Verify: `X_train_scaled` should have mean ≈ 0 and std ≈ 1 per column
7. Verify: `X_test_scaled` should **NOT** have mean 0 (it uses train's parameters!)

---

## Part 3: Linear Regression — The Theory

### 3.1 · What is Linear Regression?

Linear regression finds the **best-fitting line** (or hyperplane) through your data.

For one feature:
```
ŷ = w₁x + w₀
    │       │
    │       └─ bias (intercept) — where the line crosses the y-axis
    └───────── weight (slope) — how much y changes per unit of x
```

For multiple features:
```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + w₀
```

In matrix form (much cleaner):
```
ŷ = Xw
```
where `X` has a column of 1s prepended (for the bias term) and `w` is the weight vector.

---

### 3.2 · The Cost Function — Mean Squared Error (MSE)

"Best-fitting" means: minimize the **average squared error** between predictions and actual values.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

In NumPy:
```python
def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)
```

**Why squared?**
- Makes all errors positive (no cancellation)
- Penalizes large errors more than small ones
- Is differentiable (needed for gradient descent)

---

### 3.3 · Method 1: Normal Equation (Closed-Form Solution)

The normal equation gives you the **exact** optimal weights in one step:

$$\mathbf{w} = (X^T X)^{-1} X^T y$$

```python
def fit_normal_equation(X, y):
    """
    Solve for weights using the Normal Equation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (WITHOUT bias column — we add it).
    y : np.ndarray, shape (n_samples,)
        Target values.

    Returns
    -------
    w : np.ndarray, shape (n_features + 1,)
        Weight vector. Last element is the bias (intercept).
    """
    # Add bias column (column of 1s)
    n = X.shape[0]
    X_b = np.column_stack([X, np.ones(n)])   # shape: (n, features+1)

    # Normal equation: w = (X^T X)^(-1) X^T y
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    return w
```

**Pros and cons:**
| | Normal Equation |
|---|---|
| ✅ | Exact solution, no hyperparameters |
| ✅ | One-step computation |
| ❌ | Requires matrix inversion — O(n³) for n features |
| ❌ | Breaks if X^T X is singular (non-invertible) |
| ❌ | Slow/impossible for very large feature counts (>10k features) |

> [!TIP]
> In practice, use `np.linalg.lstsq(X_b, y)` instead of manually computing the inverse — it's numerically more stable and handles edge cases:
> ```python
> w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
> ```

---

### 3.4 · Method 2: Gradient Descent (Iterative Optimization)

Gradient descent is the **workhorse of ML**. Instead of solving in one shot, it takes many small steps toward the minimum.

**The idea:**
1. Start with random weights
2. Compute how wrong we are (the gradient of the cost function)
3. Nudge the weights in the opposite direction of the gradient
4. Repeat until convergence

**The math:**
```
Gradient of MSE with respect to w:
    ∇MSE = (2/n) · Xᵀ · (Xw - y)

Update rule:
    w = w - α · ∇MSE
```
where `α` (alpha) is the **learning rate** — how big each step is.

---

### 3.5 · Gradient Descent — Visual Intuition

Think of the cost function as a **bowl**. You're standing somewhere on the rim, and you want to reach the bottom (minimum error).

```
Cost (MSE)
  │\
  │ \        ← You are here. Gradient points uphill.
  │  \          So you step in the OPPOSITE direction (downhill).
  │   \
  │    \
  │     \_____/   ← The minimum — your goal
  │
  └──────────── w (weight)
```

- **Learning rate too small:** Tiny steps → takes forever to converge
- **Learning rate too big:** Overshoots the minimum → diverges (cost goes to infinity!)
- **Just right:** Converges smoothly in reasonable iterations

```
α too small:    α just right:    α too big:
. . . . ._     .               .     .
                .                 .   .
                  .                 . .
                    .____           BOOM! (diverges)
```

---

### 3.6 · Gradient Descent Implementation

```python
def fit_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=False):
    """
    Solve for weights using Gradient Descent.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (WITHOUT bias column — we add it).
    y : np.ndarray, shape (n_samples,)
        Target values.
    learning_rate : float, default=0.01
        Step size (α). Too small = slow. Too big = diverges.
    n_iterations : int, default=1000
        Number of gradient descent steps.
    verbose : bool, default=False
        If True, print cost every 100 iterations.

    Returns
    -------
    w : np.ndarray, shape (n_features + 1,)
        Weight vector. Last element is the bias.
    history : list of float
        MSE at each iteration (useful for plotting convergence).
    """
    n = X.shape[0]
    X_b = np.column_stack([X, np.ones(n)])   # add bias column
    n_features = X_b.shape[1]

    # Initialize weights to small random values
    rng = np.random.default_rng(42)
    w = rng.standard_normal(n_features) * 0.01

    history = []

    for i in range(n_iterations):
        # 1. Predictions
        y_pred = X_b @ w                       # shape: (n,)

        # 2. Compute error
        error = y_pred - y                      # shape: (n,)

        # 3. Compute cost (MSE) — for tracking, not for the update
        cost = np.mean(error ** 2)
        history.append(cost)

        # 4. Compute gradient: (2/n) * X^T * error
        gradient = (2 / n) * (X_b.T @ error)    # shape: (n_features,)

        # 5. Update weights
        w = w - learning_rate * gradient

        if verbose and i % 100 == 0:
            print(f"Iteration {i:4d} | MSE: {cost:.6f}")

    return w, history
```

> [!WARNING]
> **Feature scaling is critical for gradient descent!** If features have very different scales, the cost surface becomes an elongated ellipse and gradient descent zigzags slowly. Always use `StandardScaler` before gradient descent.
>
> The normal equation doesn't need scaling (it solves directly), but gradient descent does.

---

### 3.7 · Making Predictions

```python
def predict(X, w):
    """
    Predict using learned weights.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (WITHOUT bias column — we add it).
    w : np.ndarray, shape (n_features + 1,)
        Weight vector (last element is bias).

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples,)
    """
    n = X.shape[0]
    X_b = np.column_stack([X, np.ones(n)])
    return X_b @ w
```

---

### 3.8 · Evaluation Metrics

```python
def mse(y_true, y_pred):
    """Mean Squared Error — average of squared differences."""
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_true, y_pred):
    """Root Mean Squared Error — in same units as y."""
    return np.sqrt(mse(y_true, y_pred))

def r_squared(y_true, y_pred):
    """
    R² (coefficient of determination).
    = 1 means perfect predictions.
    = 0 means no better than predicting the mean.
    < 0 means worse than predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)     # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    return 1 - (ss_res / ss_tot)
```

**Interpreting R²:**
| R² | Meaning |
|----|---------|
| 1.0 | Perfect fit |
| 0.9+ | Excellent |
| 0.7–0.9 | Good |
| 0.5–0.7 | Moderate |
| < 0.5 | Poor |
| < 0 | Worse than just predicting the mean |

---

### ✏️ Exercise 3 — Linear Regression from Scratch

Create `task2.py`:

1. **Generate synthetic data:**
   ```python
   np.random.seed(42)
   X = 2 * np.random.rand(100, 1)          # 100 samples, 1 feature
   y = 4 + 3 * X.ravel() + np.random.randn(100) * 0.5   # y = 4 + 3x + noise
   ```
   (The true relationship is y = 4 + 3x. Your model should discover w ≈ [3, 4].)

2. Split into train/test (80/20) using your `train_test_split`

3. **Fit with normal equation** — print the weights

4. **Fit with gradient descent** (try `learning_rate=0.1`, `n_iterations=1000`) — print the weights

5. Compare: the weights from both methods should be **nearly identical**

6. Compute MSE and R² on the test set for both methods

7. Print a comparison table

---

## Part 4: Compare with `sklearn`

### 4.1 · sklearn's `LinearRegression`

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Split
X_train, X_test, y_train, y_test = sklearn_split(X, y, test_size=0.2, random_state=42)

# 2. Scale (optional for LinearRegression, but good practice)
scaler = SklearnScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Fit
model = LinearRegression()
model.fit(X_train_s, y_train)

# 4. Predict
y_pred = model.predict(X_test_s)

# 5. Evaluate
print(f"Coefficients: {model.coef_}")
print(f"Intercept:    {model.intercept_}")
print(f"MSE:          {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:           {r2_score(y_test, y_pred):.4f}")
```

> [!NOTE]
> `sklearn.linear_model.LinearRegression` uses a variant of the normal equation internally (specifically, it uses `scipy.linalg.lstsq` — a numerically stable least squares solver). It's not using gradient descent.

---

### 4.2 · What to Compare

| What | Your Implementation | sklearn |
|------|-------------------|---------|
| Weights (slope) | `w[:-1]` | `model.coef_` |
| Bias (intercept) | `w[-1]` | `model.intercept_` |
| MSE on test | `mse(y_test, y_pred)` | `mean_squared_error(y_test, y_pred)` |
| R² on test | `r_squared(y_test, y_pred)` | `r2_score(y_test, y_pred)` |

They should match closely (small differences due to numerical precision are normal).

---

### 4.3 · Full Comparison Template

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Your implementations
# from your_module import train_test_split, StandardScaler, fit_normal_equation, ...

# ── Generate Data ──
np.random.seed(42)
X = 2 * np.random.rand(200, 1)
y = 4 + 3 * X.ravel() + np.random.randn(200) * 0.5

# ── Split (your function) ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Method 1: Normal Equation ──
w_ne = fit_normal_equation(X_train, y_train)
y_pred_ne = predict(X_test, w_ne)

# ── Method 2: Gradient Descent ──
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

w_gd, history = fit_gradient_descent(X_train_s, y_train, learning_rate=0.1, n_iterations=1000)
y_pred_gd = predict(X_test_s, w_gd)

# ── Method 3: sklearn ──
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
y_pred_sk = sk_model.predict(X_test)

# ── Compare ──
print(f"{'Method':<20} {'MSE':>8} {'R²':>8} {'Slope':>8} {'Intercept':>10}")
print("-" * 60)
print(f"{'Normal Equation':<20} {mse(y_test, y_pred_ne):>8.4f} {r_squared(y_test, y_pred_ne):>8.4f} {w_ne[0]:>8.4f} {w_ne[-1]:>10.4f}")
print(f"{'Gradient Descent':<20} {mse(y_test, y_pred_gd):>8.4f} {r_squared(y_test, y_pred_gd):>8.4f} {'(scaled)':>8} {w_gd[-1]:>10.4f}")
print(f"{'sklearn':<20} {mean_squared_error(y_test, y_pred_sk):>8.4f} {r2_score(y_test, y_pred_sk):>8.4f} {sk_model.coef_[0]:>8.4f} {sk_model.intercept_:>10.4f}")
```

---

### ✏️ Exercise 4 — Full Pipeline Comparison

Create `task3.py`:

1. Generate a dataset with **2 features**:
   ```python
   np.random.seed(42)
   X = np.random.randn(300, 2)
   # True relationship: y = 5 + 2*x1 - 3*x2 + noise
   y = 5 + 2 * X[:, 0] - 3 * X[:, 1] + np.random.randn(300) * 0.5
   ```

2. Split 80/20 with your `train_test_split`

3. Fit using:
   - Your normal equation
   - Your gradient descent (with `StandardScaler` + `learning_rate=0.1`)
   - sklearn's `LinearRegression`

4. Print a comparison table showing weights, MSE, and R² for each method

5. All three should find weights close to **[2, -3]** with intercept close to **5**

6. **Bonus:** Print the gradient descent convergence — does MSE decrease smoothly?

---

## Part 5: Understanding the Math Intuitively

### 5.1 · Gradient Descent Step-by-Step Example

Let's trace through one iteration with tiny numbers:

```python
# Setup: 3 samples, 1 feature, bias added → X_b is (3, 2)
X_b = np.array([[1, 1],     # sample 1: x=1, bias=1
                [2, 1],     # sample 2: x=2, bias=1
                [3, 1]])    # sample 3: x=3, bias=1
y = np.array([3, 5, 7])     # true values
w = np.array([0.5, 0.5])    # initial weights: [slope, intercept]
lr = 0.1                     # learning rate

# Step 1: Predict
y_pred = X_b @ w             # [1.0, 1.5, 2.0]

# Step 2: Error
error = y_pred - y            # [-2.0, -3.5, -5.0]

# Step 3: Gradient
gradient = (2/3) * X_b.T @ error
# X_b.T @ error = [1*(-2) + 2*(-3.5) + 3*(-5), 1*(-2) + 1*(-3.5) + 1*(-5)]
#               = [-24.0, -10.5]
# gradient = (2/3) * [-24.0, -10.5] = [-16.0, -7.0]

# Step 4: Update
w = w - lr * gradient
# w = [0.5, 0.5] - 0.1 * [-16.0, -7.0]
# w = [0.5 + 1.6, 0.5 + 0.7]
# w = [2.1, 1.2]    ← moved toward the true values (slope≈2, intercept≈1)
```

One step already moved us significantly closer! After many iterations, we converge to the optimal weights.

---

### 5.2 · Effect of Learning Rate

```python
# Experiment: try different learning rates on the same data
for lr in [0.001, 0.01, 0.1, 0.5]:
    w, history = fit_gradient_descent(X_train, y_train, learning_rate=lr, n_iterations=500)
    print(f"lr={lr:.3f} → Final MSE: {history[-1]:.4f}")
```

Expected pattern:
```
lr=0.001 → Final MSE: 5.2341   (too slow, hasn't converged yet)
lr=0.010 → Final MSE: 0.2856   (getting there)
lr=0.100 → Final MSE: 0.2501   (converged nicely ✅)
lr=0.500 → Final MSE: 0.2501   (also fine, but risky with more features)
lr=1.000 → Final MSE: NaN      (exploded! 💥)
```

> [!WARNING]
> If your MSE becomes `NaN` or `inf`, your learning rate is too high. Reduce it by 10×.

---

### 5.3 · Normal Equation vs Gradient Descent — When to Use Which

| Criterion | Normal Equation | Gradient Descent |
|-----------|:-:|:-:|
| **# features < 10k** | ✅ Fast | ✅ Fast |
| **# features > 10k** | ❌ Very slow (matrix inversion) | ✅ Still works |
| **Exact solution** | ✅ Yes | ❌ Approximate (depends on lr, iters) |
| **Hyperparameters** | None | `learning_rate`, `n_iterations` |
| **Feature scaling needed** | No | **Yes** |
| **Works for other models** | No (only linear) | ✅ Yes (neural nets, logistic reg, etc.) |

**Bottom line:** Normal equation is great for small problems. Gradient descent is the **general-purpose** optimizer that scales to any model.

---

## 🗺️ Quick Reference Cheat Sheet

| Topic | Key Takeaway |
|-------|-------------|
| `train_test_split` | Shuffle indices, split into train/test. Always set `random_state` |
| Test set rule | **Never** train on test data. Never fit scalers on test data |
| `StandardScaler` | `z = (x - μ) / σ`. Fit on train, transform both train and test |
| `fit` / `transform` | `fit` = learn parameters. `transform` = apply them. Never `fit` on test |
| Data leakage | Fitting scaler/model on test data → falsely optimistic results |
| Linear regression | `ŷ = Xw`. Find `w` that minimizes MSE |
| Normal equation | `w = (XᵀX)⁻¹Xᵀy`. Exact, no hyperparameters, slow for many features |
| Gradient descent | `w = w - α·∇MSE`. Iterative, needs learning rate & scaling |
| Learning rate | Too small → slow. Too big → diverges (NaN). Start with 0.01–0.1 |
| MSE | `mean((ŷ - y)²)` — average squared error |
| R² | 1 = perfect, 0 = as good as mean, < 0 = worse than mean |
| `np.column_stack` | Add a bias column of 1s: `np.column_stack([X, np.ones(n)])` |
| `np.linalg.inv` | Matrix inverse (for normal equation) |
| `@` operator | Matrix multiplication: `X @ w` |
| sklearn comparison | Your results should match `LinearRegression` closely |
