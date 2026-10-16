import numpy as np 

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Splits dataset into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle indices
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


# Step 3 Helper: Normal Equation
def fit_normal_equation(X, y):
    """Calculates weights w = (X_b^T @ X_b)^(-1) @ X_b^T @ y"""
    n = X.shape[0]
    # Add column of ones for bias
    X_b = np.column_stack([X, np.ones(n)])

    # Normal Equation using pseudo-inverse
    w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return w



np.random.seed(42)
X=2 * np.random.rand(100,1)
y= 4+3 *X.ravel()+np.random.randn(100)*0.5


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


w=fit_normal_equation(x_train,y_train)
weight_x1 = w[0]
bias = w[-1]

print("--- Final Parameters ---")
print(f"Learned Weight Vector w: {w}")
print(f"Discovered Slope (w1):   {weight_x1:.4f} (True: 3.0)")
print(f"Discovered Intercept (b): {bias:.4f} (True: 4.0)")

def fit_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    """Fits Linear Regression weights using Gradient Descent with bias concatenated."""
    m = X.shape[0] 

    # 1. Add bias column (column of 1s)
    X_b = np.column_stack([X, np.ones(m)])

    # 2. Initialize weights to zeros: shape (n_features + 1,)
    w = np.zeros(X_b.shape[1])

    # 3. Iterative gradient updates
    for iteration in range(n_iterations):
        # Gradient formula: (1/m) * X_b^T @ (X_b @ w - y)
        gradients = (1 / m) * X_b.T @ (X_b @ w - y)
        w = w - learning_rate * gradients

    return w
# ---------------------------------------------------------
# Step 4: Fit with Gradient Descent (learning_rate=0.1, n_iterations=1000)
# ---------------------------------------------------------
w_gd = fit_gradient_descent(
    x_train, y_train, learning_rate=0.1, n_iterations=1000
)

weight_x1_gd = w_gd[0]
bias_gd = w_gd[-1]

print("--- Gradient Descent Parameters ---")
print(f"Learned Weight Vector w: {w_gd}")
print(f"Discovered Slope (w1):   {weight_x1_gd:.4f} (True: 3.0)")
print(f"Discovered Intercept (b): {bias_gd:.4f} (True: 4.0)")


def predict(X, w):
    """Predicts y values using feature matrix X and weight vector w (which includes bias)."""
    m = X.shape[0]
    X_b = np.column_stack([X, np.ones(m)])
    return X_b @ w

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)

def r2_score(y_true, y_pred):
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


y_pred_norm = predict(x_test, w)
y_pred_gd = predict(x_test, w_gd)

mse_norm = mse(y_test, y_pred_norm)
r2_norm = r2_score(y_test, y_pred_norm)

mse_gd = mse(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

# ---------------------------------------------------------
# Step 7: Print Comparison Table
# ---------------------------------------------------------
print("=" * 68)
print(f"{'Method':<20} | {'Weight (w1)':<12} | {'Bias (b)':<10} | {'MSE':<8} | {'R²':<8}")
print("=" * 68)
print(
    f"{'True Parameters':<20} | {3.0000:<12.4f} | {4.0000:<10.4f} | {'N/A':<8} | {'N/A':<8}"
)
print(
    f"{'Normal Equation':<20} | {w[0]:<12.4f} | {w[1]:<10.4f} | {mse_norm:<8.4f} | {r2_norm:<8.4f}"
)
print(
    f"{'Gradient Descent':<20} | {w_gd[0]:<12.4f} | {w_gd[1]:<10.4f} | {mse_gd:<8.4f} | {r2_gd:<8.4f}"
)
print("=" * 68)

# Check Step 5 Condition
print(f"\nWeights match closely: {np.allclose(w, w_gd, atol=1e-4)}")
