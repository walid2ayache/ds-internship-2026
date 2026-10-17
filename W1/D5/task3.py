import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Learn mean and std from data."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero if a feature has zero variance
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

    def transform(self, X):
        """Apply the transformation."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("You must call fit before transform.")
        return (X - self.mean_) / self.std_


# Example Usage
# Create a simple dataset
X = np.array([[1000, 1], [2000, 2], [3000, 3], [4000, 4], [5000, 5]])

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)

print("Original Data:")
print(X)
print("\nScaled Data:")
print(X_scaled)
print("\nMean of scaled data (should be ~0):")
print(np.mean(X_scaled, axis=0))
print("\nStd of scaled data (should be 1):")
print(np.std(X_scaled, axis=0))


def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


print(mse(np.array([1,2,3]),np.array([1,2,3])))


def fit_normal_equation(X,y):
    n= X.shape[0]
    X_b=np.column_stack([X,np.ones(n)])
    w=np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return w 

def predict(X, w):
    """
    Predict target values using learned weight vector w.
    """
    n = X.shape[0]
    X_b = np.column_stack([X, np.ones(n)])
    return X_b @ w







np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
true_weights = np.array([3.0, 5.0])
true_bias = 10.0

y = X @ true_weights + true_bias + np.random.normal(0, 0.1, size=100)
w = fit_normal_equation(X, y)

print("Learned Feature Weights:", w[:-1])  # Target: ~[3.0, 5.0]
print("Learned Bias (Intercept):", w[-1])  # Target: ~10.0

# 2. Make predictions on new test data
X_new = np.array([[0.5, 0.2], [0.8, 0.9]])
predictions = predict(X_new, w)
print("Predictions for X_new:", predictions)










from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Create dummy data
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_train = np.array([2, 4, 5, 4, 5])
X_test = np.array([[11, 12], [13, 14]])

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)





