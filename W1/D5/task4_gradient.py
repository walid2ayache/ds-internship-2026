import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
x_train = np.array([1.0, 2.0])       # features
y_train = np.array([300.0, 500.0])   # target values


def compute_cost(x, y, w, b):
    m = x.shape[0] 
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost
    return total_cost


def compute_gradient(x, y, w, b): 
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)     

        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        if i < 100000: 
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history


# Initialize parameters
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient
)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


# Vectorized prediction function (works with single numbers and arrays)
def returnY_pred(x, w, b):
    return w * np.array(x) + b


# Predictions for single test values
print(f"1000 sqft house prediction {returnY_pred(1.0, w_final, b_final):0.1f} Thousand dollars")
print(f"1200 sqft house prediction {returnY_pred(1.2, w_final, b_final):0.1f} Thousand dollars")
print(f"2000 sqft house prediction {returnY_pred(2.0, w_final, b_final):0.1f} Thousand dollars")

# Compute predictions for training set to calculate metrics
y_pred = returnY_pred(x_train, w_final, b_final)


def mse(y_true, y_pred):
    """Mean Squared Error — average of squared differences."""
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_true, y_pred):
    """Root Mean Squared Error — in same units as y."""
    return np.sqrt(mse(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# Evaluate performance metrics
print(f"RMSE: {rmse(y_train, y_pred):.4f}")
print(f"R²:   {r_squared(y_train, y_pred):.4f}")