import numpy as np

def train_wout(X_list, y_list, ridge_lambda=1e-4, washout=0):
    Z_all, Y_all = [], []
    for X, y in zip(X_list, y_list):
        Z_all.append(np.hstack([np.ones((len(y)-washout, 1)), X[washout:]]))
        Y_all.append(y[washout:].reshape(-1, 1))
    Z = np.vstack(Z_all)
    Y = np.vstack(Y_all)
    return np.linalg.solve(Z.T @ Z + ridge_lambda*np.eye(Z.shape[1]), Z.T @ Y)

def apply_wout(Wout, X):
    Z = np.hstack([np.ones((len(X), 1)), X])
    return (Z @ Wout).ravel()

def apply_wout_single(Wout, x):
    Z = np.hstack([1.0, x]).reshape(1, -1)
    return float((Z @ Wout).ravel()[0])
