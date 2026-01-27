import numpy as np

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))

def nrmse(a, b, eps=1e-12):
    return rmse(a, b) / (np.std(a) + eps)

def r2(a, b, eps=1e-12):
    a = np.asarray(a); b = np.asarray(b)
    return 1 - np.sum((b - a) ** 2) / (np.sum((a - a.mean()) ** 2) + eps)
