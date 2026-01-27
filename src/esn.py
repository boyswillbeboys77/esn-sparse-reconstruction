import numpy as np

def build_true_esn(M, spectral_radius, seed):
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-1, 1, size=M)
    W = rng.uniform(-1, 1, size=(M, M))
    rho = np.max(np.abs(np.linalg.eigvals(W))) + 1e-12
    Wres = (spectral_radius / rho) * W
    return Win, Wres

def run_true_esn_states(u, Win, Wres):
    T = len(u)
    M = len(Win)
    x = np.zeros(M)
    X = np.zeros((T, M))
    for t in range(T):
        x = np.tanh(Win * u[t] + Wres @ x) + 1
        X[t] = x
    return X
