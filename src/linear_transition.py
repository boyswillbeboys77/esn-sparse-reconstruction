import numpy as np

def learn_linear_transition_ABb(X_list, u_list, ridge=1e-6, washout=0):
    Phi_all, Y_all = [], []
    for X, u in zip(X_list, u_list):
        Xt  = X[washout:-1]
        Xt1 = X[washout+1:]
        ut  = u[washout:-1].reshape(-1, 1)
        ones = np.ones((Xt.shape[0], 1))
        Phi_all.append(np.hstack([Xt, ut, ones]))
        Y_all.append(Xt1)

    Phi = np.vstack(Phi_all)
    Y = np.vstack(Y_all)

    D = Phi.shape[1]
    W = np.linalg.solve(Phi.T @ Phi + ridge*np.eye(D), Phi.T @ Y)

    A = W[:-2].T
    B = W[-2:-1].T
    b = W[-1].reshape(-1, 1)
    return A, B, b
