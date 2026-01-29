import argparse
import os
import numpy as np

from src.esn import build_true_esn, run_true_esn_states
from src.data import make_random_sine
from src.linear_transition import learn_linear_transition_ABb
from src.readout import train_wout, apply_wout
from src.rollout import rollout_with_reset_using_uhat_from_sparse_xobs
from src.metrics import rmse, nrmse, r2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--washout", type=int, default=20)
    p.add_argument("--step_obs", type=int, default=60)
    p.add_argument("--N_train", type=int, default=100)
    p.add_argument("--N_test", type=int, default=30)

    p.add_argument("--M", type=int, default=100)
    p.add_argument("--spectral_radius", type=float, default=0.95)
    p.add_argument("--seed_esn", type=int, default=1)
    p.add_argument("--seed_train", type=int, default=0)
    p.add_argument("--seed_test", type=int, default=1)

    p.add_argument("--K_obs", type=int, default=100)

    p.add_argument("--ridge_trans", type=float, default=1e-6)
    p.add_argument("--ridge_wout", type=float, default=1e-4)

    p.add_argument("--show_train", type=int, default=10)
    p.add_argument("--show_test", type=int, default=5)
    p.add_argument("--zoom_len", type=int, default=85)

    p.add_argument("--plot", type=int, default=0, help="0: no plots, 1: show plots")
    p.add_argument("--save_dir", type=str, default="results", help="where to save plots if plot=1")
    return p.parse_args()

def main():
    args = parse_args()

    # --- optional viz import ---
    if args.plot == 1:
        import matplotlib.pyplot as plt
        from src.viz import plot_zoom_after_washout
        os.makedirs(args.save_dir, exist_ok=True)

    # --- time + obs index ---
    t = np.linspace(0, 1, args.T)
    idx_obs = np.arange(args.washout, args.T, args.step_obs)

    # --- ESN ---
    Win, Wres = build_true_esn(args.M, args.spectral_radius, seed=args.seed_esn)

    # --- RNG separation (reproducibility) ---
    rng_train = np.random.default_rng(args.seed_train)
    rng_test  = np.random.default_rng(args.seed_test)

    # observed nodes
    obs_nodes = list(range(args.K_obs))

    # ============================================================
    # training
    # ============================================================
    Xobs_list, y_list, cyc_list = [], [], []
    for _ in range(args.N_train):
        y, cyc = make_random_sine(t, rng_train)
        X_full = run_true_esn_states(y, Win, Wres)
        Xobs = X_full[:, obs_nodes]
        Xobs_list.append(Xobs)
        y_list.append(y)
        cyc_list.append(cyc)

    # learn transition + first readout
    A, B, b = learn_linear_transition_ABb(Xobs_list, y_list, args.ridge_trans, args.washout)
    Wout1 = train_wout(Xobs_list, y_list, args.ridge_wout, args.washout)

    # reconstruct x
    Xhat_list = [
        rollout_with_reset_using_uhat_from_sparse_xobs(
            Xobs[idx_obs], idx_obs, A, B, b, Wout1, args.T
        )
        for Xobs in Xobs_list
    ]

    # second readout (from reconstructed states)
    Wout2 = train_wout(Xhat_list, y_list, args.ridge_wout, args.washout)

    # ============================================================
    # test
    # ============================================================
    scores = []
    shown = 0

    for k in range(args.N_test):
        y, cyc = make_random_sine(t, rng_test)
        X_full = run_true_esn_states(y, Win, Wres)
        Xobs = X_full[:, obs_nodes]

        Xhat = rollout_with_reset_using_uhat_from_sparse_xobs(
            Xobs[idx_obs], idx_obs, A, B, b, Wout1, args.T
        )
        yhat = apply_wout(Wout2, Xhat)

        yt = y[args.washout:]
        yp = yhat[args.washout:]
        scores.append((rmse(yt, yp), nrmse(yt, yp), r2(yt, yp)))

        if args.plot == 1 and shown < args.show_test:
            shown += 1
            title = (
                f"TEST {shown} cycles={cyc:.3f} "
                f"RMSE={scores[-1][0]:.2e}, nRMSE={scores[-1][1]:.2e}, R2={scores[-1][2]:.4f}"
            )
            plot_zoom_after_washout(
                t, y, yhat, idx_obs, args.washout, args.zoom_len,
                title=f"TEST {shown} ZOOM after washout (K={args.K_obs})"
            )
            plt.savefig(os.path.join(args.save_dir, f"test_{shown}_zoom.png"), dpi=200)
            plt.show()

    scores = np.array(scores)

    # ============================================================
    # summary
    # ============================================================
    print("=== Executed successfully ===")

if __name__ == "__main__":
    main()
