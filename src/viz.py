import matplotlib.pyplot as plt
import numpy as np

def plot_training_signals(t, y_list, idx_obs, washout, show_n=10):
    n = min(show_n, len(y_list))
    plt.figure(figsize=(12, 5))
    for i in range(n):
        y = y_list[i]
        plt.plot(t, y, lw=2.0, alpha=0.85, label=f"train #{i+1}")
        plt.scatter(t[idx_obs], y[idx_obs], s=20, alpha=0.85)
    plt.axvline(t[washout], ls=":", c="k", alpha=0.6, label="washout boundary")
    plt.title("Training y(t) (examples)  (dots = observed x-times)")
    plt.xlabel("t"); plt.ylabel("y")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

def plot_test_example(t, y, yhat, idx_obs, washout, title):
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, lw=2.2, label="true y")
    plt.plot(t, yhat, "--", lw=2.0, label="y_hat")
    plt.scatter(t[idx_obs], y[idx_obs], s=26, alpha=0.85, label="y at observed x-times")
    plt.axvline(t[washout], ls=":", c="k", alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_zoom_after_washout(t, y, yhat, idx_obs, washout, zoom_len=85, title="ZOOM"):
    end = min(len(t), washout + zoom_len)
    sl = slice(washout, end)

    ymin = min(y[sl].min(), yhat[sl].min())
    ymax = max(y[sl].max(), yhat[sl].max())
    pad = 0.1 * (ymax - ymin + 1e-12)

    idx_obs_in = idx_obs[(idx_obs >= washout) & (idx_obs < end)]

    plt.figure(figsize=(12, 4))
    plt.plot(t[sl], y[sl], lw=2.6, label="true y (zoom)")
    plt.plot(t[sl], yhat[sl], "--", lw=2.2, label="y_hat (zoom)")
    if len(idx_obs_in) > 0:
        plt.scatter(t[idx_obs_in], y[idx_obs_in], s=35, alpha=0.95, label="y at observed times")
    plt.ylim(ymin - pad, ymax + pad)
    plt.title(title)
    plt.xlabel("t"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
