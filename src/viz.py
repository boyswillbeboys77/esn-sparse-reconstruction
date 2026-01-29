import matplotlib.pyplot as plt
import numpy as np

def plot_zoom_after_washout(t, y, yhat, idx_obs, washout, zoom_len=85, title="ZOOM"):
    end = min(len(t), washout + zoom_len)
    sl = slice(washout, end)

    ymin = min(y[sl].min(), yhat[sl].min())
    ymax = max(y[sl].max(), yhat[sl].max())
    pad = 0.1 * (ymax - ymin + 1e-12)

    idx_obs_in = idx_obs[(idx_obs >= washout) & (idx_obs < end)]

    plt.figure(figsize=(12, 4))
    plt.plot(t[sl], y[sl], lw=2.6, label="true y")
    plt.plot(t[sl], yhat[sl], "--", lw=2.2, label="reconstructed y")
    if len(idx_obs_in) > 0:
        plt.scatter(t[idx_obs_in], y[idx_obs_in], s=100, alpha=0.95, label="y at observed times")
    plt.ylim(ymin - pad, ymax + pad)
    plt.title(title)
    plt.xlabel("t"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
