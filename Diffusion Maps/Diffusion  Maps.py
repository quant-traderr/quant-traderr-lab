"""
DiffusionMaps_Reel_Pipeline.py
===============================
Cinematic landscape reel: Diffusion-map embedding of BTC return states.
Each day's local return window (sliding 10-day vector) is treated as a state.
Build a Markov transition kernel between states, eigendecompose, and embed
the top 3 nontrivial eigenvectors into 3D — points that the random walk can
reach quickly cluster together. Result: regime "highways" in 3D.

Cluster the embedding (k-means, K=3) and color by regime: low-vol, mid, high-vol.
Camera rotates as points fade in chronologically.

Pipeline: DATA -> KERNEL -> EIGEN -> CLUSTER -> RENDER (parallel)
Resolution: 1920x1080 @ 30 FPS, ~12s
"""

import os, time, shutil, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKER": "BTC-USD", "LOOKBACK_DAYS": 480, "WINDOW": 10, "K_CLUSTERS": 3,
    "FPS": 30, "DURATION_SEC": 12, "HOLD_LAST_SEC": 2,
    "DPI": 100, "N_WORKERS": 6,
    "TEMP_DIR": os.path.join(BASE_DIR, "temp_diffmap_frames"),
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "MAGENTA": "#ff1493", "FONT": "Arial",
}
CLUSTER_COLORS = [THEME["CYAN"], THEME["YELLOW"], THEME["RED"]]
CLUSTER_NAMES = ["LOW VOL", "MID VOL", "HIGH VOL"]


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def fetch_data():
    log(f"[Data] Fetching {CONFIG['TICKER']}...")
    try:
        data = yf.download(CONFIG["TICKER"], period="3y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG["TICKER"], axis=1, level=1)
        prices = data["Close"].values.flatten()
        returns = np.diff(np.log(prices)) * 100
        return returns[-CONFIG["LOOKBACK_DAYS"]:]
    except Exception:
        return np.random.normal(0, 2.5, CONFIG["LOOKBACK_DAYS"])


def diffusion_embed(returns):
    w = CONFIG["WINDOW"]
    n_states = len(returns) - w + 1
    states = np.array([returns[i:i + w] for i in range(n_states)])
    # Standardize features
    states = (states - states.mean(axis=0)) / (states.std(axis=0) + 1e-9)

    # Pairwise squared distances (vectorized)
    sq = np.sum(states ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2 * states @ states.T
    D2 = np.maximum(D2, 0)

    sigma2 = np.median(D2[D2 > 0])
    K = np.exp(-D2 / (2 * sigma2 + 1e-9))

    # Symmetric normalization for diffusion map
    d = K.sum(axis=1)
    K_norm = K / np.sqrt(np.outer(d, d))
    # Eigendecompose symmetric kernel
    vals, vecs = np.linalg.eigh(K_norm)
    # Sort descending
    order = np.argsort(-vals)
    vals = vals[order]; vecs = vecs[:, order]
    # Skip the trivial 0-th eigenvector (constant)
    phi = vecs[:, 1:4]  # (n, 3)
    # Robust scaling: rank-normalize each axis to spread points uniformly
    from scipy.stats import rankdata
    phi_ranked = np.column_stack([
        (rankdata(phi[:, k]) / len(phi) - 0.5) * 2.0
        for k in range(3)
    ])
    phi = phi_ranked

    # K-means on phi embedding so cluster labels match SPATIAL groups in φ-space
    state_vol = np.std(states, axis=1)
    K = CONFIG["K_CLUSTERS"]
    rng = np.random.default_rng(7)
    # Init: pick K points spread across phi
    init_idx = rng.choice(n_states, size=K, replace=False)
    centers = phi[init_idx].copy()
    labels = np.zeros(n_states, dtype=int)
    for _ in range(80):
        # Assign
        d2 = np.sum((phi[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = phi[mask].mean(axis=0)
    # Re-order labels so cluster 0=lowest mean vol, 2=highest (matches color names)
    cluster_vols = np.array([state_vol[labels == k].mean() if (labels == k).any() else 0
                             for k in range(K)])
    order = np.argsort(cluster_vols)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])

    return phi, labels, state_vol


_SH = {}
def _init(data):
    global _SH; _SH = data


def render_frame(idx):
    try:
        d = _SH
        phi, labels, vol = d["phi"], d["labels"], d["vol"]
        total, tmp = d["total"], d["tmp"]
        lim = d["lim"]

        n = phi.shape[0]
        t = idx / max(total - 1, 1)

        reveal_t = min(t / 0.85, 1.0)
        n_vis = max(2, int(reveal_t * n))

        azim = -65 + 90 * t
        elev = 22 + 6 * np.sin(t * np.pi)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=CONFIG["DPI"], facecolor=THEME["BG"])

        # Title
        fig.text(0.5, 0.945, "REGIMES, REVEALED BY GEOMETRY", ha="center",
                 fontsize=28, fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
        fig.text(0.5, 0.905,
                 f"Diffusion-map embedding  ·  {CONFIG['TICKER']}  ·  10-day return states",
                 ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

        # Side hook
        fig.text(0.04, 0.78, "PCA shows variance.",
                 fontsize=15, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.755, "Diffusion shows the",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.730, "highways between regimes.",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])

        # 3D plot
        ax = fig.add_axes([0.24, 0.05, 0.72, 0.92], projection="3d", facecolor=THEME["BG"])

        for k in range(CONFIG["K_CLUSTERS"]):
            mask = (labels[:n_vis] == k)
            if mask.sum() == 0: continue
            x = phi[:n_vis, 0][mask]
            y = phi[:n_vis, 1][mask]
            z = phi[:n_vis, 2][mask]
            ax.scatter(x, y, z, s=22, c=CLUSTER_COLORS[k],
                       edgecolors="none", alpha=0.85, depthshade=False)

        # (chronological line removed — created visual noise across cluster regions)

        # Glowing leading point
        if n_vis > 0:
            cur_label = labels[n_vis - 1]
            ax.scatter([phi[n_vis - 1, 0]], [phi[n_vis - 1, 1]], [phi[n_vis - 1, 2]],
                       s=70, color=CLUSTER_COLORS[cur_label],
                       edgecolors="white", linewidths=0.7, zorder=10)

        ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))

        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("φ₁", color=THEME["TEXT_DIM"], fontsize=10, labelpad=-10)
        ax.set_ylabel("φ₂", color=THEME["TEXT_DIM"], fontsize=10, labelpad=-10)
        ax.set_zlabel("φ₃", color=THEME["TEXT_DIM"], fontsize=10, labelpad=-10)

        # HUD: cluster legend
        fig.text(0.04, 0.62, "REGIMES", fontsize=11, color=THEME["TEXT_DIM"],
                 fontweight="bold", family=THEME["FONT"])
        for k in range(CONFIG["K_CLUSTERS"]):
            mask = labels[:n_vis] == k
            cnt = mask.sum()
            fig.text(0.04, 0.58 - k * 0.030,
                     f"●  {CLUSTER_NAMES[k]}   {cnt} days",
                     fontsize=11, color=CLUSTER_COLORS[k], family=THEME["FONT"])

        fig.text(0.04, 0.45, "Method:", fontsize=10, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.425, "1. State = 10-day return window",
                 fontsize=9, color=THEME["TEXT"])
        fig.text(0.04, 0.405, "2. Gaussian kernel between states",
                 fontsize=9, color=THEME["TEXT"])
        fig.text(0.04, 0.385, "3. Markov transition matrix P",
                 fontsize=9, color=THEME["TEXT"])
        fig.text(0.04, 0.365, "4. Eigenvectors of P → 3D embed",
                 fontsize=9, color=THEME["TEXT"])

        fig.text(0.04, 0.30, f"states embedded: {n_vis} / {n}",
                 fontsize=11, color=THEME["YELLOW"], family=THEME["FONT"])

        if n_vis > 0:
            fig.text(0.04, 0.255, f"current σ = {vol[n_vis-1]:.2f}%",
                     fontsize=10, color=CLUSTER_COLORS[labels[n_vis-1]],
                     fontweight="bold")

        fig.text(0.04, 0.18, "Same data.",
                 fontsize=10, color=THEME["TEXT_DIM"], style="italic")
        fig.text(0.04, 0.155, "Different geometry.",
                 fontsize=10, color=THEME["TEXT_DIM"], style="italic")
        fig.text(0.04, 0.13, "Three regimes appear",
                 fontsize=10, color=THEME["TEXT"], style="italic")
        fig.text(0.04, 0.107, "as separate clusters.",
                 fontsize=10, color=THEME["TEXT"], style="italic")

        fig.text(0.985, 0.018, "@quant.traderr", ha="right", va="bottom",
                 fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

        fp = os.path.join(tmp, f"frame_{idx:04d}.png")
        fig.savefig(fp, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
        plt.close(fig)
        if idx % 30 == 0:
            print(f"  [worker] frame {idx}/{total}")
        return True
    except Exception:
        import traceback; traceback.print_exc(); return False


def main():
    t0 = time.time(); log("=== DIFFUSION MAPS REEL ===")
    returns = fetch_data()
    phi, labels, vol = diffusion_embed(returns)
    log(f"Embedding shape: {phi.shape}  cluster counts: {np.bincount(labels)}")

    pad = 0.1
    lim = []
    for k in range(3):
        col = phi[:, k]
        rng = np.ptp(col)
        lim.append((col.min() - pad * rng, col.max() + pad * rng))

    total = CONFIG["FPS"] * CONFIG["DURATION_SEC"]
    tmp = CONFIG["TEMP_DIR"]
    if os.path.exists(tmp): shutil.rmtree(tmp)
    os.makedirs(tmp)

    shared = {"phi": phi, "labels": labels, "vol": vol,
              "total": total, "tmp": tmp, "lim": lim}

    log(f"Rendering {total} frames on {CONFIG['N_WORKERS']} workers...")
    with Pool(CONFIG["N_WORKERS"], _init, (shared,)) as pool:
        res = list(pool.imap_unordered(render_frame, range(total), chunksize=2))
    log(f"Rendered {sum(1 for r in res if r)}/{total}")

    log(f"Frames saved in {tmp}")
    log(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
