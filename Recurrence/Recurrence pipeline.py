"""
Recurrence_Reel_Pipeline.py
============================
BTC recurrence plot — distance matrix D[i,j] = |σ_i − σ_j| converted to
recurrence intensity R(i,j) = exp(−D / median). Bright cells = market
state at i resembles state at j. Dark cells = unique regime.

Static-image only. The video render (frame loop + compile) was removed;
this script now produces a single composed PNG of the final 3D topographic
view at full reveal.

Pipeline: DATA -> DISTANCE MATRIX -> SAVE STATIC PNG
Resolution: 1920x1080
"""

import os, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKER": "BTC-USD", "LOOKBACK_DAYS": 360, "WIN": 5,
    "DPI": 100,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "Recurrence_Plot.png"),
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "FONT": "Arial",
}

REC_CMAP = LinearSegmentedColormap.from_list(
    "recurrence",
    ["#000010", "#04144a", "#0066ff", "#00f2ff", "#ffd400", "#ff9500", "#ffffff"],
    N=256,
)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def fetch_data():
    log(f"[Data] Fetching {CONFIG['TICKER']}...")
    try:
        data = yf.download(CONFIG["TICKER"], period="2y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG["TICKER"], axis=1, level=1)
        prices = data["Close"].values.flatten()
        returns = np.diff(np.log(prices)) * 100
        return returns[-CONFIG["LOOKBACK_DAYS"]:]
    except Exception:
        return np.random.normal(0, 2.5, CONFIG["LOOKBACK_DAYS"])


def build_recurrence(returns):
    n = len(returns)
    w = CONFIG["WIN"]
    state = np.array([np.std(returns[max(0, i - w):i + 1]) for i in range(n)])
    state = (state - state.mean()) / (state.std() + 1e-9)
    D = np.abs(state[:, None] - state[None, :])
    sigma = np.median(D)
    R = np.exp(-D / (sigma + 1e-9))
    try:
        from scipy.ndimage import gaussian_filter
        R = gaussian_filter(R, sigma=2.5)
    except Exception:
        pass
    return R, state


def save_static(R, state, out_path):
    n = R.shape[0]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=CONFIG["DPI"], facecolor=THEME["BG"])

    # Title
    fig.text(0.5, 0.945, "MARKETS REPEAT THEMSELVES", ha="center",
             fontsize=28, fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.905, f"Recurrence plot  ·  {CONFIG['TICKER']}  ·  {n} days",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

    # Side hook
    fig.text(0.04, 0.78, "The chart looks new.",
             fontsize=15, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
    fig.text(0.04, 0.74, "The geometry says\nit's been here before.",
             fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"],
             linespacing=1.3)
    fig.text(0.04, 0.62, "Bright cells = recurrence",
             fontsize=10, color=THEME["YELLOW"], family=THEME["FONT"])
    fig.text(0.04, 0.595, "(state at row i ≈ state at col j)",
             fontsize=9, color=THEME["TEXT_DIM"], style="italic")

    # 3D plot
    ax = fig.add_axes([0.26, 0.05, 0.70, 0.92], projection="3d", facecolor=THEME["BG"])

    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    rs, cs = 3, 3
    face_norm = np.clip(R, 0, 1)
    face_colors = REC_CMAP(face_norm)
    face_colors[..., -1] = 0.95

    ax.plot_surface(X, Y, R, facecolors=face_colors, rstride=rs, cstride=cs,
                    linewidth=0, antialiased=False, shade=False)

    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, 1.05)
    ax.view_init(elev=30, azim=-65)
    ax.set_box_aspect((1, 1, 0.32))

    ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # Method block
    fig.text(0.04, 0.49, "RECURRENCE TOPOGRAPHY", fontsize=11, color=THEME["CYAN"],
             fontweight="bold", family=THEME["FONT"])
    fig.text(0.04, 0.46, f"D(i,j) = |σ_i − σ_j|", fontsize=12, color=THEME["TEXT"])
    fig.text(0.04, 0.43, f"R(i,j) = exp(−D / median)",
             fontsize=11, color=THEME["TEXT_DIM"])

    fig.text(0.04, 0.30, f"days: {n}", fontsize=11, color=THEME["YELLOW"],
             family=THEME["FONT"])

    # Legend strip
    ax_legend = fig.add_axes([0.30, 0.045, 0.60, 0.018])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_legend.imshow(gradient, aspect="auto", cmap=REC_CMAP)
    ax_legend.set_xticks([]); ax_legend.set_yticks([])
    for s in ax_legend.spines.values():
        s.set_color(THEME["TEXT_DIM"]); s.set_linewidth(0.4)
    fig.text(0.30, 0.025, "novel", fontsize=9, color=THEME["TEXT_DIM"], ha="left")
    fig.text(0.90, 0.025, "recurrent", fontsize=9, color=THEME["TEXT_DIM"], ha="right")

    fig.text(0.985, 0.018, "@quant.traderr", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static image saved: {out_path}")


def main():
    t0 = time.time(); log("=== RECURRENCE PLOT (static) ===")
    returns = fetch_data()
    R, state = build_recurrence(returns)
    log(f"Matrix {R.shape}  density mean={R.mean():.3f}")
    save_static(R, state, CONFIG["OUTPUT_FILE"])
    log(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
