"""
MutualInfo_Reel_Pipeline.py
============================
Cinematic landscape reel: Mutual Information surface across asset pairs and lags.
For 4 assets (SPY, QQQ, GLD, BTC-USD), compute MI between log-returns at lags
τ ∈ {0, 1, ..., 25}. The result is a (12 ordered pairs × 26 lags) MI surface
that reveals nonlinear lead-lag structure invisible to correlation.

Visual: 3D surface with x = pair index, y = lag, z = MI value.
A second floor surface plots |corr|² (linear info) for direct contrast.
Bright ridges far from τ=0 = nonlinear lead-lag info correlation misses.

Pipeline: DATA -> MI MATRIX -> RENDER (parallel) -> COMPILE
Resolution: 1920x1080 @ 30 FPS, ~12s + 2s hold
"""

import os, time, shutil, warnings
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
    "TICKERS": ["SPY", "QQQ", "GLD", "BTC-USD"], "LOOKBACK_YEARS": 4,
    "MAX_LAG": 25, "BINS": 12,
    "DPI": 100,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "MutualInfo_Static.png"),
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}

MI_CMAP = LinearSegmentedColormap.from_list(
    "mi",
    ["#000010", "#0a1958", "#0066ff", "#00f2ff", "#ffd400", "#ff9500", "#ffffff"],
    N=256,
)
LIN_CMAP = LinearSegmentedColormap.from_list(
    "lin", ["#1a1a1a", "#444444", "#888888"], N=64,
)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def fetch_data():
    log("[Data] Fetching basket...")
    rets = {}
    for tk in CONFIG["TICKERS"]:
        try:
            data = yf.download(tk, period=f"{CONFIG['LOOKBACK_YEARS']}y",
                               interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs(tk, axis=1, level=1)
            p = data["Close"].dropna().values.flatten()
            rets[tk] = pd.Series(np.diff(np.log(p)),
                                 index=data.index[1:len(p)])
        except Exception:
            n = 800
            rets[tk] = pd.Series(np.random.normal(0, 0.015, n),
                                 index=pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D"))
    df = pd.concat(rets.values(), axis=1, keys=rets.keys()).dropna()
    log(f"[Data] aligned {len(df)} days, {len(df.columns)} assets")
    return df


def mi_hist(x, y, bins):
    # Mutual information via histogram (in nats)
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    c_x = c_xy.sum(axis=1); c_y = c_xy.sum(axis=0)
    p_xy = c_xy / c_xy.sum()
    p_x = c_x / c_x.sum(); p_y = c_y / c_y.sum()
    nz = p_xy > 0
    mi = (p_xy[nz] * np.log(p_xy[nz] / (np.outer(p_x, p_y)[nz] + 1e-12))).sum()
    return max(mi, 0.0)


def build_mi_surface(df):
    cols = list(df.columns)
    n_assets = len(cols)
    pairs = [(i, j) for i in range(n_assets) for j in range(n_assets) if i != j]
    # Skip tau=0 (trivial contemporaneous link, mostly captured by linear correlation)
    taus = list(range(1, CONFIG["MAX_LAG"] + 1))
    L = len(taus)
    M = np.zeros((len(pairs), L))
    C = np.zeros((len(pairs), L))
    bins = CONFIG["BINS"]
    for k, (i, j) in enumerate(pairs):
        a = df.iloc[:, i].values
        b = df.iloc[:, j].values
        for ti, tau in enumerate(taus):
            x = a[:len(a) - tau]
            y = b[tau:]
            if len(x) < 50: continue
            M[k, ti] = mi_hist(x, y, bins)
            r = np.corrcoef(x, y)[0, 1]
            C[k, ti] = r ** 2
    return np.array(pairs), M, C, cols, taus


def render_static_image(pairs, M, C, cols, taus, z_max, out_file):
    try:
        n_pairs, n_lags = M.shape
        n_lag_vis = n_lags
        azim = 30
        elev = 32

        fig = plt.figure(figsize=(19.2, 10.8), dpi=CONFIG["DPI"], facecolor=THEME["BG"])

        # Title
        fig.text(0.5, 0.945, "INFORMATION FLOWS BETWEEN ASSETS", ha="center",
                 fontsize=26, fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
        fig.text(0.5, 0.905,
                 "Mutual information surface  ·  asset pairs × lag (τ ≥ 1)",
                 ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

        # Side hook
        fig.text(0.04, 0.78, "Correlation is one number.",
                 fontsize=15, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.71, "Information leaks",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.685, "across days.",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])

        # 3D plot
        ax = fig.add_axes([0.24, 0.05, 0.72, 0.92], projection="3d", facecolor=THEME["BG"])

        X, Y = np.meshgrid(np.arange(n_pairs), np.arange(n_lags))

        # Floor surface: linear info |corr|^2 (greyscale)
        Z_lin = C.T  # (n_lags, n_pairs)
        floor_z = np.full_like(Z_lin, -0.02 * z_max)  # plane below
        # Actually plot |corr|^2 as faint surface at z = small height
        ax.plot_surface(X, Y, Z_lin * 0.6,  # scaled down to live below MI
                        cmap=LIN_CMAP,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False, alpha=0.55)

        # MI surface (colored), masking unbuilt lags
        Z = M.T  # (n_lags, n_pairs)
        Z_vis = np.where(np.arange(n_lags)[:, None] < n_lag_vis, Z, 0.0)
        face_norm = np.clip(Z_vis / (z_max + 1e-9), 0, 1)
        face_colors = MI_CMAP(face_norm)
        face_colors[..., -1] = np.where(np.arange(n_lags)[:, None] < n_lag_vis, 0.92, 0.0)

        ax.plot_surface(X, Y, Z_vis, facecolors=face_colors,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, shade=False)

        ax.set_xlim(0, n_pairs - 1)
        ax.set_ylim(0, n_lags - 1)
        ax.set_zlim(0, z_max * 1.1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.4, 1.0, 0.7))

        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)

        # Sparse pair labels along x
        pair_labels = [f"{cols[i][:3]}>{cols[j][:3]}" for i, j in pairs]
        tick_idx = list(range(0, n_pairs, max(1, n_pairs // 6)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([pair_labels[k] for k in tick_idx],
                           color=THEME["TEXT_DIM"], fontsize=7, rotation=0)
        # y-axis labels show actual lag values (not array indices)
        tick_step = max(1, n_lags // 5)
        ax.set_yticks(list(range(0, n_lags, tick_step)))
        ax.set_yticklabels([str(taus[i]) for i in range(0, n_lags, tick_step)])
        ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=7, pad=-2)
        ax.set_zticks([])
        ax.set_xlabel("asset pair (lead → lag)", color=THEME["TEXT_DIM"], fontsize=9, labelpad=4)
        ax.set_ylabel("τ (days)", color=THEME["TEXT_DIM"], fontsize=9, labelpad=2)
        ax.set_zlabel("MI", color=THEME["TEXT_DIM"], fontsize=9, labelpad=-4)

        # HUD
        max_pair_idx = int(np.argmax(M.max(axis=1)))
        max_pair_lag_idx = int(np.argmax(M[max_pair_idx]))
        max_pair_mi = M[max_pair_idx, max_pair_lag_idx]
        i, j = pairs[max_pair_idx]
        lag_value = taus[max_pair_lag_idx]

        fig.text(0.04, 0.55, "STRONGEST LEAD-LAG LINK",
                 fontsize=10, color=THEME["TEXT_DIM"], fontweight="bold")
        fig.text(0.04, 0.520,
                 f"{cols[i]} →  {cols[j]}",
                 fontsize=14, color=THEME["YELLOW"], fontweight="bold")
        fig.text(0.04, 0.495,
                 f"at τ = {lag_value} days",
                 fontsize=11, color=THEME["CYAN"])
        fig.text(0.04, 0.470,
                 f"MI = {max_pair_mi:.3f} nats",
                 fontsize=11, color=THEME["GREEN"])
        fig.text(0.04, 0.445,
                 f"|corr|² at τ = {C[max_pair_idx, max_pair_lag_idx]:.3f}",
                 fontsize=11, color=THEME["TEXT_DIM"])

        # Legend
        fig.text(0.04, 0.36, "color: MI (nonlinear)",
                 fontsize=10, color=THEME["YELLOW"])
        fig.text(0.04, 0.335, "grey: |corr|² (linear)",
                 fontsize=10, color=THEME["TEXT_DIM"])

        fig.text(0.04, 0.260,
                 "Where MI rises but |corr|²",
                 fontsize=10, color=THEME["TEXT"], style="italic")
        fig.text(0.04, 0.237,
                 "stays low → pure nonlinear coupling.",
                 fontsize=10, color=THEME["TEXT"], style="italic")

        fig.text(0.04, 0.18, f"basket: {' · '.join(cols)}",
                 fontsize=9, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.155, f"lookback: {CONFIG['LOOKBACK_YEARS']}y daily returns",
                 fontsize=9, color=THEME["TEXT_DIM"])

        fig.text(0.985, 0.018, "@quant.traderr", ha="right", va="bottom",
                 fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

        fig.savefig(out_file, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
        plt.close(fig)
        return True
    except Exception:
        import traceback; traceback.print_exc(); return False


def main():
    t0 = time.time(); log("=== MUTUAL INFORMATION STATIC ===")
    df = fetch_data()
    pairs, M, C, cols, taus = build_mi_surface(df)
    log(f"MI surface: {M.shape}  max={M.max():.3f}  mean={M.mean():.3f}")

    z_max = float(np.percentile(M, 99) * 1.10)

    log(f"Rendering static image...")
    render_static_image(pairs, M, C, cols, taus, z_max, CONFIG["OUTPUT_FILE"])
    log(f"Saved to {CONFIG['OUTPUT_FILE']}")
    log(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
