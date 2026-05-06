"""
Path_Signatures_Pipeline.py
===========================
Reel pipeline — Lyons rough-path signatures applied to a BTC-style return path.

Hook: "Most traders see a price path. We see its tensor algebra."

Pipeline:
  1. Generate a 2D path X_t = (t, log-price) for a synthetic BTC-style series
     (heavy-tailed innovations + mild stochastic vol).
  2. Compute the truncated signature S(X)_{0,t}^{<=N} via iterated integrals,
     using the Chen-style additive update on each interval. Levels k=1,2,3.
  3. Visualize:
       Left  : 3D wireframe of the path (t, x, y) with the cumulative
               level-2 area term (Levy area) breathing as a hue-mapped ribbon.
       Right : level-1 (increments), level-2 (area), level-3 (volume)
               coordinates plotted as the window slides.

Dependencies: numpy, matplotlib  (pure NumPy — no iisignature dep needed for k<=3)
"""

import os, sys, time, warnings
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION":   (1920, 1080),
    "OUTPUT_IMAGE": "Path_Signatures_Output.png",
    "OUTPUT_VIDEO": "Path_Signatures_Output.mp4",
    "FRAME_DIR":    "temp_signature_frames",
    "LOG_FILE":     "signature_pipeline.log",
    "FPS":          30,
    "N_JOBS":       6,
}

SIG = {
    "n_points":   600,         # path resolution
    "T":          1.0,
    "sigma":      0.022,       # base vol per step
    "tail_df":    4.0,         # Student-t df for fat tails
    "vol_kappa":  3.0,         # OU mean-reversion of log-vol
    "vol_xi":     0.6,         # vol-of-vol
    "trunc":      3,           # signature truncation level
    "window":     120,         # rolling window size for live signature
    "seed":       7,
}

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "SPINE":      "#333333",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "ORANGE":     "#ff9500",
    "YELLOW":     "#ffd400",
    "CYAN":       "#00f2ff",
    "GREEN":      "#00ff7f",
    "RED":        "#ff3050",
    "PINK":       "#ff2a9e",
    "BLUE":       "#00bfff",
    "PALE":       "#88aaff",
    "FONT":       "Arial",
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — SYNTHETIC BTC-STYLE PATH
# ═══════════════════════════════════════════════════════════════════

def generate_path():
    """2D path X_t = (t, log-price) with stochastic vol + Student-t shocks."""
    p = SIG
    rng = np.random.default_rng(p["seed"])
    n = p["n_points"]
    dt = p["T"] / n

    # OU log-vol
    log_v = np.zeros(n)
    for i in range(1, n):
        log_v[i] = log_v[i-1] - p["vol_kappa"] * log_v[i-1] * dt + \
                   p["vol_xi"] * np.sqrt(dt) * rng.standard_normal()
    v = p["sigma"] * np.exp(log_v)

    z = rng.standard_t(p["tail_df"], size=n) / np.sqrt(p["tail_df"] / (p["tail_df"] - 2))
    increments = v * z
    log_price = np.cumsum(increments)

    t = np.linspace(0.0, p["T"], n)
    X = np.column_stack([t, log_price])
    return X


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — TRUNCATED SIGNATURE (Chen, levels 1..3)
# ═══════════════════════════════════════════════════════════════════
#
# For a 2D path X with components (X^1, X^2), the truncated signature
# S^{<=3}(X)_{0,T} contains:
#   level 1: (S^1, S^2)               -- 2 coords
#   level 2: (S^{11}, S^{12}, S^{21}, S^{22})  -- 4 coords
#   level 3: 8 coords (i,j,k in {1,2})
#
# We compute exactly via cumulative iterated integrals:
#   S^{i_1...i_k}_{0,t} = int_0^t S^{i_1...i_{k-1}}_{0,s} dX^{i_k}_s
# discretized with the trapezoidal rule.

def signature_running(X):
    """Return arrays of shape (n,) for each multi-index up to level 3."""
    t = X[:, 0]
    Xc = X - X[0:1, :]  # center so signature starts at 0

    dim = 2
    n = X.shape[0]

    # level 1
    S1 = {(i,): Xc[:, i].copy() for i in range(dim)}

    # helper: integrate prev level w.r.t. dX^i (left-Riemann is fine here)
    def integrate(prev_path, comp):
        dX = np.diff(Xc[:, comp])
        # trapezoidal: 0.5*(f_k + f_{k+1}) * dX_k
        f_avg = 0.5 * (prev_path[:-1] + prev_path[1:])
        out = np.zeros(n)
        out[1:] = np.cumsum(f_avg * dX)
        return out

    # level 2
    S2 = {}
    for i in range(dim):
        for j in range(dim):
            S2[(i, j)] = integrate(S1[(i,)], j)

    # level 3
    S3 = {}
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                S3[(i, j, k)] = integrate(S2[(i, j)], k)

    return S1, S2, S3


def levy_area(S2):
    """A(s,t) = 0.5 * (S^{12} - S^{21})  -- signed area enclosed by chord."""
    return 0.5 * (S2[(0, 1)] - S2[(1, 0)])


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — STATIC FRAME RENDER
# ═══════════════════════════════════════════════════════════════════

def render_static(X, S1, S2, S3, area, out_path):
    res = CONFIG["RESOLUTION"]
    fig = plt.figure(figsize=(res[0]/100, res[1]/100), dpi=100,
                     facecolor=THEME["BG"])
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.4, 1.0],
                           hspace=0.35, wspace=0.18,
                           left=0.04, right=0.97, top=0.92, bottom=0.07)

    # ─── LEFT: 3D path lifted into (t, x, y) with Levy-area ribbon ──
    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor=THEME["PANEL_BG"])
    t = X[:, 0]
    x = X[:, 1]
    y = area  # use Levy-area as third axis to lift the path

    # color by level-1 increment magnitude (hue-mapped trajectory)
    seg_color = np.tanh(np.gradient(x) * 50.0)
    for k in range(len(t) - 1):
        c = plt.cm.plasma(0.15 + 0.7 * (0.5 + 0.5 * seg_color[k]))
        ax3d.plot(t[k:k+2], x[k:k+2], y[k:k+2], color=c, linewidth=1.4, alpha=0.95)

    ax3d.set_xlabel("t", color=THEME["TEXT_DIM"])
    ax3d.set_ylabel("log price", color=THEME["TEXT_DIM"])
    ax3d.set_zlabel("Lévy area  A(0,t)", color=THEME["TEXT_DIM"])
    ax3d.set_title("Path lifted by signature   X_t  →  (t, X, A(0,t))",
                   color=THEME["TEXT"], fontsize=13, pad=14)
    ax3d.tick_params(colors=THEME["TEXT_DIM"])
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor(THEME["PANEL_BG"])
        pane.set_edgecolor(THEME["GRID"])

    # ─── RIGHT: signature coordinate stack (levels 1, 2, 3) ──
    titles = ["Level 1  —  increments  S^i",
              "Level 2  —  area / cross  S^{ij}",
              "Level 3  —  volume  S^{ijk}"]
    series_groups = [
        [("S^1", S1[(0,)], THEME["CYAN"]), ("S^2", S1[(1,)], THEME["ORANGE"])],
        [("S^{12}", S2[(0,1)], THEME["GREEN"]),
         ("S^{21}", S2[(1,0)], THEME["RED"]),
         ("A(0,t)", area, THEME["YELLOW"])],
        [("S^{121}", S3[(0,1,0)], THEME["PINK"]),
         ("S^{212}", S3[(1,0,1)], THEME["BLUE"]),
         ("S^{122}", S3[(0,1,1)], THEME["PALE"])],
    ]
    for row, (title, group) in enumerate(zip(titles, series_groups)):
        ax = fig.add_subplot(gs[row, 1], facecolor=THEME["PANEL_BG"])
        for label, series, col in group:
            ax.plot(t, series, color=col, lw=1.3, label=label, alpha=0.95)
        ax.set_title(title, color=THEME["TEXT"], fontsize=11, loc="left", pad=4)
        ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
        for s in ax.spines.values():
            s.set_color(THEME["SPINE"])
        ax.grid(True, color=THEME["GRID"], lw=0.4, alpha=0.5)
        ax.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                  labelcolor=THEME["TEXT_DIM"], fontsize=8, loc="upper left")

    fig.suptitle("Path Signatures   ·   truncated tensor S^{≤3}(X)",
                 color=THEME["TEXT"], fontsize=18, y=0.975)
    fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log("Path Signatures pipeline — start")
    X = generate_path()
    S1, S2, S3 = signature_running(X)
    area = levy_area(S2)
    log(f"Path: n={X.shape[0]}  range_x=[{X[:,1].min():.3f},{X[:,1].max():.3f}]")
    log(f"Final signature: S^1={S1[(0,)][-1]:.4f}  S^2={S1[(1,)][-1]:.4f}  "
        f"A(0,T)={area[-1]:.5f}")

    render_static(X, S1, S2, S3, area, CONFIG["OUTPUT_IMAGE"])
    log(f"Wrote static frame  →  {CONFIG['OUTPUT_IMAGE']}")
    log("Done.  (Video render: TODO — sweep window, save frames to FRAME_DIR, "
        "stitch via ffmpeg.)")


if __name__ == "__main__":
    main()
