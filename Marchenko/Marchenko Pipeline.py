"""
MarchenkoPastur_Static_Pipeline.py
====================================
Single 1920x1080 hero image for the Marchenko-Pastur eigenvalue noise floor.
Reuses the cached rolling-eigenvalues bundle from the reel pipeline.

Layout (landscape):
    Top  : title strip
    Left : 3D eigenvalue trails over time (log-z), MP bulk planes,
           market-mode red trail with glowing head
    Right: stacked panels --
              * current eigenvalue histogram + MP density theory
              * HUD: window end, market eigenvalue, above-bulk count
              * equation foot
"""

import os, sys, time, warnings, pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import the reel pipeline's CONFIG/THEME and helpers via sibling import
sys.path.insert(0, BASE_DIR)
from MarchenkoPastur_Reel_Pipeline import (  # noqa: E402
    THEME, mp_density, load_or_compute,
)

CONFIG = {
    "WIDTH":  1920, "HEIGHT": 1080, "DPI": 100,
    "FOCUS_FRAC": 0.40,   # which window to render (0..1 across timeline)
    "OUTPUT_FILE": os.path.join(BASE_DIR, "MarchenkoPastur_Static.png"),
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def render_static(data, frac, out_path):
    eigs_t    = data["eigs_t"]
    dates_t   = data["dates_t"]
    Q         = data["Q"]
    lam_plus  = data["lam_plus"]
    lam_minus = data["lam_minus"]
    N         = data["N"]

    n_w = eigs_t.shape[0]
    cur = int(np.clip(round(frac * (n_w - 1)), 0, n_w - 1))

    def Z(x): return np.log10(np.maximum(x, 1e-3))
    z_lam_plus  = Z(lam_plus)
    z_lam_minus = Z(lam_minus)
    z_market_max = Z(eigs_t[:, -1].max())

    fig = plt.figure(figsize=(CONFIG["WIDTH"]/CONFIG["DPI"],
                              CONFIG["HEIGHT"]/CONFIG["DPI"]),
                     dpi=CONFIG["DPI"], facecolor=THEME["BG"])

    # ---------- TITLE ----------
    fig.text(0.025, 0.945, "MARCHENKO-PASTUR",
             fontsize=22, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.025, 0.910,
             "rolling eigenvalues vs the analytic noise floor",
             fontsize=12, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.975, 0.945, "QUANT . LAB", ha="right",
             fontsize=11, color=THEME["TEXT_DIM"], family=THEME["FONT"])

    # ---------- 3D HERO (LEFT) ----------
    ax3d = fig.add_axes([-0.02, 0.05, 0.66, 0.84],
                        projection="3d", facecolor=THEME["BG"])

    t_axis = np.arange(cur + 1)
    cur_eigs = eigs_t[: cur + 1]

    # MP bulk planes (translucent yellow) in log-z
    xx = np.array([[0, n_w], [0, n_w]])
    yy = np.array([[-0.5, -0.5], [0.5, 0.5]])
    for lam, alpha in [(lam_plus, 0.22), (lam_minus, 0.10)]:
        zz = np.full_like(xx, Z(lam), dtype=float)
        ax3d.plot_surface(xx, yy, zz, color=THEME["YELLOW"],
                          alpha=alpha, linewidth=0, antialiased=False, shade=False)

    T_mesh, _ = np.meshgrid(t_axis, np.arange(N), indexing="ij")
    E = cur_eigs
    Zm = Z(E)
    above = E > lam_plus
    below = E < lam_minus
    inside = ~above & ~below

    ax3d.scatter(T_mesh[inside], np.zeros(inside.sum()), Zm[inside],
                 s=2.2, color="#888888", alpha=0.5, depthshade=False)
    if below.any():
        ax3d.scatter(T_mesh[below], np.zeros(below.sum()), Zm[below],
                     s=4, color=THEME["CYAN"], alpha=0.5, depthshade=False)
    if above.any():
        mag = E[above]
        ax3d.scatter(T_mesh[above], np.zeros(above.sum()), Zm[above],
                     s=6 + 1.1 * mag, color=THEME["ORANGE"],
                     alpha=0.85, edgecolors="none", depthshade=False)

    # Market mode red trail
    market = cur_eigs[:, -1]
    market_z = Z(market)
    pts = np.array([t_axis, np.zeros(cur + 1), market_z]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    col = np.zeros((len(segs), 4))
    col[:, 0] = 1.0; col[:, 1] = 0.19; col[:, 2] = 0.31
    col[:, 3] = np.linspace(0.4, 1.0, len(segs)) ** 1.2
    ax3d.add_collection3d(Line3DCollection(segs, colors=col, linewidths=2.4))

    for s, a in [(500, 0.10), (250, 0.25), (120, 0.6), (50, 1.0)]:
        ax3d.scatter([t_axis[-1]], [0], [market_z[-1]],
                     s=s, color=THEME["RED"], alpha=a,
                     edgecolors="none", depthshade=False)

    # Floating labels
    ax3d.text(n_w * 0.02, 0, z_market_max * 0.97,
              "MARKET MODE", color=THEME["RED"], fontsize=11, fontweight="bold")
    ax3d.text(n_w * 0.02, 0, (z_lam_plus + z_lam_minus) / 2,
              "MP NOISE BULK", color=THEME["YELLOW"], fontsize=9)
    ax3d.text(n_w * 0.55, 0, Z(lam_minus * 0.55),
              "bulk noise (below λ_-)",
              color=THEME["CYAN"], fontsize=8, alpha=0.7)

    ax3d.view_init(elev=12, azim=-58)
    ax3d.set_xlim(0, n_w); ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(Z(0.12), z_market_max + 0.10)
    ax3d.set_box_aspect((1.7, 0.5, 1.5))
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_alpha(0); pane.set_edgecolor((0, 0, 0, 0))
    ax3d.grid(False)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

    # ---------- RIGHT: histogram ----------
    ax_h = fig.add_axes([0.66, 0.55, 0.32, 0.30], facecolor=THEME["PANEL"])
    bulk = E[cur][E[cur] < lam_plus * 2.5]
    bins = np.linspace(0, lam_plus * 2.5, 50)
    ax_h.hist(bulk, bins=bins, color=THEME["CYAN"], alpha=0.55,
              edgecolor="none", density=True)
    lam_grid = np.linspace(lam_minus * 0.95, lam_plus * 1.05, 400)
    ax_h.plot(lam_grid, mp_density(lam_grid, Q),
              color=THEME["YELLOW"], linewidth=1.8, alpha=0.95)
    ax_h.axvline(lam_plus,  color=THEME["YELLOW"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax_h.axvline(lam_minus, color=THEME["YELLOW"], linewidth=0.8, linestyle="--", alpha=0.7)
    if E[cur, -1] < lam_plus * 2.5:
        ax_h.axvline(E[cur, -1], color=THEME["RED"], linewidth=1.5, alpha=0.9)
    ax_h.set_xlim(0, lam_plus * 2.5); ax_h.set_yticks([])
    ax_h.tick_params(colors=THEME["TEXT_DIM"], labelsize=9)
    for s in ax_h.spines.values(): s.set_color("#1f1f1f"); s.set_linewidth(0.5)
    ax_h.set_xlabel("eigenvalue λ", color=THEME["TEXT_DIM"], fontsize=10)
    ax_h.set_title("current window  ·  yellow = MP theory",
                   color=THEME["TEXT_DIM"], fontsize=10, loc="left", pad=4)

    # ---------- RIGHT HUD ----------
    cur_date = pd.Timestamp(dates_t[cur]).strftime("%Y-%m-%d")
    n_signal = int((E[cur] > lam_plus).sum())

    fig.text(0.67, 0.45, "window end", fontsize=11, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.41, cur_date, fontsize=14, color=THEME["TEXT"], fontweight="bold")
    fig.text(0.67, 0.375, f"Q = {Q:.2f}   N = {N}",
             fontsize=11, color=THEME["TEXT_DIM"])

    fig.text(0.67, 0.30, "market eigenvalue", fontsize=11, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.235, f"{E[cur, -1]:5.2f}",
             fontsize=42, color=THEME["RED"], fontweight="bold")

    fig.text(0.67, 0.18, "above bulk", fontsize=11, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.135, f"{n_signal} / {N}",
             fontsize=28, color=THEME["ORANGE"], fontweight="bold")
    fig.text(0.67, 0.110, "eigenvalues outside  [λ-, λ+]",
             fontsize=10, color=THEME["TEXT_DIM"])

    fig.text(0.025, 0.025,
             rf"$\lambda_{{\pm}} = (1 \pm 1/\sqrt{{Q}})^{{2}}$"
             rf"     $\lambda_+ = {lam_plus:.3f}$"
             rf"     $\lambda_- = {lam_minus:.3f}$",
             fontsize=12, color=THEME["YELLOW"])
    fig.text(0.975, 0.025, "@quant.traderr",
             ha="right", fontsize=11, color=THEME["TEXT_DIM"],
             alpha=0.65, family=THEME["FONT"])

    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Saved: {out_path}")


def main():
    t0 = time.time()
    log("=== MARCHENKO-PASTUR STATIC ===")
    data = load_or_compute()
    render_static(data, CONFIG["FOCUS_FRAC"], CONFIG["OUTPUT_FILE"])
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
