"""
Bachelier_Static_Pipeline.py
==============================
Single 1920x1080 hero image: Bachelier (1900) vs Black-Scholes GBM,
with the 2020 negative-oil pin as the anchor.

Layout (landscape):
    Top  : title strip
    Left : 3D scene -- GBM ribbon above the red zero-plane, Bachelier
           ribbon crossing through it.  Below-zero segments highlighted
           in yellow.
    Right: terminal density panel (both distributions overlaid),
           HUD (counters), equations, oil callout.
"""

import os, sys, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
from Bachelier_Reel_Pipeline import THEME, simulate, _hex_rgba  # noqa: E402


CONFIG = {
    "WIDTH":  1920, "HEIGHT": 1080, "DPI": 100,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "Bachelier_Static.png"),
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def render_static(t_axis, gbm, bch, out_path):
    n_s = len(t_axis) - 1
    n_p = gbm.shape[1]

    fig = plt.figure(figsize=(CONFIG["WIDTH"]/CONFIG["DPI"],
                              CONFIG["HEIGHT"]/CONFIG["DPI"]),
                     dpi=CONFIG["DPI"], facecolor=THEME["BG"])

    # ---------- TITLE ----------
    fig.text(0.025, 0.945, "BACHELIER  vs  BLACK-SCHOLES",
             fontsize=22, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.025, 0.910,
             "1900 arithmetic Brownian motion still works when 1973 lognormal breaks",
             fontsize=12, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.975, 0.945, "WTI · 2020-04-20", ha="right",
             fontsize=11, color=THEME["RED"], family=THEME["FONT"])

    # ---------- 3D HERO (LEFT) ----------
    ax3d = fig.add_axes([-0.02, 0.05, 0.66, 0.85],
                        projection="3d", facecolor=THEME["BG"])

    # ZERO plane (red)
    tt = np.array([[t_axis[0], t_axis[-1]], [t_axis[0], t_axis[-1]]])
    yy = np.array([[-0.6, -0.6], [0.6, 0.6]])
    ax3d.plot_surface(tt, yy, np.zeros_like(tt), color=THEME["RED"],
                      alpha=0.12, linewidth=0, antialiased=False, shade=False)

    # S0 reference plane (faint cyan)
    ax3d.plot_surface(tt, yy, np.full_like(tt, 50.0),
                      color=THEME["CYAN"], alpha=0.05, linewidth=0,
                      antialiased=False, shade=False)

    def bundle(y_off, paths, base_hex, base_alpha=0.32, fall_alpha=0.85):
        for k in range(n_p):
            p = paths[:, k]
            y = np.full_like(t_axis, y_off)
            pts = np.array([t_axis, y, p]).T.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            neg = (p[:-1] < 0) | (p[1:] < 0)
            cols = np.tile(np.array(_hex_rgba(base_hex, base_alpha)),
                           (len(segs), 1))
            if neg.any():
                cols[neg] = _hex_rgba(THEME["YELLOW"], fall_alpha)
            ax3d.add_collection3d(
                Line3DCollection(segs, colors=cols, linewidths=0.55))

    bundle(+0.30, gbm, THEME["CYAN"], base_alpha=0.30)
    bundle(-0.30, bch, THEME["ORANGE"], base_alpha=0.32, fall_alpha=0.90)

    # Floating labels
    p_max = max(gbm.max(), bch.max()) * 1.05
    p_min = min(bch.min() * 1.05, -10)
    ax3d.text(t_axis[-1] * 0.03, +0.30, p_max * 0.94,
              "GBM  (Black-Scholes)", color=THEME["CYAN"], fontsize=11)
    ax3d.text(t_axis[-1] * 0.03, -0.30, p_max * 0.94,
              "Bachelier  (1900)",   color=THEME["ORANGE"], fontsize=11)
    ax3d.text(t_axis[-1] * 0.55, 0.0, 1.5,
              "ZERO", color=THEME["RED"], fontsize=10, alpha=0.8)

    ax3d.view_init(elev=14, azim=-58)
    ax3d.set_xlim(t_axis[0], t_axis[-1])
    ax3d.set_ylim(-0.7, 0.7)
    ax3d.set_zlim(p_min, p_max)
    ax3d.set_box_aspect((1.7, 0.55, 1.5))
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_alpha(0); pane.set_edgecolor((0, 0, 0, 0))
    ax3d.grid(False)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

    # ---------- RIGHT: terminal density panel ----------
    ax_d = fig.add_axes([0.66, 0.55, 0.32, 0.30], facecolor=THEME["PANEL"])
    g_t = gbm[-1]; b_t = bch[-1]
    p_lo = min(b_t.min(), -25); p_hi = max(g_t.max(), b_t.max())
    bins = np.linspace(p_lo, p_hi, 60)
    ax_d.hist(g_t, bins=bins, color=THEME["CYAN"],
              alpha=0.55, edgecolor="none", density=True, label="GBM")
    ax_d.hist(b_t, bins=bins, color=THEME["ORANGE"],
              alpha=0.55, edgecolor="none", density=True, label="Bachelier")
    ax_d.axvline(0, color=THEME["RED"], linewidth=1.0, alpha=0.8)
    ax_d.set_yticks([])
    ax_d.tick_params(colors=THEME["TEXT_DIM"], labelsize=9)
    for s in ax_d.spines.values(): s.set_color("#1f1f1f"); s.set_linewidth(0.5)
    ax_d.set_xlabel("price at T", color=THEME["TEXT_DIM"], fontsize=10, labelpad=2)
    ax_d.set_title("terminal density", color=THEME["TEXT_DIM"],
                   fontsize=10, loc="left", pad=4)
    ax_d.legend(loc="upper right", fontsize=9, frameon=False,
                labelcolor=THEME["TEXT"])

    # ---------- RIGHT HUD ----------
    bch_below = int((bch < 0).any(axis=0).sum())
    bch_min   = bch.min()
    gbm_min   = gbm.min()

    fig.text(0.67, 0.45, "Bachelier paths below 0",
             fontsize=11, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.395, f"{bch_below} / {n_p}",
             fontsize=36, color=THEME["YELLOW"], fontweight="bold")

    fig.text(0.67, 0.32, "GBM min   |   Bachelier min",
             fontsize=10, color=THEME["TEXT_DIM"])
    fig.text(0.67, 0.272,
             f"${gbm_min:6.2f}     |     ${bch_min:6.2f}",
             fontsize=18, color=THEME["TEXT"], fontweight="bold")

    fig.text(0.67, 0.21,
             r"$dS_{t}^{\,GBM}  = \mu S\, dt + \sigma S\, dW$",
             fontsize=14, color=THEME["CYAN"])
    fig.text(0.67, 0.165,
             r"$dS_{t}^{\,BCH}  = \mu\, dt + \sigma_{B}\, dW$",
             fontsize=14, color=THEME["ORANGE"])

    fig.text(0.025, 0.04,
             "WTI Crude · 2020-04-20 · −$37.63",
             fontsize=14, fontweight="bold", color=THEME["RED"])
    fig.text(0.025, 0.015,
             "lognormal cannot model that.  arithmetic can.",
             fontsize=10, color=THEME["TEXT_DIM"], style="italic")
    fig.text(0.975, 0.020, "@quant.traderr",
             ha="right", fontsize=11, color=THEME["TEXT_DIM"],
             alpha=0.65, family=THEME["FONT"])

    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Saved: {out_path}")


def main():
    t0 = time.time()
    log("=== BACHELIER STATIC ===")
    t_axis, gbm, bch = simulate()
    log(f"GBM   terminal: mean={gbm[-1].mean():.2f}  min={gbm[-1].min():.2f}")
    log(f"BCH   terminal: mean={bch[-1].mean():.2f}  min={bch[-1].min():.2f}")
    render_static(t_axis, gbm, bch, CONFIG["OUTPUT_FILE"])
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
