"""
Optimal_Stopping_Pipeline.py
============================
Reel pipeline — Snell envelope & the early-exercise boundary for an American put.

Hook: "Most traders see when to exit. We see the boundary that decides for them."

This is the *parent theory* of the Longstaff-Schwartz reel: instead of regressing
continuation values on basis functions, we solve the dynamic programming
recursion exactly on a fine (S, t) grid, giving us the Snell envelope V*(S, t)
and the continuation-vs-exercise region as a 3D surface.

Pipeline:
  1. Build a (S, t) grid under risk-neutral GBM.
  2. Backward induction:
        V(S, T) = payoff(S)
        V(S, t) = max( payoff(S), e^{-r dt} E[V(S', t+dt) | S] )
     E[.] computed exactly via log-normal transition density on the grid
     (vectorized matrix-vector multiply).
  3. Extract early-exercise boundary  S*(t)  =  sup{ S : V(S,t) = payoff(S) }.
  4. Render:
       Left   : 3D Snell envelope V*(S, t)  with stopping region shaded
       Right  : early-exercise boundary  S*(t)  +  payoff cross-section
                +  continuation-region heatmap

Dependencies: numpy, matplotlib
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
    "OUTPUT_IMAGE": "Optimal_Stopping_Output.png",
    "OUTPUT_VIDEO": "Optimal_Stopping_Output.mp4",
    "FRAME_DIR":    "temp_stopping_frames",
    "LOG_FILE":     "stopping_pipeline.log",
    "FPS":          30,
    "N_JOBS":       6,
}

OPT = {
    "S0":     100.0,
    "K":      100.0,
    "r":      0.05,
    "sigma":  0.30,
    "T":      1.0,
    "n_t":    140,         # time grid
    "n_S":    260,         # spot grid
    "S_min":  20.0,
    "S_max":  220.0,
    "seed":   3,
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
# MODULE 1 — TRANSITION KERNEL
# ═══════════════════════════════════════════════════════════════════

def build_transition_matrix(S_grid, dt):
    """
    Discrete log-normal transition probabilities  P[i, j] = P(S_{t+dt}=S_j | S_t=S_i).
    Computed on the *log* grid; rows renormalized to sum to 1 to handle truncation.
    """
    p = OPT
    logS = np.log(S_grid)
    mu = (p["r"] - 0.5 * p["sigma"]**2) * dt
    s  = p["sigma"] * np.sqrt(dt)

    # mean and std for each starting node
    mean = logS[:, None] + mu                       # (n_S, 1)
    diff = (logS[None, :] - mean) / s               # (n_S, n_S)
    pdf  = np.exp(-0.5 * diff**2) / (s * np.sqrt(2*np.pi))
    # convert density to probability mass via log-grid spacing
    dlogS = np.gradient(logS)
    P = pdf * dlogS[None, :]
    # renormalize rows
    P /= P.sum(axis=1, keepdims=True)
    return P


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — BACKWARD INDUCTION (Snell envelope)
# ═══════════════════════════════════════════════════════════════════

def solve_snell_envelope():
    p = OPT
    S = np.linspace(p["S_min"], p["S_max"], p["n_S"])
    t = np.linspace(0.0, p["T"], p["n_t"])
    dt = t[1] - t[0]

    payoff = np.maximum(p["K"] - S, 0.0)            # American put
    P = build_transition_matrix(S, dt)
    disc = np.exp(-p["r"] * dt)

    V = np.zeros((p["n_t"], p["n_S"]))
    V[-1] = payoff
    stop = np.zeros((p["n_t"], p["n_S"]), dtype=bool)
    stop[-1] = payoff > 0

    for k in range(p["n_t"] - 2, -1, -1):
        cont = disc * (P @ V[k+1])
        V[k] = np.maximum(payoff, cont)
        stop[k] = payoff >= cont - 1e-10

    # early-exercise boundary  S*(t) = max{S : in stop region AND payoff>0}
    boundary = np.full(p["n_t"], np.nan)
    for k in range(p["n_t"]):
        idx = np.where(stop[k] & (payoff > 0))[0]
        if len(idx) > 0:
            boundary[k] = S[idx.max()]

    return S, t, V, stop, boundary, payoff


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — STATIC RENDER
# ═══════════════════════════════════════════════════════════════════

def render_static(S, t, V, stop, boundary, payoff, out_path):
    res = CONFIG["RESOLUTION"]
    fig = plt.figure(figsize=(res[0]/100, res[1]/100), dpi=100,
                     facecolor=THEME["BG"])
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.4, 1.0],
                           hspace=0.45, wspace=0.18,
                           left=0.04, right=0.97, top=0.92, bottom=0.07)

    # ─── LEFT: 3D Snell envelope V*(S, t) ──
    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor=THEME["PANEL_BG"])
    Tg, Sg = np.meshgrid(t, S, indexing="ij")
    surf = ax3d.plot_surface(Tg, Sg, V, cmap="magma", alpha=0.92,
                             linewidth=0, antialiased=True,
                             rcount=80, ccount=80)
    # boundary curve riding on the surface
    Vb = np.array([np.interp(b, S, V[k]) if not np.isnan(b) else np.nan
                   for k, b in enumerate(boundary)])
    ax3d.plot(t, boundary, Vb, color=THEME["CYAN"], lw=2.4, label="S*(t)  boundary")
    ax3d.set_xlabel("t", color=THEME["TEXT_DIM"])
    ax3d.set_ylabel("S", color=THEME["TEXT_DIM"])
    ax3d.set_zlabel("V*(S, t)", color=THEME["TEXT_DIM"])
    ax3d.set_title("Snell envelope   V*(S, t) = sup_τ E[ e^{-rτ} (K-S_τ)^+ ]",
                   color=THEME["TEXT"], fontsize=12, pad=14)
    ax3d.tick_params(colors=THEME["TEXT_DIM"])
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor(THEME["PANEL_BG"])
        pane.set_edgecolor(THEME["GRID"])
    ax3d.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                labelcolor=THEME["TEXT_DIM"], fontsize=9)

    # ─── RIGHT row 0: stopping/continuation heatmap ──
    ax_h = fig.add_subplot(gs[0, 1], facecolor=THEME["PANEL_BG"])
    ax_h.imshow(stop.T, aspect="auto", origin="lower", cmap="bwr",
                extent=[t[0], t[-1], S[0], S[-1]], alpha=0.55)
    ax_h.plot(t, boundary, color=THEME["YELLOW"], lw=2.0, label="S*(t)")
    ax_h.set_title("Continuation (blue)  vs  Stopping (red)",
                   color=THEME["TEXT"], fontsize=11, loc="left")
    ax_h.set_xlabel("t", color=THEME["TEXT_DIM"])
    ax_h.set_ylabel("S", color=THEME["TEXT_DIM"])
    ax_h.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    for s in ax_h.spines.values(): s.set_color(THEME["SPINE"])
    ax_h.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                labelcolor=THEME["TEXT_DIM"], fontsize=8)

    # ─── RIGHT row 1: boundary S*(t) ──
    ax_b = fig.add_subplot(gs[1, 1], facecolor=THEME["PANEL_BG"])
    ax_b.plot(t, boundary, color=THEME["CYAN"], lw=1.8, label="S*(t)")
    ax_b.axhline(OPT["K"], color=THEME["TEXT_DIM"], lw=0.8, ls="--",
                 label=f"K = {OPT['K']:.0f}")
    ax_b.set_title("Early-exercise boundary  S*(t)", color=THEME["TEXT"],
                   fontsize=11, loc="left")
    ax_b.set_xlabel("t", color=THEME["TEXT_DIM"])
    ax_b.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    ax_b.grid(True, color=THEME["GRID"], lw=0.4, alpha=0.5)
    for s in ax_b.spines.values(): s.set_color(THEME["SPINE"])
    ax_b.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                labelcolor=THEME["TEXT_DIM"], fontsize=8)

    # ─── RIGHT row 2: V(S, t=0) vs payoff ──
    ax_v = fig.add_subplot(gs[2, 1], facecolor=THEME["PANEL_BG"])
    ax_v.plot(S, V[0], color=THEME["GREEN"], lw=1.6, label="V*(S, 0)")
    ax_v.plot(S, payoff, color=THEME["ORANGE"], lw=1.2, ls="--",
              label="(K - S)^+")
    ax_v.axvline(boundary[0], color=THEME["CYAN"], lw=1.0, ls=":",
                 label=f"S*(0) ≈ {boundary[0]:.1f}")
    ax_v.set_title("Value function at t = 0", color=THEME["TEXT"],
                   fontsize=11, loc="left")
    ax_v.set_xlabel("S", color=THEME["TEXT_DIM"])
    ax_v.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    ax_v.grid(True, color=THEME["GRID"], lw=0.4, alpha=0.5)
    for s in ax_v.spines.values(): s.set_color(THEME["SPINE"])
    ax_v.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                labelcolor=THEME["TEXT_DIM"], fontsize=8)

    fig.suptitle("Optimal Stopping   ·   Snell envelope of an American put",
                 color=THEME["TEXT"], fontsize=18, y=0.975)
    fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log("Optimal Stopping pipeline — start")
    S, t, V, stop, boundary, payoff = solve_snell_envelope()
    log(f"Grid: n_S={len(S)}  n_t={len(t)}   "
        f"S*(0)≈{boundary[0]:.2f}   V*(S0,0)={np.interp(OPT['S0'], S, V[0]):.4f}")
    render_static(S, t, V, stop, boundary, payoff, CONFIG["OUTPUT_IMAGE"])
    log(f"Wrote static frame  →  {CONFIG['OUTPUT_IMAGE']}")
    log("Done.  (Video render: TODO — sweep camera azimuth + reveal boundary "
        "backward from t=T, save to FRAME_DIR, stitch via ffmpeg.)")


if __name__ == "__main__":
    main()
