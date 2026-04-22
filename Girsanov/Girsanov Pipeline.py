"""
Girsanov_Pipeline.py
=====================
Project: Quant Trader Lab - Girsanov / Change of Measure
Author: quant.traderr (Instagram)
License: MIT

Description:
    Dark static visualization of Girsanov's theorem. Shows
    the SAME Brownian path reinterpreted under two measures — the
    real-world measure P (drift mu) and the risk-neutral measure Q
    (drift r). The paths share wiggle structure but diverge in
    trajectory because the DRIFT is reweighted.

    Panel 1 (top)    - Price path ensembles under P and Q, shared Brownian.
                       Pink = real-world drift mu. Green = risk-neutral drift r.
                       The gap is the change-of-measure effect.

    Panel 2 (bottom) - Terminal distributions under P and Q with the
                       option strike K marked. The "in-the-money probability"
                       is different under each measure. This is why
                       Black-Scholes does not give you the real-world
                       probability of exercise.

    Output: Girsanov_Output.png (1920x1080, pure black background)

Dependencies:
    pip install numpy matplotlib

Usage:
    python Girsanov_Pipeline.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects


# --- CONFIGURATION ---

CONFIG = {
    "S0":             100.0,
    "MU":             0.15,       # real-world drift
    "R":              0.05,       # risk-free rate
    "SIGMA":          0.25,       # volatility
    "T":              1.0,
    "N_STEPS":        252,
    "N_PATHS":        140,        # per measure, for the path cloud
    "STRIKE":         110.0,
    "SEED":           3,
    "RESOLUTION":     (1920, 1080),
    "DPI":            170,
    "OUTPUT_IMAGE":   "Girsanov_Output.png",
}

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#050505",
    "GRID":       "#1f1f1f",
    "EDGE":       "#2a2a2a",
    "TEXT":       "#ffffff",
    "TEXT_SEC":   "#cccccc",
    "TEXT_MUTED": "#777777",
    "P_COLOR":    "#e8b0c8",      # pink — real-world drift
    "Q_COLOR":    "#a0d8a0",      # green — risk-neutral drift
    "STRIKE":     "#f0c8a0",      # amber — strike K
    "RN":         "#a0c8e8",      # blue — Radon-Nikodym density
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --- MODULE 1: SIMULATION ---

def simulate_paths():
    """
    Generate N_PATHS shared Brownian paths. Same noise is reinterpreted
    under P (drift mu) and under Q (drift r). This is the geometric
    statement of Girsanov: one underlying Brownian, two drifts.
    """
    rng = np.random.default_rng(CONFIG["SEED"])
    N = CONFIG["N_STEPS"]
    n_paths = CONFIG["N_PATHS"]
    T = CONFIG["T"]
    dt = T / N
    sigma = CONFIG["SIGMA"]
    S0 = CONFIG["S0"]
    mu = CONFIG["MU"]
    r = CONFIG["R"]

    t = np.linspace(0.0, T, N + 1)
    # Shared Brownian increments across all paths
    dW = rng.standard_normal((n_paths, N)) * np.sqrt(dt)
    W = np.concatenate([np.zeros((n_paths, 1)),
                        np.cumsum(dW, axis=1)], axis=1)

    # Under P: S_t = S0 * exp((mu - 0.5 sigma^2) t + sigma W_t)
    drift_p = (mu - 0.5 * sigma**2) * t[None, :]
    paths_P = S0 * np.exp(drift_p + sigma * W)

    # Under Q: S_t = S0 * exp((r - 0.5 sigma^2) t + sigma W_t)
    drift_q = (r - 0.5 * sigma**2) * t[None, :]
    paths_Q = S0 * np.exp(drift_q + sigma * W)

    # Terminal distributions for richer histogram
    big_rng = np.random.default_rng(CONFIG["SEED"] + 1)
    n_big = 80_000
    W_T_big = big_rng.standard_normal(n_big) * np.sqrt(T)
    S_T_P = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T_big)
    S_T_Q = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T_big)

    # Girsanov drift-shift parameter
    theta = (mu - r) / sigma

    return {
        "t":      t,
        "P":      paths_P,
        "Q":      paths_Q,
        "S_T_P":  S_T_P,
        "S_T_Q":  S_T_Q,
        "theta":  theta,
    }


# --- MODULE 2: VISUAL ---

def _style_axes(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for spine in ax.spines.values():
        spine.set_color(THEME["EDGE"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["TEXT_MUTED"], labelsize=10)
    ax.grid(True, color=THEME["GRID"], linewidth=0.6, alpha=0.9)


def _glow(color):
    return [patheffects.Stroke(linewidth=4, foreground=color, alpha=0.14),
            patheffects.Normal()]


def make_figure(sim):
    plt.rcParams.update({
        "figure.facecolor":  THEME["BG"],
        "savefig.facecolor": THEME["BG"],
        "text.color":        THEME["TEXT"],
        "font.family":       "DejaVu Sans",
    })

    w_in = CONFIG["RESOLUTION"][0] / CONFIG["DPI"]
    h_in = CONFIG["RESOLUTION"][1] / CONFIG["DPI"]
    fig = plt.figure(figsize=(w_in, h_in), facecolor=THEME["BG"])

    # Title band
    fig.text(0.5, 0.945,
             "GIRSANOV THEOREM  //  CHANGE OF MEASURE",
             ha="center", va="center",
             color=THEME["TEXT"], fontsize=22, fontweight="bold")
    fig.text(0.5, 0.905,
             f"Same Brownian path, two drifts   |   "
             f"μ = {CONFIG['MU']:.2f}   r = {CONFIG['R']:.2f}   "
             f"σ = {CONFIG['SIGMA']:.2f}   θ = {sim['theta']:.2f}",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=12)

    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.07, right=0.97, top=0.85, bottom=0.10,
        wspace=0.22,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax1)
    _style_axes(ax2)

    # ---- Panel 1: Path ensembles under P and Q ----
    t = sim["t"]
    paths_P = sim["P"]
    paths_Q = sim["Q"]
    n_show = paths_P.shape[0]

    for i in range(n_show):
        ax1.plot(t, paths_P[i], color=THEME["P_COLOR"],
                 linewidth=0.5, alpha=0.35)
        ax1.plot(t, paths_Q[i], color=THEME["Q_COLOR"],
                 linewidth=0.5, alpha=0.35)

    # Expected curves
    S0 = CONFIG["S0"]
    ax1.plot(t, S0 * np.exp(CONFIG["MU"] * t),
             color=THEME["P_COLOR"], linewidth=2.2,
             label=fr"Real-world  $\mathbb{{E}}_P[S_t] = S_0 e^{{\mu t}}$")
    ax1.plot(t, S0 * np.exp(CONFIG["R"] * t),
             color=THEME["Q_COLOR"], linewidth=2.2,
             label=fr"Risk-neutral  $\mathbb{{E}}_Q[S_t] = S_0 e^{{r t}}$")

    # Strike line
    ax1.axhline(CONFIG["STRIKE"], color=THEME["STRIKE"],
                linestyle="--", linewidth=1.2,
                label=f"Strike  K = {CONFIG['STRIKE']:.0f}")

    ax1.set_title("Same Brownian path, reinterpreted",
                  color=THEME["TEXT_SEC"], fontsize=13,
                  pad=12, loc="left")
    ax1.set_xlabel("Time  t", color=THEME["TEXT_MUTED"])
    ax1.set_ylabel("Price  S_t", color=THEME["TEXT_MUTED"])
    leg = ax1.legend(loc="upper left",
                     facecolor=THEME["PANEL_BG"],
                     edgecolor=THEME["EDGE"],
                     fontsize=10, labelcolor=THEME["TEXT_SEC"])
    for txt in leg.get_texts():
        txt.set_color(THEME["TEXT_SEC"])

    # ---- Panel 2: Terminal distributions + strike ----
    S_T_P = sim["S_T_P"]
    S_T_Q = sim["S_T_Q"]
    K = CONFIG["STRIKE"]

    x_lo = min(S_T_P.min(), S_T_Q.min())
    x_hi = max(S_T_P.max(), S_T_Q.max())
    bins = np.linspace(x_lo, x_hi, 120)

    ax2.hist(S_T_P, bins=bins, color=THEME["P_COLOR"],
             alpha=0.35, density=True, edgecolor=THEME["P_COLOR"],
             linewidth=0.4, label=r"$P$ — real world")
    ax2.hist(S_T_Q, bins=bins, color=THEME["Q_COLOR"],
             alpha=0.35, density=True, edgecolor=THEME["Q_COLOR"],
             linewidth=0.4, label=r"$Q$ — risk neutral")

    ax2.axvline(K, color=THEME["STRIKE"],
                linestyle="--", linewidth=1.4,
                label=f"Strike  K = {K:.0f}")

    # Shade ITM regions for illustration
    mask_p = S_T_P > K
    mask_q = S_T_Q > K
    p_itm = mask_p.mean()
    q_itm = mask_q.mean()

    ax2.set_title(
        f"Terminal distributions  |  "
        f"P(S_T > K) = {p_itm:.3f} under P   "
        f"{q_itm:.3f} under Q",
        color=THEME["TEXT_SEC"], fontsize=13, pad=12, loc="left",
    )
    ax2.set_xlabel("Terminal price  $S_T$", color=THEME["TEXT_MUTED"])
    ax2.set_ylabel("Density", color=THEME["TEXT_MUTED"])
    leg2 = ax2.legend(loc="upper right",
                      facecolor=THEME["PANEL_BG"],
                      edgecolor=THEME["EDGE"],
                      fontsize=10, labelcolor=THEME["TEXT_SEC"])
    for txt in leg2.get_texts():
        txt.set_color(THEME["TEXT_SEC"])

    # Footer
    fig.text(0.015, 0.025, "@quant.traderr",
             color=THEME["TEXT_MUTED"], fontsize=11)
    fig.text(0.985, 0.025,
             r"$\frac{dQ}{dP} = \exp(-\frac{1}{2}\theta^2 T - \theta W_T)$",
             color=THEME["TEXT_MUTED"], fontsize=13, ha="right")

    return fig


# --- MAIN ---

def main():
    t_start = time.time()
    log("=" * 60)
    log("Girsanov_Pipeline - Bloomberg Dark Static")
    log("=" * 60)

    sim = simulate_paths()
    log(f"  mu = {CONFIG['MU']}  r = {CONFIG['R']}  sigma = {CONFIG['SIGMA']}")
    log(f"  theta = (mu - r) / sigma = {sim['theta']:.4f}")
    log(f"  N_paths = {CONFIG['N_PATHS']}  T = {CONFIG['T']}")

    p_itm = (sim["S_T_P"] > CONFIG["STRIKE"]).mean()
    q_itm = (sim["S_T_Q"] > CONFIG["STRIKE"]).mean()
    log(f"  P(S_T > K) under P = {p_itm:.4f}")
    log(f"  P(S_T > K) under Q = {q_itm:.4f}")
    log(f"  Girsanov shift: P drift {p_itm:.3f}, Q drift {q_itm:.3f}")

    fig = make_figure(sim)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"],
                facecolor=THEME["BG"], bbox_inches=None)
    plt.close(fig)

    log(f"Saved: {out_path}")
    log(f"Time:  {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
