"""
Itos_Lemma_Pipeline.py
=======================
Project: Quant Trader Lab - Itô's Lemma Visualization
Author: quant.traderr (Instagram)
License: MIT

Description:
    Bloomberg Dark static visualization of Itô's Lemma. Two-panel
    cinematic layout that makes the -sigma^2/2 correction and the
    dW^2 = dt identity visible at a glance. Ships at 1920x1080 for
    standalone social-media use (reel thumbnail, cross-posts, archive).

    Panel 1 (left) - Two interpretations of the SAME Brownian path:
        Naive:         S_t = S_0 * exp(mu*t + sigma*W_t)
        Ito-corrected: S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*W_t)
        The visible gap is the multiplicative correction exp(sigma^2 t / 2).
        That gap is Ito's Lemma.

    Panel 2 (right) - Quadratic variation of the same Brownian path:
        sum_k (Delta W_k)^2  converges a.s. to  t
        This is the identity (dW)^2 = dt that breaks classical calculus.

    Pipeline Steps:
        1. MATH       - Generate a single shared Brownian path.
        2. PATHS      - Compute naive and Ito-corrected GBM paths.
        3. QUADVAR    - Accumulate the quadratic variation.
        4. VISUAL     - Two-panel matplotlib figure on pure black.

Dependencies:
    pip install numpy matplotlib

Usage:
    python Itos_Lemma_Pipeline.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects


# --- CONFIGURATION ---

CONFIG = {
    "T":               1.0,
    "N":               2500,
    "MU":              0.10,
    "SIGMA":           0.40,
    "S0":              100.0,
    "SEED":            3,
    "RESOLUTION":      (1920, 1080),
    "DPI":             170,
    "OUTPUT_IMAGE":    "Itos_Lemma_Output.png",
}

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#050505",
    "GRID":       "#1f1f1f",
    "EDGE":       "#2a2a2a",
    "TEXT":       "#ffffff",
    "TEXT_SEC":   "#cccccc",
    "TEXT_MUTED": "#777777",
    "NAIVE":      "#e8b0c8",   # pink — the wrong path
    "ITO":        "#a0d8a0",   # green — the corrected path
    "GAP":        "#f0c8a0",   # amber — the Itô correction shaded region
    "QV":         "#a0c8e8",   # blue — cumulative (dW)^2
    "IDENT":      "#f0c8a0",   # amber dashed — y = t identity
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --- MODULE 1: MATH ---

def build_brownian_path(seed):
    rng = np.random.default_rng(seed)
    T, N = CONFIG["T"], CONFIG["N"]
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    dW = rng.standard_normal(N) * np.sqrt(dt)
    W = np.concatenate([[0.0], np.cumsum(dW)])
    return t, dW, W, dt


def paths_from_bm(t, W):
    mu = CONFIG["MU"]
    sigma = CONFIG["SIGMA"]
    S0 = CONFIG["S0"]
    S_naive = S0 * np.exp(mu * t + sigma * W)
    S_ito = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S_naive, S_ito


def quadratic_variation(dW):
    return np.concatenate([[0.0], np.cumsum(dW**2)])


# --- MODULE 2: VISUAL ---

def _style_axes(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for spine in ax.spines.values():
        spine.set_color(THEME["EDGE"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["TEXT_MUTED"], labelsize=10)
    ax.grid(True, color=THEME["GRID"], linewidth=0.6, alpha=0.9)


def _glow(color):
    return [patheffects.Stroke(linewidth=4, foreground=color, alpha=0.18),
            patheffects.Normal()]


def make_figure(t, S_naive, S_ito, QV):
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
             "ITÔ'S LEMMA  //  THE CHAIN RULE FOR BROWNIAN MOTION",
             ha="center", va="center",
             color=THEME["TEXT"], fontsize=22, fontweight="bold",
             family="DejaVu Sans")
    fig.text(0.5, 0.905,
             f"Same Brownian path, two chain rules   |   "
             f"μ = {CONFIG['MU']:.2f}   σ = {CONFIG['SIGMA']:.2f}   "
             f"T = {CONFIG['T']:.1f}   |   N = {CONFIG['N']} steps",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=12,
             family="DejaVu Sans")

    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.07, right=0.97, top=0.85, bottom=0.10,
        wspace=0.22,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax1)
    _style_axes(ax2)

    # ---- Panel 1: Two paths diverging via the Itô correction ----
    line_naive, = ax1.plot(
        t, S_naive, color=THEME["NAIVE"], linewidth=1.6,
        label=r"Naive   $S_t = S_0\,e^{\mu t + \sigma W_t}$",
    )
    line_naive.set_path_effects(_glow(THEME["NAIVE"]))

    line_ito, = ax1.plot(
        t, S_ito, color=THEME["ITO"], linewidth=1.6,
        label=r"Itô   $S_t = S_0\,e^{(\mu - \sigma^2/2)t + \sigma W_t}$",
    )
    line_ito.set_path_effects(_glow(THEME["ITO"]))

    ax1.fill_between(
        t, S_ito, S_naive,
        color=THEME["GAP"], alpha=0.20,
        label=r"Itô correction gap   $e^{\sigma^2 t / 2}$",
    )
    ax1.set_title("Same Brownian path, two chain rules",
                  color=THEME["TEXT_SEC"], fontsize=13, pad=12, loc="left")
    ax1.set_xlabel(r"Time   $t$", color=THEME["TEXT_MUTED"])
    ax1.set_ylabel(r"Price   $S_t$", color=THEME["TEXT_MUTED"])
    leg = ax1.legend(loc="upper left",
                     facecolor=THEME["PANEL_BG"],
                     edgecolor=THEME["EDGE"],
                     fontsize=10, labelcolor=THEME["TEXT_SEC"])
    for txt in leg.get_texts():
        txt.set_color(THEME["TEXT_SEC"])

    # ---- Panel 2: Quadratic variation -> y = t ----
    line_qv, = ax2.plot(
        t, QV, color=THEME["QV"], linewidth=1.6,
        label=r"$\sum_k (\Delta W_k)^2$",
    )
    line_qv.set_path_effects(_glow(THEME["QV"]))

    line_id, = ax2.plot(
        t, t, color=THEME["IDENT"], linewidth=1.4, linestyle="--",
        label=r"$y = t$   (the identity)",
    )
    line_id.set_path_effects(_glow(THEME["IDENT"]))

    ax2.set_title(r"Quadratic variation: $(dW)^2 = dt$",
                  color=THEME["TEXT_SEC"], fontsize=13, pad=12, loc="left")
    ax2.set_xlabel(r"Time   $t$", color=THEME["TEXT_MUTED"])
    ax2.set_ylabel(r"$\sum (\Delta W)^2$", color=THEME["TEXT_MUTED"])
    leg2 = ax2.legend(loc="upper left",
                      facecolor=THEME["PANEL_BG"],
                      edgecolor=THEME["EDGE"],
                      fontsize=10, labelcolor=THEME["TEXT_SEC"])
    for txt in leg2.get_texts():
        txt.set_color(THEME["TEXT_SEC"])

    # Footer
    fig.text(0.015, 0.025, "@quant.traderr",
             color=THEME["TEXT_MUTED"], fontsize=11,
             family="DejaVu Sans")
    fig.text(0.985, 0.025,
             r"$df = (f_t + \mu f_x + \frac{1}{2}\sigma^2 f_{xx})\,dt + \sigma f_x \, dW$",
             color=THEME["TEXT_MUTED"], fontsize=12, ha="right",
             family="DejaVu Sans")

    return fig


# --- MAIN ---

def main():
    t0 = time.time()
    log("=" * 60)
    log("Itos_Lemma_Pipeline - Bloomberg Dark Static")
    log("=" * 60)

    t, dW, W, dt = build_brownian_path(CONFIG["SEED"])
    log(f"  T = {CONFIG['T']}  N = {CONFIG['N']}  dt = {dt:.2e}")
    S_naive, S_ito = paths_from_bm(t, W)
    log(f"  S_naive[T] = {S_naive[-1]:.2f}")
    log(f"  S_ito[T]   = {S_ito[-1]:.2f}")
    log(f"  Itô gap    = {S_naive[-1] / S_ito[-1]:.4f}  "
        f"(theory: {np.exp(0.5 * CONFIG['SIGMA']**2 * CONFIG['T']):.4f})")

    QV = quadratic_variation(dW)
    log(f"  QV[T]      = {QV[-1]:.4f}  (theory: {CONFIG['T']:.4f})")

    fig = make_figure(t, S_naive, S_ito, QV)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"],
                facecolor=THEME["BG"], bbox_inches=None)
    plt.close(fig)

    log(f"Saved: {out_path}")
    log(f"Time:  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
