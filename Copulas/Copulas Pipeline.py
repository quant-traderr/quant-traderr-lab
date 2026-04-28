"""
Copulas_Pipeline_v2.py
=======================
Project: Quant Trader Lab - Copulas (Instagram-native static)
Author: quant.traderr (Instagram)
License: MIT

Description:
    Rebuild of the Copulas static optimised for Instagram feed. Portrait
    1080x1350 (4:5), two stacked scatter panels, tail-crash region
    highlighted, bold single-idea readout.

    Panel 1 (top)    - Gaussian copula (rho=0.7) scatter. No asymptotic
                       tail dependence. The red square in the lower-left
                       corner shows how rarely joint crashes occur.

    Panel 2 (bottom) - Student-t copula (nu=3, rho=0.7) scatter. Visible
                       clumping in the lower-left corner. That clumping
                       is tail dependence, the phenomenon Gaussian models
                       miss in real crashes.

    The story: same marginals, same rank correlation, one has crash
    clustering and one does not. If you are pricing a CDO or a
    multi-asset portfolio under Gaussian, you are systematically
    underpricing joint tail risk.

    Output: Copulas_Output_v2.png  (1080x1350 PNG, black background)

Dependencies:
    pip install numpy matplotlib scipy

Usage:
    python Copulas_Pipeline_v2.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.stats import norm, t as t_dist


# --- CONFIGURATION ---

CONFIG = {
    "NU":             3,          # Student-t degrees of freedom
    "RHO":            0.7,        # copula correlation
    "N_POINTS":       6000,       # scatter samples per copula
    "TAIL_Q":         0.10,       # threshold for "joint crash" region
    "RESOLUTION":     (1080, 1350),
    "DPI":            150,
    "SEED":           11,
    "OUTPUT_IMAGE":   "Copulas_Output_v2.png",
}

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#060606",
    "GRID":       "#1a1a1a",
    "EDGE":       "#2a2a2a",
    "TEXT":       "#ffffff",
    "TEXT_SEC":   "#cccccc",
    "TEXT_MUTED": "#888888",
    "GAUSSIAN":   "#a0c8e8",
    "STUDENT_T":  "#e8b0c8",
    "CRASH":      "#ff4060",
    "ACCENT":     "#f0c8a0",
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --- MODULE 1: SAMPLING ---

def sample_copulas():
    """Draw N samples each from Gaussian and Student-t copulas."""
    rng = np.random.default_rng(CONFIG["SEED"])
    n = CONFIG["N_POINTS"]
    rho = CONFIG["RHO"]
    nu = CONFIG["NU"]

    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)

    # Gaussian copula: (Z @ L.T) then transform via Phi
    z1 = rng.standard_normal((n, 2))
    xy_g = z1 @ L.T
    u_g = norm.cdf(xy_g[:, 0])
    v_g = norm.cdf(xy_g[:, 1])

    # Student-t copula: (Z @ L.T) / sqrt(W/nu) then transform via T_nu
    z2 = rng.standard_normal((n, 2))
    xy_norm = z2 @ L.T
    w = rng.chisquare(nu, n)
    xy_t = xy_norm / np.sqrt(w[:, None] / nu)
    u_t = t_dist.cdf(xy_t[:, 0], nu)
    v_t = t_dist.cdf(xy_t[:, 1], nu)

    # Lower tail dependence coefficient for t-copula:
    #   lambda_L = 2 * T_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
    arg = np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    lambda_L_t = 2.0 * t_dist.cdf(-arg, nu + 1)

    return u_g, v_g, u_t, v_t, lambda_L_t


# --- MODULE 2: FIGURE ---

def _style_axes(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for spine in ax.spines.values():
        spine.set_color(THEME["EDGE"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=THEME["TEXT_MUTED"], labelsize=9)
    ax.grid(True, color=THEME["GRID"], linewidth=0.5, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u", color=THEME["TEXT_MUTED"], fontsize=10)
    ax.set_ylabel("v", color=THEME["TEXT_MUTED"], fontsize=10)


def make_figure(u_g, v_g, u_t, v_t, lambda_L_t):
    plt.rcParams.update({
        "figure.facecolor":  THEME["BG"],
        "savefig.facecolor": THEME["BG"],
        "text.color":        THEME["TEXT"],
        "font.family":       "DejaVu Sans",
    })

    w_in = CONFIG["RESOLUTION"][0] / CONFIG["DPI"]
    h_in = CONFIG["RESOLUTION"][1] / CONFIG["DPI"]
    fig = plt.figure(figsize=(w_in, h_in), facecolor=THEME["BG"])

    # ---- Title band ----
    fig.text(0.5, 0.965, "COPULAS",
             ha="center", va="center",
             color=THEME["TEXT"], fontsize=32, fontweight="bold")
    fig.text(0.5, 0.940, "HIDDEN CRASH RISK",
             ha="center", va="center",
             color=THEME["ACCENT"], fontsize=15, fontweight="bold")
    fig.text(0.5, 0.913,
             "Same marginals. Same correlation.",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=12)
    fig.text(0.5, 0.895,
             "Only one clumps crashes together.",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=12)

    # ---- Panels ----
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        left=0.12, right=0.92, top=0.855, bottom=0.075,
        hspace=0.28,
    )
    ax_g = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[1, 0])
    _style_axes(ax_g)
    _style_axes(ax_t)

    q = CONFIG["TAIL_Q"]
    n_total = len(u_g)
    pct_g = 100 * np.mean((u_g < q) & (v_g < q))
    pct_t = 100 * np.mean((u_t < q) & (v_t < q))
    # ratio of joint-crash rates
    ratio = pct_t / pct_g if pct_g > 0 else float("inf")

    # ---- Panel 1: Gaussian copula ----
    ax_g.scatter(u_g, v_g, s=3.0, c=THEME["GAUSSIAN"],
                 alpha=0.55, edgecolors="none")
    ax_g.set_title(
        f"Gaussian copula    (ρ = {CONFIG['RHO']})",
        color=THEME["TEXT_SEC"], fontsize=12, pad=8, loc="left",
    )
    crash_g = patches.Rectangle(
        (0, 0), q, q, linewidth=0.0,
        facecolor=THEME["CRASH"], alpha=0.35,
    )
    ax_g.add_patch(crash_g)
    ax_g.annotate(
        "joint crash\nregion", xy=(q * 1.5, q * 1.5),
        xytext=(0.22, 0.18),
        color=THEME["CRASH"], fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color=THEME["CRASH"], lw=0.8),
    )
    ax_g.text(
        0.97, 0.06,
        f"λ_L = 0.00\n{pct_g:.2f}% joint crashes",
        transform=ax_g.transAxes, ha="right", va="bottom",
        color=THEME["TEXT_SEC"], fontsize=11,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.45",
                  facecolor=THEME["BG"],
                  edgecolor=THEME["EDGE"], alpha=0.88),
    )

    # ---- Panel 2: Student-t copula ----
    ax_t.scatter(u_t, v_t, s=3.0, c=THEME["STUDENT_T"],
                 alpha=0.55, edgecolors="none")
    ax_t.set_title(
        f"Student-t copula    (ν = {CONFIG['NU']}, ρ = {CONFIG['RHO']})",
        color=THEME["TEXT_SEC"], fontsize=12, pad=8, loc="left",
    )
    crash_t = patches.Rectangle(
        (0, 0), q, q, linewidth=0.0,
        facecolor=THEME["CRASH"], alpha=0.35,
    )
    ax_t.add_patch(crash_t)
    ax_t.annotate(
        "joint crash\nregion", xy=(q * 1.5, q * 1.5),
        xytext=(0.22, 0.18),
        color=THEME["CRASH"], fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color=THEME["CRASH"], lw=0.8),
    )
    ax_t.text(
        0.97, 0.06,
        f"λ_L = {lambda_L_t:.2f}\n{pct_t:.2f}% joint crashes",
        transform=ax_t.transAxes, ha="right", va="bottom",
        color=THEME["TEXT_SEC"], fontsize=11,
        family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.45",
                  facecolor=THEME["BG"],
                  edgecolor=THEME["EDGE"], alpha=0.88),
    )

    # ---- Bottom takeaway line ----
    fig.text(
        0.5, 0.050,
        f"Student-t: 45% of extreme co-moves are joint crashes.  Gaussian: 0%.",
        ha="center", va="center",
        color=THEME["ACCENT"], fontsize=12, fontweight="bold",
    )

    # ---- Footer ----
    fig.text(0.03, 0.020, "@quant.traderr",
             color=THEME["TEXT_MUTED"], fontsize=10)
    fig.text(0.97, 0.020,
             r"$\lambda_L = \lim_{q \to 0^+} P(V \leq q \mid U \leq q)$",
             color=THEME["TEXT_MUTED"], fontsize=11, ha="right")

    return fig


# --- MAIN ---

def main():
    t_start = time.time()
    log("=" * 60)
    log("Copulas_Pipeline_v2 - Instagram-Native Portrait")
    log("=" * 60)

    u_g, v_g, u_t, v_t, lambda_L_t = sample_copulas()
    log(f"  nu = {CONFIG['NU']}  rho = {CONFIG['RHO']}")
    log(f"  N = {CONFIG['N_POINTS']} samples each")
    log(f"  lambda_L (Gaussian, theory)  = 0.00")
    log(f"  lambda_L (Student-t, theory) = {lambda_L_t:.4f}")

    q = CONFIG["TAIL_Q"]
    pct_g = 100 * np.mean((u_g < q) & (v_g < q))
    pct_t = 100 * np.mean((u_t < q) & (v_t < q))
    log(f"  Joint tail at q = {q}:")
    log(f"    Gaussian:  {pct_g:.3f}%")
    log(f"    Student-t: {pct_t:.3f}%")
    log(f"    Ratio:     {pct_t / max(pct_g, 1e-9):.2f}x")

    fig = make_figure(u_g, v_g, u_t, v_t, lambda_L_t)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CONFIG["OUTPUT_IMAGE"],
    )
    fig.savefig(out_path, dpi=CONFIG["DPI"],
                facecolor=THEME["BG"], bbox_inches=None)
    plt.close(fig)

    log(f"Saved: {out_path}")
    log(f"Time:  {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
