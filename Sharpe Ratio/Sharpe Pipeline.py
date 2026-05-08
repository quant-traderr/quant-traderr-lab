"""
Sharpe_Video.py
================
Project: Quant Trader Lab - Sharpe Ratio Timelapse
Author: quant.traderr (Instagram)
License: MIT

Description:
    Bloomberg Dark MP4 timelapse of the Sharpe Ratio. Three
    hypothetical fund managers run the same market sequence (shared
    Brownian shocks) at different true Sharpe levels (0.8, 1.8, 3.0).

    Layout (1080 x 1920):
      • Top panel    - Equity curves, log scale
      • Bottom panel - Expanding annualized Sharpe for each track,
                       drifting noisily early and converging to its
                       true value as the track record matures.
      • Header       - SHARPE RATIO title + subtitle
      • Footer       - Sharpe equation + @quant.traderr

    Pedagogical point: Sharpe needs years of data to settle. Early
    track records with headline-grade Sharpe are statistical artifacts
    more often than they are alpha.

    Structure:
      • Output - Static image with settled Sharpe readouts

Dependencies:
    pip install numpy matplotlib

Usage:
    python Sharpe_Video.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects


# --- CONFIGURATION ---

CONFIG = {
    "N_DAYS":       1260,       # ~5 years daily
    "WIDTH":        1080,
    "HEIGHT":       1920,
    "DPI":          120,
    "RF_ANNUAL":    0.04,
    "SEED":         7,
    "MIN_WINDOW":   30,          # no Sharpe estimate below this many days
    "OUTPUT_IMAGE": "sharpe_final.png",
}

# (label, daily_mu, daily_sigma, color) — mu/sigma chosen to land on
# true annualized Sharpe = {0.8, 1.8, 3.0} with rf = 4%/yr.
SCENARIOS = [
    ("Sharpe 0.8",  0.00070, 0.0110, "#a0c8e8"),
    ("Sharpe 1.8",  0.00105, 0.0078, "#a0d8a0"),
    ("Sharpe 3.0",  0.00145, 0.0068, "#f0c8a0"),
]

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#050505",
    "GRID":       "#1f1f1f",
    "EDGE":       "#2a2a2a",
    "TEXT":       "#ffffff",
    "TEXT_SEC":   "#cccccc",
    "TEXT_MUTED": "#777777",
    "TRUE_LINE":  "#555555",
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --- MODULE 1: SIMULATION ---

def simulate():
    """Generate 3 scenarios sharing common Brownian shocks."""
    rng = np.random.default_rng(CONFIG["SEED"])
    n = CONFIG["N_DAYS"]
    rf_daily = CONFIG["RF_ANNUAL"] / 252

    # Shared standardised shocks → all 3 "see" the same market timing
    z = rng.standard_normal(n)

    scenarios = []
    for label, mu, sigma, color in SCENARIOS:
        r = mu + sigma * z
        equity = np.cumprod(1.0 + r)
        true_sharpe = (mu - rf_daily) / sigma * np.sqrt(252)
        # Expanding annualized Sharpe estimate
        exp_sh = _expanding_sharpe(r, rf_daily, CONFIG["MIN_WINDOW"])
        scenarios.append({
            "label":   label,
            "mu":      mu,
            "sigma":   sigma,
            "color":   color,
            "returns": r,
            "equity":  equity,
            "exp_sh":  exp_sh,
            "true":    true_sharpe,
        })
    return scenarios


def _expanding_sharpe(r, rf_daily, min_window):
    """Expanding-window annualized Sharpe estimate across the series."""
    n = len(r)
    out = np.full(n, np.nan)
    csum = np.cumsum(r)
    csum2 = np.cumsum(r * r)
    for i in range(min_window, n + 1):
        m = csum[i - 1] / i
        var = csum2[i - 1] / i - m * m
        if var <= 0:
            continue
        sd = np.sqrt(var * i / (i - 1))  # unbiased
        out[i - 1] = (m - rf_daily) / sd * np.sqrt(252)
    return out


# --- MODULE 2: FIGURE ---

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


def build_figure(scenarios):
    plt.rcParams.update({
        "figure.facecolor":  THEME["BG"],
        "savefig.facecolor": THEME["BG"],
        "text.color":        THEME["TEXT"],
        "font.family":       "DejaVu Sans",
    })

    w_in = CONFIG["WIDTH"] / CONFIG["DPI"]
    h_in = CONFIG["HEIGHT"] / CONFIG["DPI"]
    fig = plt.figure(figsize=(w_in, h_in), facecolor=THEME["BG"])

    fig.text(0.5, 0.955, "SHARPE RATIO",
             ha="center", va="center",
             color=THEME["TEXT"], fontsize=34, fontweight="bold")
    fig.text(0.5, 0.925, "Same market shocks, three skill levels",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=14)
    fig.text(0.5, 0.900,
             f"rf = {CONFIG['RF_ANNUAL']:.0%}   "
             f"N = {CONFIG['N_DAYS']} days   "
             f"expanding Sharpe",
             ha="center", va="center",
             color=THEME["TEXT_MUTED"], fontsize=12)

    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        left=0.11, right=0.94, top=0.87, bottom=0.09,
        hspace=0.28,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    _style_axes(ax1)
    _style_axes(ax2)

    ax1.set_title("Equity curves (log scale)",
                  color=THEME["TEXT_SEC"], fontsize=13,
                  pad=10, loc="left")
    ax1.set_xlabel("Trading day", color=THEME["TEXT_MUTED"])
    ax1.set_ylabel("Capital", color=THEME["TEXT_MUTED"])
    ax1.set_yscale("log")

    ax2.set_title("Expanding annualized Sharpe estimate",
                  color=THEME["TEXT_SEC"], fontsize=13,
                  pad=10, loc="left")
    ax2.set_xlabel("Trading day", color=THEME["TEXT_MUTED"])
    ax2.set_ylabel("Sharpe (annualized)", color=THEME["TEXT_MUTED"])

    # Horizontal reference lines at true Sharpe values
    for s in scenarios:
        ax2.axhline(
            s["true"], color=THEME["TRUE_LINE"],
            linestyle=":", linewidth=0.8, alpha=0.7,
        )

    fig.text(0.025, 0.03, "@quant.traderr",
             color=THEME["TEXT_MUTED"], fontsize=11)
    fig.text(0.975, 0.03,
             r"$S = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p}$",
             color=THEME["TEXT_MUTED"], fontsize=13, ha="right")

    return fig, ax1, ax2


def set_axis_limits(ax1, ax2, scenarios):
    n = CONFIG["N_DAYS"]
    all_equity = np.concatenate([s["equity"] for s in scenarios])
    all_sharpe = np.concatenate([s["exp_sh"][~np.isnan(s["exp_sh"])]
                                  for s in scenarios])
    ax1.set_xlim(0, n)
    ax1.set_ylim(all_equity.min() * 0.9, all_equity.max() * 1.1)
    ax2.set_xlim(0, n)
    # Leave headroom for the wild early swings
    sh_lo = min(all_sharpe.min(), -2.0)
    sh_hi = max(all_sharpe.max(), 5.0)
    ax2.set_ylim(sh_lo - 0.3, sh_hi + 0.3)


# --- MODULE 3: RENDER ---

def render_image(scenarios):
    t_start = time.time()
    log("=" * 60)
    log("Sharpe_Video - Static Image")
    log("=" * 60)

    for s in scenarios:
        log(f"  {s['label']}: mu={s['mu']:.5f} sigma={s['sigma']:.5f} "
            f"true_sharpe={s['true']:.2f}")

    fig, ax1, ax2 = build_figure(scenarios)
    set_axis_limits(ax1, ax2, scenarios)

    n_days = CONFIG["N_DAYS"]
    t_days = np.arange(n_days)

    for s in scenarios:
        el, = ax1.plot(
            t_days, s["equity"], color=s["color"], linewidth=2.0,
            label=f"{s['label']}  (mu={s['mu']:.4f}, sigma={s['sigma']:.4f})",
        )
        el.set_path_effects(_glow(s["color"]))

        sl, = ax2.plot(
            t_days, s["exp_sh"], color=s["color"], linewidth=2.0,
            label=f"{s['label']}  (true = {s['true']:.2f})",
        )
        sl.set_path_effects(_glow(s["color"]))

    k = n_days
    eq_lines = "\n".join(
        f"{s['label']:<12s} {s['equity'][k-1]:6.2f}x"
        for s in scenarios
    )
    sh_vals = []
    for s in scenarios:
        val = s["exp_sh"][k - 1]
        sh_vals.append(
            f"{s['label']:<12s} "
            f"{'--' if np.isnan(val) else f'{val:5.2f}'}"
        )
        
    ax1.text(
        0.98, 0.05, f"day {k:>4d}\n{eq_lines}", transform=ax1.transAxes,
        ha="right", va="bottom",
        color=THEME["TEXT_MUTED"], fontsize=11,
        family="DejaVu Sans Mono",
    )
    ax2.text(
        0.98, 0.05, "\n".join(sh_vals), transform=ax2.transAxes,
        ha="right", va="bottom",
        color=THEME["TEXT_MUTED"], fontsize=11,
        family="DejaVu Sans Mono",
    )

    ax1.legend(
        loc="upper left",
        facecolor=THEME["PANEL_BG"], edgecolor=THEME["EDGE"],
        fontsize=10, labelcolor=THEME["TEXT_SEC"],
    )
    ax2.legend(
        loc="upper left",
        facecolor=THEME["PANEL_BG"], edgecolor=THEME["EDGE"],
        fontsize=10, labelcolor=THEME["TEXT_SEC"],
    )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CONFIG["OUTPUT_IMAGE"],
    )
    log(f"  Writing: {out_path}")
    
    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log(f"Saved: {out_path}  ({size_mb:.2f} MB)")
    log(f"Time:  {time.time() - t_start:.1f}s")


# --- MAIN ---

if __name__ == "__main__":
    scenarios = simulate()
    render_image(scenarios)
