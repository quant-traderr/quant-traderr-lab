"""
Fat Tails Pipeline.py
=====================
Project: Quant Trader Lab — Tail Risk Analysis
Author:  quant.traderr (Instagram)
License: MIT

Description:
    Production pipeline that quantifies the failure of the Gaussian
    assumption on real market returns and ships a single static PNG
    suitable for Instagram, Demiurge title slides, or report covers.

    Pipeline Steps:
    1. Data Acquisition: BTC-USD daily closes via yfinance.
    2. Returns + Tail Estimation:
         - log returns
         - empirical survival of |r|
         - Gaussian baseline at matched sigma
         - Hill estimator alpha_hat across k
         - power-law fit on the upper tail
    3. Visualization: 4-panel "Bloomberg Dark" composition saved as
       fattails_static.png at 2560x1600.

Output:
    fattails_static.png — 2560x1600, dark void background, warm
    accent palette. Drop-in for the Demiurge title slide
    (frame_path) or standalone post.

Dependencies:
    pip install numpy pandas yfinance scipy matplotlib

Performance:
    Single-process pipeline (data + math < 2s). Heavy work (Hill
    sweep, plotting) is vectorised. No joblib needed at this scale.
"""

from __future__ import annotations
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import yfinance as yf

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER":   "BTC-USD",
    "PERIOD":   "max",        # full available history
    "INTERVAL": "1d",

    # Tail analysis
    "K_TAIL_FRAC":  0.05,     # top 5% of |returns| treated as tail
    "HILL_K_MIN":   20,       # smallest k for the Hill stability sweep
    "HILL_K_MAX":   400,      # largest k

    # Output
    "OUT_PATH": Path(__file__).parent / "fattails_static.png",
    "WIDTH":    2560,
    "HEIGHT":   1600,
    "DPI":      170,
}

# Bloomberg-dark palette (hand-tuned for warm contrast against void black)
PAL = {
    "bg":        "#000000",
    "panel":     "#0a0a0a",
    "axis":      "#3a3a3a",
    "grid":      "#181818",
    "text":      "#ffffff",
    "text_dim":  "#9a9a9a",
    "empirical": "#ff8c42",   # warm orange — real / fat-tailed
    "gaussian":  "#5fa8d3",   # steel blue — naive Gaussian
    "power":     "#d4af37",   # muted gold — fitted power law
    "tail_hi":   "#ff5252",   # red — flagged tail events
}

# --- UTILITIES ---

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --- MODULE 1: DATA ---

def fetch_returns() -> pd.Series:
    """Download daily closes and return a clean log-returns series."""
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")
    df = yf.download(
        CONFIG["TICKER"],
        period=CONFIG["PERIOD"],
        interval=CONFIG["INTERVAL"],
        progress=False,
    )
    if df.empty:
        raise RuntimeError("yfinance returned no rows.")

    prices = df["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    log_ret = np.log(prices).diff().dropna()
    log(f"[Data] {len(log_ret)} daily returns "
        f"({log_ret.index[0].date()} -> {log_ret.index[-1].date()}).")
    return log_ret

# --- MODULE 2: TAIL ANALYSIS ---

def hill_estimator(abs_sorted: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index alpha for the top k order stats."""
    tail = abs_sorted[:k]
    if tail[-1] <= 0:
        return np.nan
    return 1.0 / np.log(tail / tail[-1]).mean()


def hill_sweep(abs_returns: np.ndarray,
               k_min: int, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """alpha_hat as a function of k (Hill plot — checks stability)."""
    sorted_desc = np.sort(abs_returns)[::-1]
    ks = np.arange(k_min, min(k_max, len(sorted_desc) - 1) + 1)
    alphas = np.array([hill_estimator(sorted_desc, k) for k in ks])
    return ks, alphas


def power_law_fit(abs_returns: np.ndarray, frac: float
                  ) -> tuple[float, float, float]:
    """Fit P(|X| > x) ~ C x^{-alpha} on the upper `frac` of |returns|.

    Returns (alpha_hat, C_norm, x_min).
    """
    sorted_desc = np.sort(abs_returns)[::-1]
    k = int(len(sorted_desc) * frac)
    alpha = hill_estimator(sorted_desc, k)
    x_min = sorted_desc[k - 1]
    # Normalise so survival at x_min equals empirical k/n.
    n = len(sorted_desc)
    surv_at_xmin = k / n
    C = surv_at_xmin * x_min ** alpha
    log(f"[Tail] k={k}, x_min={x_min:.4f}, alpha_hat={alpha:.3f}")
    return alpha, C, x_min

# --- MODULE 3: VISUALIZATION ---

def style_axes(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor(PAL["panel"])
    for spine in ax.spines.values():
        spine.set_color(PAL["axis"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=PAL["text_dim"], labelsize=10, length=4)
    ax.grid(True, color=PAL["grid"], linewidth=0.6, alpha=0.9)
    if xlabel:
        ax.set_xlabel(xlabel, color=PAL["text_dim"], fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=PAL["text_dim"], fontsize=11)
    if title:
        ax.set_title(title, color=PAL["text"], fontsize=13,
                     pad=10, loc="left", fontweight="bold")


def render(returns: pd.Series) -> None:
    log("[Render] Building 4-panel static...")

    arr = returns.to_numpy()
    abs_ret = np.abs(arr)
    abs_ret = abs_ret[abs_ret > 0]
    sorted_desc = np.sort(abs_ret)[::-1]
    n = len(sorted_desc)
    surv = np.arange(1, n + 1) / n

    sigma = arr.std(ddof=1)
    mu = arr.mean()
    log(f"[Stats] mu={mu:.5f}, sigma={sigma:.5f}, "
        f"max |r| = {abs_ret.max():.4f} ({abs_ret.max()/sigma:.1f} sigma)")

    # Power-law fit on the upper 5%.
    alpha, C, x_min = power_law_fit(abs_ret, CONFIG["K_TAIL_FRAC"])
    x_pl = np.logspace(np.log10(x_min), np.log10(abs_ret.max() * 1.05), 200)
    surv_pl = C * x_pl ** (-alpha)

    # Gaussian baseline survival at matched sigma.
    x_grid = np.logspace(np.log10(abs_ret.min() + 1e-6),
                         np.log10(abs_ret.max() * 1.05), 400)
    surv_gauss = 2.0 * (1.0 - stats.norm.cdf(x_grid, loc=0.0, scale=sigma))

    # Hill sweep.
    ks, alphas = hill_sweep(abs_ret, CONFIG["HILL_K_MIN"],
                            CONFIG["HILL_K_MAX"])

    # Tail flag for time-series panel: |r| > 4 sigma.
    flag = np.abs(arr) > 4.0 * sigma

    # ---- Figure ----
    fig = plt.figure(
        figsize=(CONFIG["WIDTH"] / CONFIG["DPI"],
                 CONFIG["HEIGHT"] / CONFIG["DPI"]),
        facecolor=PAL["bg"],
    )
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.06, right=0.97, top=0.88, bottom=0.09,
        wspace=0.22, hspace=0.36,
    )

    # Top banner.
    fig.text(0.06, 0.945, "FAT TAILS", color=PAL["text"],
             fontsize=22, fontweight="bold", family="DejaVu Sans")
    fig.text(0.06, 0.918,
             f"{CONFIG['TICKER']}  daily log returns,  n = {len(arr):,} days",
             color=PAL["text_dim"], fontsize=11, family="DejaVu Sans")
    fig.text(0.97, 0.945,
             f"alpha = {alpha:.2f}",
             color=PAL["power"], fontsize=22, fontweight="bold",
             ha="right", family="DejaVu Sans")
    fig.text(0.97, 0.918,
             "Hill estimator, top 5% of |returns|",
             color=PAL["text_dim"], fontsize=11, ha="right",
             family="DejaVu Sans")

    # Panel 1: Returns time series with tail events highlighted.
    ax1 = fig.add_subplot(gs[0, 0])
    style_axes(ax1, xlabel="date", ylabel="log return",
               title="Daily returns  (|r| > 4 sigma flagged)")
    ax1.plot(returns.index, arr, color=PAL["empirical"],
             linewidth=0.6, alpha=0.85)
    ax1.scatter(returns.index[flag], arr[flag],
                s=22, color=PAL["tail_hi"], zorder=5,
                edgecolor=PAL["bg"], linewidth=0.4,
                label=f"tail events: {flag.sum()}")
    ax1.axhline(0, color=PAL["axis"], linewidth=0.5)
    for k_sig in (-4, 4):
        ax1.axhline(k_sig * sigma, color=PAL["text_dim"],
                    linewidth=0.5, linestyle=":")
    ax1.legend(loc="lower left", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # Panel 2: Histogram with Gaussian overlay (log y).
    ax2 = fig.add_subplot(gs[0, 1])
    style_axes(ax2, xlabel="log return", ylabel="frequency  (log)",
               title="Return distribution vs Gaussian fit")
    bins = np.linspace(arr.min(), arr.max(), 120)
    ax2.hist(arr, bins=bins, color=PAL["empirical"],
             edgecolor="none", alpha=0.85, label="empirical")
    x_g = np.linspace(arr.min(), arr.max(), 400)
    pdf_g = stats.norm.pdf(x_g, loc=mu, scale=sigma)
    bin_width = bins[1] - bins[0]
    ax2.plot(x_g, pdf_g * len(arr) * bin_width,
             color=PAL["gaussian"], linewidth=1.6,
             label=f"Gaussian, sigma = {sigma:.4f}")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # Panel 3: Log-log tail survival, three curves.
    ax3 = fig.add_subplot(gs[1, 0])
    style_axes(ax3, xlabel=r"$|x|$  (absolute return)",
               ylabel=r"$P(|X| > x)$",
               title="Tail survival:  empirical vs Gaussian vs power law")
    ax3.loglog(sorted_desc, surv, color=PAL["empirical"],
               linewidth=1.8, label="empirical |returns|")
    ax3.loglog(x_grid, surv_gauss, color=PAL["gaussian"],
               linewidth=1.4, linestyle="--",
               label="Gaussian, matched sigma")
    ax3.loglog(x_pl, surv_pl, color=PAL["power"],
               linewidth=1.6, linestyle=":",
               label=fr"power law  $\alpha = {alpha:.2f}$")
    ax3.axvline(x_min, color=PAL["axis"], linewidth=0.6,
                linestyle=":", alpha=0.7)
    ax3.text(x_min, surv[len(surv)//100], "  $x_{\\min}$",
             color=PAL["text_dim"], fontsize=9, va="bottom")
    ax3.legend(loc="lower left", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # Panel 4: Hill plot — alpha_hat as a function of k.
    ax4 = fig.add_subplot(gs[1, 1])
    style_axes(ax4, xlabel="k  (number of upper order statistics)",
               ylabel=r"$\hat{\alpha}$",
               title="Hill plot: tail-index stability")
    ax4.plot(ks, alphas, color=PAL["power"], linewidth=1.6)
    k_chosen = int(len(sorted_desc) * CONFIG["K_TAIL_FRAC"])
    if k_chosen >= ks[0] and k_chosen <= ks[-1]:
        ax4.axvline(k_chosen, color=PAL["empirical"],
                    linewidth=0.9, linestyle="--",
                    label=f"k = {k_chosen}  (5%)")
        ax4.axhline(alpha, color=PAL["empirical"],
                    linewidth=0.9, linestyle="--", alpha=0.6)
    ax4.fill_between([ks[0], ks[-1]], 2.0, 4.0,
                     color=PAL["gaussian"], alpha=0.06,
                     label="typical equity range  (2-4)")
    ax4.set_ylim(max(1.0, alphas.min() * 0.85),
                 min(8.0, alphas.max() * 1.15))
    ax4.legend(loc="upper right", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # Footer
    fig.text(0.06, 0.025,
             "Empirical |returns| do NOT decay exponentially. "
             "Gaussian models the body, the power law models the tail.",
             color=PAL["text_dim"], fontsize=10, family="DejaVu Sans")
    fig.text(0.97, 0.025, "@quant.traderr",
             color=PAL["text_dim"], fontsize=10, ha="right",
             family="DejaVu Sans")

    out = CONFIG["OUT_PATH"]
    fig.savefig(out, dpi=CONFIG["DPI"], facecolor=PAL["bg"],
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    log(f"[Render] Saved -> {out}  "
        f"({out.stat().st_size/1024:.1f} KB)")

# --- ENTRY POINT ---

def main() -> None:
    t0 = time.time()
    log("=" * 60)
    log("FAT TAILS PIPELINE")
    log("=" * 60)
    returns = fetch_returns()
    render(returns)
    log(f"[Done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
