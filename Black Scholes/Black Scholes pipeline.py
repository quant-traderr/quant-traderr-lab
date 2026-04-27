"""
BS_Static_Pipeline.py
=====================
Project: Quant Trader Lab — Option Pricing
Author:  quant.traderr (Instagram)
License: MIT

Description:
    Production pipeline that ships a single static PNG telling the
    full Black-Scholes story: the iconic 3D price surface, time-decay
    slices, the Greeks, and the volatility smile that breaks the
    flat-sigma assumption. Drop-in for Demiurge title slides or
    standalone posts.

    Pipeline Steps:
    1. Closed-form Black-Scholes call pricer (no finite difference).
    2. Build V(S, tau) on a dense grid + Delta and Gamma slices.
    3. Pull a live SPY option chain via yfinance, invert BS to get
       implied vol per strike, plot the smile vs the flat-sigma
       assumption. Falls back to a stylised smile if the network
       call fails.
    4. Render 2560x1600 "Bloomberg Dark" 4-panel composition.

Output:
    bs_static.png — 2560x1600. Title slide ready, no in-frame
    title or watermark text inside the figure body.

Dependencies:
    pip install numpy scipy matplotlib yfinance

Performance:
    Closed-form BS is vectorised over the (S, tau) grid; full
    pipeline including network call < 6s.
"""

from __future__ import annotations
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq
from scipy.stats import norm

import yfinance as yf

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Surface domain
    "K":      100.0,
    "R":      0.05,
    "SIGMA":  0.30,    # flat sigma used for surface + Greeks
    "T":      1.0,
    "S_MIN":  40.0,
    "S_MAX":  200.0,
    "N_S":    140,
    "N_TAU":  60,

    # Greeks slice: tau evaluated at this time-to-expiry (in years)
    "GREEKS_TAU": 0.25,

    # Smile data source
    "SMILE_TICKER":   "SPY",
    "SMILE_FALLBACK": True,   # if yfinance fails, render synthetic smile

    # Output
    "OUT_PATH": Path(__file__).parent / "bs_static.png",
    "WIDTH":  2560,
    "HEIGHT": 1600,
    "DPI":    170,
}

# Bloomberg-dark palette (matches Fat Tails Pipeline)
PAL = {
    "bg":        "#000000",
    "panel":     "#0a0a0a",
    "axis":      "#3a3a3a",
    "grid":      "#181818",
    "text":      "#ffffff",
    "text_dim":  "#9a9a9a",
    "surface":   "#5fa8d3",   # steel blue surface
    "surface_2": "#d4af37",   # gold rim/highlights
    "payoff":    "#ff8c42",   # warm orange payoff
    "delta":     "#5fa8d3",
    "gamma":     "#d4af37",
    "smile":     "#ff8c42",
    "ref":       "#9a9a9a",
}


# --- UTILITIES ---

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# --- MODULE 1: BLACK-SCHOLES CORE ---

def bs_call(S, K, T, r, sigma):
    """Closed-form European call. Vectorised over S and T."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    eps = 1e-12
    sqrtT = np.sqrt(np.maximum(T, eps))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    # Handle T=0 limit cleanly: payoff
    price = np.where(T <= eps, np.maximum(S - K, 0.0), price)
    return price


def bs_delta(S, K, T, r, sigma):
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return norm.cdf(d1)


def bs_gamma(S, K, T, r, sigma):
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return norm.pdf(d1) / (S * sigma * sqrtT)


def implied_vol_call(price, S, K, T, r) -> Optional[float]:
    """Brent root-find on Black-Scholes call -> implied vol. Returns None on
    no-arbitrage violation."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price < intrinsic - 1e-6 or price > S - 1e-6:
        return None
    try:
        return brentq(
            lambda s: bs_call(S, K, T, r, s) - price,
            1e-4, 5.0, xtol=1e-6, maxiter=100,
        )
    except (ValueError, RuntimeError):
        return None


# --- MODULE 2: SURFACE + GREEKS ---

def build_surface():
    S = np.linspace(CONFIG["S_MIN"], CONFIG["S_MAX"], CONFIG["N_S"])
    tau = np.linspace(1e-4, CONFIG["T"], CONFIG["N_TAU"])
    Sg, Tg = np.meshgrid(S, tau, indexing="xy")
    V = bs_call(Sg, CONFIG["K"], Tg, CONFIG["R"], CONFIG["SIGMA"])
    return S, tau, Sg, Tg, V


def greeks_slice(tau_slice: float):
    S = np.linspace(CONFIG["S_MIN"], CONFIG["S_MAX"], 300)
    delta = bs_delta(S, CONFIG["K"], tau_slice, CONFIG["R"], CONFIG["SIGMA"])
    gamma = bs_gamma(S, CONFIG["K"], tau_slice, CONFIG["R"], CONFIG["SIGMA"])
    return S, delta, gamma


# --- MODULE 3: VOLATILITY SMILE FROM LIVE OPTION CHAIN ---

def fetch_smile():
    """Pull the nearest-expiry call chain for SMILE_TICKER, invert BS,
    return moneyness vs implied vol arrays."""
    log(f"[Smile] Fetching {CONFIG['SMILE_TICKER']} option chain...")
    tk = yf.Ticker(CONFIG["SMILE_TICKER"])
    expiries = tk.options
    if not expiries:
        raise RuntimeError("No expiries returned.")
    # Pick the first expiry that's at least 14 days out (avoids week-of noise)
    today = np.datetime64("today", "D")
    chosen = None
    for e in expiries:
        if (np.datetime64(e, "D") - today).astype(int) >= 14:
            chosen = e
            break
    chosen = chosen or expiries[0]
    chain = tk.option_chain(chosen)
    calls = chain.calls
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    days = (np.datetime64(chosen, "D") - today).astype(int)
    T = max(days / 365.0, 1e-3)
    r = 0.05  # rough US risk-free

    # Filter: ATM neighbourhood. Prefer bid/ask mid; fall back to lastPrice
    # so the smile renders even after market hours (when bid=ask=0).
    moneyness, ivs = [], []
    for _, row in calls.iterrows():
        K = float(row["strike"])
        if not (0.7 * spot <= K <= 1.3 * spot):
            continue
        bid = row.get("bid", 0.0) or 0.0
        ask = row.get("ask", 0.0) or 0.0
        last = row.get("lastPrice", 0.0) or 0.0
        if bid > 0 and ask > 0:
            price = 0.5 * (bid + ask)
        elif last > 0:
            price = last
        else:
            continue
        iv = implied_vol_call(price, spot, K, T, r)
        if iv is None or iv > 2.0 or iv < 0.02:
            continue
        moneyness.append(K / spot)
        ivs.append(iv)
    if len(ivs) < 6:
        raise RuntimeError("Not enough valid IV points.")
    log(f"[Smile] {chosen}  spot={spot:.2f}  T={T:.3f}y  "
        f"{len(ivs)} valid strikes")
    order = np.argsort(moneyness)
    return (np.array(moneyness)[order], np.array(ivs)[order],
            CONFIG["SMILE_TICKER"], chosen, spot, T)


def synthetic_smile():
    """Stylised smile if the network call fails. Quadratic in log-moneyness."""
    log("[Smile] Using synthetic fallback smile.")
    m = np.linspace(0.75, 1.25, 41)
    base = 0.18
    skew = 0.08 * (1.0 - m)         # left-skew (puts richer)
    curv = 0.40 * (np.log(m)) ** 2  # convex curvature
    iv = base + skew + curv
    return m, iv, "synthetic", "stylised", 100.0, 0.25


# --- MODULE 4: RENDER ---

def style_axes(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor(PAL["panel"])
    for s in ax.spines.values():
        s.set_color(PAL["axis"])
        s.set_linewidth(0.8)
    ax.tick_params(colors=PAL["text_dim"], labelsize=10, length=4)
    ax.grid(True, color=PAL["grid"], linewidth=0.6, alpha=0.9)
    if xlabel: ax.set_xlabel(xlabel, color=PAL["text_dim"], fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=PAL["text_dim"], fontsize=11)
    if title:
        ax.set_title(title, color=PAL["text"], fontsize=13, pad=10,
                     loc="left", fontweight="bold")


def style_axes3d(ax, *, title=""):
    ax.set_facecolor(PAL["panel"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.04, 0.04, 0.04, 1.0))
        axis._axinfo["grid"]["color"] = PAL["grid"]
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.label.set_color(PAL["text_dim"])
        axis.label.set_fontsize(11)
    ax.tick_params(colors=PAL["text_dim"], labelsize=9)
    if title:
        ax.set_title(title, color=PAL["text"], fontsize=13, pad=10,
                     loc="left", fontweight="bold")


def render(S, tau, V, S_g, delta, gamma, smile_data):
    log("[Render] Building 4-panel static...")
    moneyness, ivs, smile_ticker, smile_expiry, spot, smile_T = smile_data

    fig = plt.figure(
        figsize=(CONFIG["WIDTH"] / CONFIG["DPI"],
                 CONFIG["HEIGHT"] / CONFIG["DPI"]),
        facecolor=PAL["bg"],
    )
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.05, right=0.97, top=0.88, bottom=0.09,
        wspace=0.22, hspace=0.36,
    )

    # Header banner
    fig.text(0.05, 0.945, "BLACK-SCHOLES",
             color=PAL["text"], fontsize=22, fontweight="bold",
             family="DejaVu Sans")
    fig.text(0.05, 0.918,
             f"European call,  K={CONFIG['K']:.0f},  r={CONFIG['R']:.0%},  "
             f"sigma={CONFIG['SIGMA']:.0%},  T={CONFIG['T']:.0f}y",
             color=PAL["text_dim"], fontsize=11, family="DejaVu Sans")
    atm = bs_call(CONFIG["K"], CONFIG["K"], CONFIG["T"],
                  CONFIG["R"], CONFIG["SIGMA"])
    fig.text(0.97, 0.945, f"ATM = {atm:.2f}",
             color=PAL["surface_2"], fontsize=22, fontweight="bold",
             ha="right", family="DejaVu Sans")
    fig.text(0.97, 0.918, "closed-form call value at S = K",
             color=PAL["text_dim"], fontsize=11, ha="right",
             family="DejaVu Sans")

    # ---- Panel 1: 3D surface V(S, tau) ----
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    style_axes3d(ax1, title="Option value surface  V(S, tau)")
    Sg2, Tg2 = np.meshgrid(S, tau, indexing="xy")
    surf = ax1.plot_surface(
        Sg2, Tg2, V,
        cmap="cividis", linewidth=0, antialiased=True,
        rcount=80, ccount=80, alpha=0.95,
    )
    # Wireframe rim for definition
    ax1.plot_wireframe(
        Sg2, Tg2, V,
        rcount=12, ccount=12,
        color=PAL["surface_2"], linewidth=0.4, alpha=0.45,
    )
    # Payoff edge at tau=0 in vivid orange
    payoff = np.maximum(S - CONFIG["K"], 0.0)
    ax1.plot(S, np.zeros_like(S), payoff,
             color=PAL["payoff"], linewidth=2.0, zorder=10,
             label="payoff at expiry")
    ax1.set_xlabel("spot  S")
    ax1.set_ylabel("time to expiry  tau")
    ax1.set_zlabel("V(S, tau)")
    ax1.view_init(elev=22, azim=-58)
    ax1.legend(loc="upper left", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # ---- Panel 2: V(S) slices at multiple tau ----
    ax2 = fig.add_subplot(gs[0, 1])
    style_axes(ax2, xlabel="spot  S", ylabel="call value  V",
               title="Time decay:  V(S) at multiple tau")
    tau_picks = [0.0, 0.05, 0.25, 0.5, 1.0]
    cmap = plt.cm.cividis(np.linspace(0.25, 0.95, len(tau_picks)))
    for color, t_pick in zip(cmap, tau_picks):
        if t_pick <= 1e-6:
            ax2.plot(S, np.maximum(S - CONFIG["K"], 0.0),
                     color=PAL["payoff"], linewidth=2.0,
                     label=f"tau = {t_pick:.2f}  (payoff)")
        else:
            ax2.plot(S, bs_call(S, CONFIG["K"], t_pick,
                                CONFIG["R"], CONFIG["SIGMA"]),
                     color=color, linewidth=1.6,
                     label=f"tau = {t_pick:.2f}")
    ax2.axvline(CONFIG["K"], color=PAL["ref"], linewidth=0.6,
                linestyle=":", alpha=0.7)
    ax2.text(CONFIG["K"], ax2.get_ylim()[1] * 0.95, "  K",
             color=PAL["text_dim"], fontsize=9, va="top")
    ax2.legend(loc="upper left", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # ---- Panel 3: Greeks slice ----
    ax3 = fig.add_subplot(gs[1, 0])
    style_axes(ax3, xlabel="spot  S", ylabel="Delta",
               title=f"Greeks at tau = {CONFIG['GREEKS_TAU']}  "
                     "(Delta and Gamma)")
    ax3.plot(S_g, delta, color=PAL["delta"], linewidth=1.8,
             label="Delta  (left axis)")
    ax3.set_ylim(-0.05, 1.05)
    ax3b = ax3.twinx()
    ax3b.plot(S_g, gamma, color=PAL["gamma"], linewidth=1.6,
              linestyle="--", label="Gamma  (right axis)")
    ax3b.set_ylabel("Gamma", color=PAL["text_dim"], fontsize=11)
    ax3b.tick_params(colors=PAL["text_dim"], labelsize=10)
    for s in ax3b.spines.values():
        s.set_color(PAL["axis"])
        s.set_linewidth(0.8)
    ax3.axvline(CONFIG["K"], color=PAL["ref"], linewidth=0.6,
                linestyle=":", alpha=0.7)
    # Combined legend
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, loc="center right",
               facecolor=PAL["panel"], edgecolor=PAL["axis"],
               fontsize=9, labelcolor=PAL["text_dim"])

    # ---- Panel 4: Volatility smile ----
    ax4 = fig.add_subplot(gs[1, 1])
    style_axes(
        ax4, xlabel="moneyness  K / S",
        ylabel="implied volatility",
        title=f"Where flat sigma breaks:  {smile_ticker} smile  ({smile_expiry})",
    )
    ax4.plot(moneyness, ivs, color=PAL["smile"], linewidth=2.0,
             marker="o", markersize=4, markerfacecolor=PAL["smile"],
             markeredgecolor=PAL["bg"], markeredgewidth=0.5,
             label="market IV per strike")
    flat_sigma = float(np.median(ivs))
    ax4.axhline(flat_sigma, color=PAL["surface"], linewidth=1.3,
                linestyle="--",
                label=f"flat BS sigma = {flat_sigma:.1%}")
    ax4.axvline(1.0, color=PAL["ref"], linewidth=0.6,
                linestyle=":", alpha=0.7)
    ax4.text(1.0, ax4.get_ylim()[1] * 0.95, "  ATM",
             color=PAL["text_dim"], fontsize=9, va="top")
    ax4.legend(loc="upper center", facecolor=PAL["panel"],
               edgecolor=PAL["axis"], fontsize=9,
               labelcolor=PAL["text_dim"])

    # Footer
    fig.text(0.05, 0.025,
             "One PDE prices the surface. The smile shows the market disagrees.",
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
    log("BLACK-SCHOLES STATIC PIPELINE")
    log("=" * 60)

    # Surface + Greeks
    S, tau, _, _, V = build_surface()
    S_g, delta, gamma = greeks_slice(CONFIG["GREEKS_TAU"])
    log(f"[Surface] grid {len(S)} x {len(tau)}, "
        f"V range [{V.min():.2f}, {V.max():.2f}]")

    # Smile (with fallback)
    try:
        smile_data = fetch_smile()
    except Exception as e:
        log(f"[Smile] live fetch failed: {e}")
        if not CONFIG["SMILE_FALLBACK"]:
            raise
        smile_data = synthetic_smile()

    render(S, tau, V, S_g, delta, gamma, smile_data)
    log(f"[Done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
