"""
StatArb_Pipeline.py
====================
Project  : Quant Trader Lab - Statistical Arbitrage & Cointegration
Author   : quant.traderr (Instagram)

Description:
    Static visualization of a cointegrated pairs trade.
    Two correlated assets drift together while the mispricing spread
    mean-reverts around zero - visualized as green/red bars with
    two price lines weaving through them.

    Aesthetic: Cream/beige background, matplotlib classic quant style,
    thin vertical bars, bold marker lines, color-coded right y-axes.

    Pipeline Steps:
    1. DATA      - Generate two cointegrated price series (OU spread).
    2. RENDERING - Single matplotlib static image (1080p PNG).

Dependencies:
    pip install numpy pandas matplotlib
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

CONFIG = {
    # Synthetic data
    "N_POINTS": 500,
    "SEED": 42,
    # OU process for spread
    "OU_THETA": 0.08,
    "OU_MU": 0.0,
    "OU_SIGMA": 0.012,
    # Price params
    "PRICE_A_START": 55.15,
    "PRICE_B_START": 52.30,
    "DRIFT_SIGMA": 0.004,
    # Output
    "RESOLUTION_W": 1920,
    "RESOLUTION_H": 1080,
    "DPI": 150,
    "OUTPUT_FILE": "StatArb_Output.png",
    "LOG_FILE": "statarb_pipeline.log",
}

# Exact aesthetic from reference image
THEME = {
    "BG":          "#f5edd6",     # Warm cream/beige paper background
    "PLOT_BG":     "#f5edd6",     # Same cream inside plot area
    "BAR_GREEN":   "#2e8b3e",     # Solid green bars (spread > 0)
    "BAR_RED":     "#cc2233",     # Solid red bars (spread < 0)
    "BAR_GRAY":    "#c0bcae",     # Faded gray for weak bars
    "LINE_A":      "#1a1acd",     # Dark blue - Asset A
    "LINE_B":      "#ee8800",     # Warm orange - Asset B
    "MARKER_A":    "#1a1acd",
    "MARKER_B":    "#ee8800",
    "AXIS_TEXT":   "#333333",
    "TICK_TEXT":   "#555555",
    "GRID":        "#ddd8c8",     # Very subtle grid
    "FONT":        "monospace",
}

# --- UTILS ---

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

# --- MODULE 1: DATA ---

def generate_cointegrated_data():
    """
    Generate two synthetic cointegrated price series.
    Shared random-walk + OU mean-reverting spread.
    """
    np.random.seed(CONFIG["SEED"])
    n = CONFIG["N_POINTS"]

    # Shared stochastic trend
    shared_noise = np.random.randn(n) * CONFIG["DRIFT_SIGMA"]
    shared_trend = np.cumsum(shared_noise)

    # OU process for spread
    theta = CONFIG["OU_THETA"]
    mu = CONFIG["OU_MU"]
    sigma = CONFIG["OU_SIGMA"]
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = (spread[i-1]
                     + theta * (mu - spread[i-1])
                     + sigma * np.random.randn())

    # Small idiosyncratic noise
    noise_a = np.cumsum(np.random.randn(n) * 0.0008)
    noise_b = np.cumsum(np.random.randn(n) * 0.0008)

    price_a = CONFIG["PRICE_A_START"] + shared_trend + noise_a + spread * 0.5
    price_b = CONFIG["PRICE_B_START"] + shared_trend + noise_b - spread * 0.5

    df = pd.DataFrame({
        "price_a": price_a,
        "price_b": price_b,
        "spread":  spread,
    })

    log(f"[Data] Generated {n} cointegrated points.")
    log(f"[Data] Spread: [{spread.min():.4f}, {spread.max():.4f}]")
    log(f"[Data] Price A: {price_a[0]:.2f} -> {price_a[-1]:.2f}")
    log(f"[Data] Price B: {price_b[0]:.2f} -> {price_b[-1]:.2f}")
    return df

# --- MODULE 2: RENDERING ---

def render_image(df):
    """Render the full static chart as a single PNG."""
    log("[Render] Building static image...")

    n = len(df)
    x = np.arange(n)

    spread_vals = df["spread"].values
    pa_vals = df["price_a"].values
    pb_vals = df["price_b"].values

    # Axis ranges
    spread_absmax = max(abs(spread_vals.min()), abs(spread_vals.max())) * 1.15
    pa_min_raw, pa_max_raw = pa_vals.min(), pa_vals.max()
    pb_min_raw, pb_max_raw = pb_vals.min(), pb_vals.max()
    pa_pad = (pa_max_raw - pa_min_raw) * 0.08
    pb_pad = (pb_max_raw - pb_min_raw) * 0.08
    pa_mid = (pa_min_raw + pa_max_raw) / 2
    pb_mid = (pb_min_raw + pb_max_raw) / 2

    # --- Figure setup ---
    fig_w = CONFIG["RESOLUTION_W"] / CONFIG["DPI"]
    fig_h = CONFIG["RESOLUTION_H"] / CONFIG["DPI"]
    fig, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=CONFIG["DPI"])
    fig.patch.set_facecolor(THEME["BG"])
    ax1.set_facecolor(THEME["PLOT_BG"])

    # --- 1) Spread bars (left y-axis) ---
    bar_colors = []
    bar_alphas = []
    for v in spread_vals:
        mag = abs(v) / spread_absmax
        if v >= 0:
            bar_colors.append(THEME["BAR_GREEN"])
            bar_alphas.append(0.3 + 0.7 * min(mag * 2, 1.0))
        else:
            bar_colors.append(THEME["BAR_RED"])
            bar_alphas.append(0.3 + 0.7 * min(mag * 2, 1.0))

    for i_bar in range(n):
        ax1.vlines(
            x[i_bar], 0, spread_vals[i_bar],
            colors=bar_colors[i_bar],
            alpha=bar_alphas[i_bar],
            linewidth=1.8,
        )

    # Zero line
    ax1.axhline(0, color="#999999", linewidth=0.5, alpha=0.5)

    ax1.set_ylabel("mispricing", fontsize=13, fontfamily=THEME["FONT"],
                    color=THEME["AXIS_TEXT"])
    ax1.set_xlabel("time", fontsize=13, fontfamily=THEME["FONT"],
                    color=THEME["AXIS_TEXT"])
    ax1.set_xlim(-2, n + 5)
    ax1.set_ylim(-spread_absmax, spread_absmax)
    ax1.tick_params(axis="y", labelcolor=THEME["TICK_TEXT"], labelsize=10)
    ax1.tick_params(axis="x", labelcolor=THEME["TICK_TEXT"], labelsize=10)
    ax1.grid(True, alpha=0.3, color=THEME["GRID"], linewidth=0.5)

    # --- 2) Asset A (right y-axis #1 - blue) ---
    ax2 = ax1.twinx()
    ax2.plot(x, pa_vals, color=THEME["LINE_A"], linewidth=2.2, zorder=5)
    ax2.plot(x, pa_vals, "o", color=THEME["MARKER_A"],
             markersize=3.5, zorder=6)

    ax2.set_ylabel(
        f"marketId = {pa_mid:.1f}",
        fontsize=11, fontfamily=THEME["FONT"],
        color=THEME["LINE_A"], rotation=270, labelpad=22,
    )
    ax2.set_ylim(pa_min_raw - pa_pad, pa_max_raw + pa_pad)
    ax2.tick_params(axis="y", labelcolor=THEME["LINE_A"], labelsize=8, pad=3)
    ax2.spines["right"].set_color(THEME["LINE_A"])

    # --- 3) Asset B (right y-axis #2 - orange, offset far right) ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 65))
    ax3.plot(x, pb_vals, color=THEME["LINE_B"], linewidth=2.2, zorder=5)
    ax3.plot(x, pb_vals, "o", color=THEME["MARKER_B"],
             markersize=3.5, zorder=6)

    ax3.set_ylabel(
        f"marketId = {pb_mid:.0f}",
        fontsize=11, fontfamily=THEME["FONT"],
        color=THEME["LINE_B"], rotation=270, labelpad=22,
    )
    ax3.set_ylim(pb_min_raw - pb_pad, pb_max_raw + pb_pad)
    ax3.tick_params(axis="y", labelcolor=THEME["LINE_B"], labelsize=8, pad=3)
    ax3.spines["right"].set_color(THEME["LINE_B"])

    # Clean up spines
    for spine in ["top"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
        ax3.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.78, top=0.95, bottom=0.10)

    out = os.path.abspath(CONFIG["OUTPUT_FILE"])
    fig.savefig(out, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"[Success] Output saved: {out}")


# --- MAIN ---

def main():
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    log("=" * 50)
    log("  STAT-ARB & COINTEGRATION PIPELINE")
    log("=" * 50)

    df = generate_cointegrated_data()
    render_image(df)

    log("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
