"""
GARCH_Pipeline.py
==================
Project: Quant Trader Lab - Volatility Forecasting
Author: quant.traderr (Instagram)
License: MIT

Description:
    Production-ready pipeline for GARCH(1,1) volatility modeling.

    Fits a GARCH(1,1) model to real market data and visualizes:
    - Conditional volatility vs realized volatility
    - Volatility clustering (the signature GARCH phenomenon)
    - Forecast cone projecting forward from today

    GARCH(1,1):  sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Pipeline:  DATA -> FIT -> FORECAST -> VISUALIZE (static PNG)
    Resolution: 1920x1080

Dependencies:
    pip install numpy pandas yfinance matplotlib
"""

import os, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "TICKER":           "SPY",
    "LOOKBACK_YEARS":   3,
    "INTERVAL":         "1d",
    "FORECAST_DAYS":    60,
    "OUTPUT_IMAGE":     "GARCH_Output.png",
    "RESOLUTION":       (1920, 1080),
    "DPI":              100,
    "LOG_FILE":         os.path.join(BASE_DIR, "garch_pipeline.log"),
}

THEME = {
    "BG": "#0b0b0b", "PANEL_BG": "#0e0e0e", "GRID": "#1a1a1a",
    "TEXT": "#ffffff", "TEXT_DIM": "#aaaaaa",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "MAGENTA": "#ff1493",
    "YELLOW": "#ffd400", "GREEN": "#00ff41", "RED": "#ff3050",
    "FONT": "Arial",
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
# MODULE 1: DATA
# ═══════════════════════════════════════════════════════════════════
def fetch_data():
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['LOOKBACK_YEARS']}y)...")
    try:
        data = yf.download(CONFIG["TICKER"],
                           period=f"{CONFIG['LOOKBACK_YEARS']}y",
                           interval=CONFIG["INTERVAL"], progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG["TICKER"], axis=1, level=1)
        prices = data["Close"].values.flatten()
        dates = data.index
        returns = np.diff(np.log(prices)) * 100  # percentage log returns
        log(f"[Data] {len(returns)} returns loaded")
        return prices, dates, returns
    except Exception as e:
        log(f"[Error] {e}, using synthetic data")
        n = 756
        returns = np.random.normal(0.04, 1.2, n)
        # Add volatility clustering manually
        for i in range(1, n):
            if abs(returns[i-1]) > 2:
                returns[i] *= 1.8
        prices = 100 * np.exp(np.cumsum(returns / 100))
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n + 1, freq="D")
        return prices, dates, returns


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: GARCH(1,1) FIT
# ═══════════════════════════════════════════════════════════════════
def fit_garch(returns):
    """Fit GARCH(1,1) via maximum likelihood (simplified closed-form)."""
    log("[GARCH] Fitting GARCH(1,1)...")

    n = len(returns)
    r = returns - np.mean(returns)  # demean

    # Initialize with reasonable starting params
    # sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
    omega = 0.01
    alpha = 0.08
    beta = 0.90

    # Simple grid optimization (robust, no scipy needed)
    best_ll = -np.inf
    best_params = (omega, alpha, beta)

    for a in np.linspace(0.02, 0.20, 10):
        for b in np.linspace(0.70, 0.95, 10):
            if a + b >= 0.999:
                continue
            o = np.var(r) * (1 - a - b)
            if o <= 0:
                continue

            sigma2 = np.zeros(n)
            sigma2[0] = np.var(r)
            for t in range(1, n):
                sigma2[t] = o + a * r[t-1]**2 + b * sigma2[t-1]
                if sigma2[t] <= 0:
                    sigma2[t] = 1e-6

            ll = -0.5 * np.sum(np.log(sigma2) + r**2 / sigma2)
            if ll > best_ll:
                best_ll = ll
                best_params = (o, a, b)

    omega, alpha, beta = best_params
    log(f"[GARCH] omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
    log(f"[GARCH] Persistence: {alpha + beta:.4f}")

    # Compute conditional variance series with best params
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]

    cond_vol = np.sqrt(sigma2) * np.sqrt(252)  # annualize

    return cond_vol, best_params


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: FORECAST
# ═══════════════════════════════════════════════════════════════════
def forecast_garch(returns, params, n_days):
    """Multi-step GARCH forecast cone."""
    log(f"[Forecast] {n_days} day forecast...")
    omega, alpha, beta = params
    r = returns - np.mean(returns)

    # Last known values
    last_r2 = r[-1]**2
    last_sigma2 = omega + alpha * r[-2]**2 + beta * (omega / (1 - alpha - beta))

    # Long-run variance
    long_run = omega / (1 - alpha - beta)

    forecast = np.zeros(n_days)
    forecast[0] = omega + alpha * last_r2 + beta * last_sigma2
    for t in range(1, n_days):
        forecast[t] = omega + (alpha + beta) * forecast[t-1]

    fc_vol = np.sqrt(forecast) * np.sqrt(252)
    lr_vol = np.sqrt(long_run) * np.sqrt(252)

    log(f"[Forecast] 1-day vol: {fc_vol[0]:.1f}%, long-run: {lr_vol:.1f}%")
    return fc_vol, lr_vol


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(prices, dates, returns, cond_vol, params, fc_vol, lr_vol):
    log("[Visual] Generating static snapshot...")

    fig = plt.figure(
        figsize=(CONFIG["RESOLUTION"][0] / CONFIG["DPI"],
                 CONFIG["RESOLUTION"][1] / CONFIG["DPI"]),
        dpi=CONFIG["DPI"], facecolor=THEME["BG"],
    )

    gs = GridSpec(3, 1, height_ratios=[1.2, 1.5, 1.0], hspace=0.30,
                  left=0.06, right=0.94, top=0.91, bottom=0.06)

    # Title
    omega, alpha, beta = params
    fig.text(0.50, 0.97,
             f"GARCH(1,1) VOLATILITY MODEL: {CONFIG['TICKER']}",
             ha="center", fontsize=18, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=THEME["FONT"])
    fig.text(0.50, 0.94,
             r"$\sigma^2_t = \omega + \alpha \, r^2_{t-1} + \beta \, \sigma^2_{t-1}$"
             f"          "
             f"$\\alpha = {alpha:.3f}$    $\\beta = {beta:.3f}$    "
             f"persistence = {alpha+beta:.3f}",
             ha="center", fontsize=11, color=THEME["TEXT_DIM"],
             fontfamily=THEME["FONT"])
    fig.text(0.98, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

    def style(ax):
        ax.set_facecolor(THEME["PANEL_BG"])
        ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
        ax.grid(True, color=THEME["GRID"], linewidth=0.3, alpha=0.5)
        for s in ax.spines.values():
            s.set_color(THEME["GRID"]); s.set_linewidth(0.5)

    # Panel 1: Price
    ax1 = fig.add_subplot(gs[0])
    style(ax1)
    ax1.set_title("Price", color=THEME["TEXT_DIM"], fontsize=10, loc="left")
    ax1.plot(dates, prices, color=THEME["CYAN"], linewidth=0.8)
    ax1.set_xlim(dates[0], dates[-1])
    ax1.set_ylabel("USD", color=THEME["TEXT_DIM"], fontsize=9)

    # Panel 2: Conditional Vol + Realized Vol
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    style(ax2)
    ax2.set_title("Conditional vs Realized Volatility (annualized)",
                  color=THEME["TEXT_DIM"], fontsize=10, loc="left")

    # 21-day realized vol
    rv = pd.Series(returns).rolling(21).std().values * np.sqrt(252)
    ax2.plot(dates[1:], rv, color=THEME["CYAN"], linewidth=0.8,
             alpha=0.6, label="21d Realized Vol")
    ax2.plot(dates[1:], cond_vol, color=THEME["ORANGE"], linewidth=1.5,
             label="GARCH Cond. Vol")
    ax2.set_ylabel("Vol (%)", color=THEME["TEXT_DIM"], fontsize=9)
    ax2.set_ylim(0, max(np.nanmax(rv), np.max(cond_vol)) * 1.2)
    leg = ax2.legend(fontsize=8, facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for t in leg.get_texts(): t.set_color(THEME["TEXT_DIM"])

    # Panel 3: Forecast Cone
    ax3 = fig.add_subplot(gs[2])
    style(ax3)
    ax3.set_title(f"{CONFIG['FORECAST_DAYS']}-Day Volatility Forecast",
                  color=THEME["TEXT_DIM"], fontsize=10, loc="left")

    fc_days = np.arange(1, len(fc_vol) + 1)
    ax3.plot(fc_days, fc_vol, color=THEME["ORANGE"], linewidth=2.0,
             label="GARCH Forecast")
    ax3.axhline(lr_vol, color=THEME["YELLOW"], linewidth=1.0,
                linestyle="--", label=f"Long-run: {lr_vol:.1f}%")
    ax3.fill_between(fc_days, fc_vol * 0.8, fc_vol * 1.2,
                     color=THEME["ORANGE"], alpha=0.08)
    ax3.set_xlabel("Days Ahead", color=THEME["TEXT_DIM"], fontsize=9)
    ax3.set_ylabel("Ann. Vol (%)", color=THEME["TEXT_DIM"], fontsize=9)
    leg3 = ax3.legend(fontsize=8, facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for t in leg3.get_texts(): t.set_color(THEME["TEXT_DIM"])

    # HUD
    fig.text(0.94, 0.91,
             f"Current Vol: {cond_vol[-1]:.1f}%    Long-run: {lr_vol:.1f}%",
             ha="right", fontsize=10, fontweight="bold",
             color=THEME["YELLOW"], fontfamily=THEME["FONT"])

    out = os.path.join(BASE_DIR, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"[Visual] Saved to {out}")


def main():
    t0 = time.time()
    log("=== GARCH PIPELINE ===")
    prices, dates, returns = fetch_data()
    cond_vol, params = fit_garch(returns)
    fc_vol, lr_vol = forecast_garch(returns, params, CONFIG["FORECAST_DAYS"])
    visualize(prices, dates, returns, cond_vol, params, fc_vol, lr_vol)
    log(f"=== PIPELINE FINISHED ({time.time()-t0:.1f}s) ===")

if __name__ == "__main__":
    main()
