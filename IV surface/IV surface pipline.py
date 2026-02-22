"""
IV Surface Pipeline.py
======================
Project: Quant Trader Lab - Implied Volatility Surface
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for generating a jagged, market-realistic
    Implied Volatility (IV) surface with scattered market quotes.

    The surface is fitted via linear interpolation from noisy option quotes,
    producing the rough, spiky terrain characteristic of real IV surfaces.

    Pipeline Steps:
    1.  **Data Acquisition**: Generates synthetic IV scatter data (User can plug in API).
    2.  **Surface Engine**: Fits a jagged surface via scipy griddata interpolation.
    3.  **Visualization**: Generates a high-fidelity static 16:9 dashboard.

    NOTE: Video rendering removed for pipeline efficiency. See visual.py for timelapse.

Dependencies:
    pip install numpy matplotlib scipy
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "N_POINTS": 1800,       # Number of market quote scatter points
    "SEED": 42,

    # Surface Grid
    "GRID_RES": 50,         # Resolution of the interpolated surface mesh
    "MONEYNESS_RANGE": (-0.15, 0.15),   # ATM-centered moneyness (m = K/S - 1)
    "MATURITY_RANGE": (10, 200),        # Time to maturity in days

    # IV Model Parameters (controls smile shape)
    "BASE_IV": 0.42,        # At-the-money base implied volatility
    "SMILE_COEFF": 3.5,     # Smile curvature (higher = steeper wings)
    "SKEW_COEFF": -1.5,     # Put skew (negative = puts more expensive)
    "TERM_SLOPE": 0.0005,   # Term structure slope
    "NOISE_STD": 0.05,      # Gaussian noise standard deviation
    "FAT_TAIL_SCALE": 0.5,  # Exponential fat-tail spike scale
    "RIDGE_AMPLITUDE": 0.12,# Localized ridge sine-wave amplitude

    # Output
    "OUTPUT_IMAGE": "iv_surface_static.png",

    # Aesthetics
    "COLOR_BG": '#0a0a0f',
    "SCATTER_COLOR": '#2266EE',
    "SCATTER_SIZE": 14,
    "SCATTER_ALPHA": 0.65,
    "SURFACE_CMAP": 'viridis',
    "SURFACE_ALPHA": 0.88,
    "SURFACE_EDGE": '#2a4a6a',
    "LABEL_COLOR": '#9999BB',
    "ACCENT_COLOR": '#FF9933',
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


# --- MODULE 1: DATA ---

def fetch_data():
    """
    Generates or fetches option market IV scatter data.
    Returns: (moneyness_array, maturity_array, iv_array)
    """
    log("[Data] Initializing data module...")

    # -------------------------------------------------------------------------
    # [USER INPUT REQUIRED] - API / BROKER INTEGRATION TEMPLATE
    # -------------------------------------------------------------------------
    # To use your own options data (e.g., from IBKR, Deribit, or CSV):
    #
    # import pandas as pd
    #
    # def get_option_chain(underlying, expiry_range):
    #     """
    #     Fetch real option chain data from your broker/API.
    #     Must return three arrays:
    #       - moneyness: (strike / spot - 1), centered at 0 for ATM
    #       - maturity:  days to expiration
    #       - iv:        implied volatility (decimal, e.g. 0.35 = 35%)
    #     """
    #     # Example: Load from CSV
    #     # df = pd.read_csv('option_chain.csv')
    #     # moneyness = (df['strike'] / df['spot_price']) - 1.0
    #     # maturity = df['days_to_expiry'].values
    #     # iv = df['implied_vol'].values
    #     #
    #     # Example: Deribit API
    #     # import ccxt
    #     # exchange = ccxt.deribit({'enableRateLimit': True})
    #     # options = exchange.fetch_option_chain('BTC')
    #     # ... parse strikes, expiries, IVs ...
    #     #
    #     # return moneyness.values, maturity, iv
    #
    # scatter_m, scatter_t, scatter_iv = get_option_chain('BTC', '30-180d')
    # -------------------------------------------------------------------------

    log("[Data] Using Synthetic IV Surface Data.")
    np.random.seed(CONFIG['SEED'])
    n = CONFIG['N_POINTS']
    m_lo, m_hi = CONFIG['MONEYNESS_RANGE']
    t_lo, t_hi = CONFIG['MATURITY_RANGE']

    scatter_m = np.random.uniform(m_lo, m_hi, n)
    scatter_t = np.random.uniform(t_lo, t_hi, n)

    # Realistic IV: smile + skew + term structure + noise + fat tails + ridges
    scatter_iv = (
        CONFIG['BASE_IV']
        + CONFIG['SMILE_COEFF'] * (scatter_m ** 2) * np.exp(-0.006 * scatter_t)
        + CONFIG['SKEW_COEFF'] * scatter_m * np.exp(-0.008 * scatter_t)
        + CONFIG['TERM_SLOPE'] * scatter_t
        + np.random.normal(0, CONFIG['NOISE_STD'], n)
        + CONFIG['FAT_TAIL_SCALE'] * 0.15 * np.random.exponential(1.0, n)
        + CONFIG['RIDGE_AMPLITUDE'] * np.sin(scatter_m * 80) * np.exp(-0.01 * scatter_t)
    )

    log(f"[Data] Generated {n} synthetic option quotes.")
    return scatter_m, scatter_t, scatter_iv


# --- MODULE 2: SURFACE ENGINE ---

def fit_surface(scatter_m, scatter_t, scatter_iv):
    """
    Fits a jagged surface via linear interpolation from scattered IV quotes.
    Returns: (grid_m_mesh, grid_t_mesh, Z_surface)
    """
    log("[Engine] Fitting IV surface via griddata (linear interpolation)...")

    res = CONFIG['GRID_RES']
    m_lo, m_hi = CONFIG['MONEYNESS_RANGE']
    t_lo, t_hi = CONFIG['MATURITY_RANGE']

    grid_m = np.linspace(m_lo, m_hi, res)
    grid_t = np.linspace(t_lo, t_hi, res)
    GM, GT = np.meshgrid(grid_m, grid_t)

    # Linear interpolation → jagged/rough terrain (NOT smooth)
    Z = griddata((scatter_m, scatter_t), scatter_iv, (GM, GT), method='linear')

    # Fill NaN edges with nearest-neighbor
    Z_nearest = griddata((scatter_m, scatter_t), scatter_iv, (GM, GT), method='nearest')
    mask = np.isnan(Z)
    Z[mask] = Z_nearest[mask]

    z_range = Z.max() - Z.min()
    log(f"[Engine] Surface fitted. Z range: [{Z.min():.3f}, {Z.max():.3f}] (Δ={z_range:.3f})")

    return GM, GT, Z


# --- MODULE 3: VISUALIZATION ---

def render_static_dashboard(GM, GT, Z, scatter_m, scatter_t, scatter_iv):
    """
    Generates the static high-fidelity 16:9 IV surface image.
    """
    log("[Visual] Rendering static dashboard...")

    bg = CONFIG['COLOR_BG']
    fig = plt.figure(figsize=(16, 9), facecolor=bg, dpi=150)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax = fig.add_axes([0.03, 0.10, 0.94, 0.82], projection='3d', facecolor=bg)

    # Dark panes + subtle grid
    ax.xaxis.set_pane_color((0.05, 0.05, 0.08, 1.0))
    ax.yaxis.set_pane_color((0.05, 0.05, 0.08, 1.0))
    ax.zaxis.set_pane_color((0.05, 0.05, 0.08, 1.0))
    ax.xaxis._axinfo['grid'].update(color='#1a1a2e', linewidth=0.4)
    ax.yaxis._axinfo['grid'].update(color='#1a1a2e', linewidth=0.4)
    ax.zaxis._axinfo['grid'].update(color='#1a1a2e', linewidth=0.4)
    ax.tick_params(colors='#555555', labelsize=7)

    # 1. Surface (jagged, with visible mesh edges)
    ax.plot_surface(
        GM, GT, Z,
        cmap=CONFIG['SURFACE_CMAP'],
        edgecolor=CONFIG['SURFACE_EDGE'],
        linewidth=0.25,
        alpha=CONFIG['SURFACE_ALPHA'],
        rstride=1, cstride=1
    )

    # 2. Scatter cloud (market quotes)
    ax.scatter(
        scatter_m, scatter_t, scatter_iv,
        c=CONFIG['SCATTER_COLOR'],
        s=CONFIG['SCATTER_SIZE'],
        alpha=CONFIG['SCATTER_ALPHA'],
        depthshade=True,
        edgecolors='none'
    )

    # Axis limits
    m_lo, m_hi = CONFIG['MONEYNESS_RANGE']
    t_lo, t_hi = CONFIG['MATURITY_RANGE']
    ax.set_xlim(m_lo, m_hi)
    ax.set_ylim(t_lo, t_hi)
    z_pad = 0.03
    ax.set_zlim(Z.min() - z_pad, Z.max() + z_pad)

    # Labels
    ax.set_xlabel('\nMoneyness: m', color=CONFIG['LABEL_COLOR'], fontsize=10, linespacing=2.5)
    ax.set_ylabel('\nTime to Maturity: τ', color=CONFIG['LABEL_COLOR'], fontsize=10, linespacing=2.5)
    ax.set_zlabel('\nImplied Volatility: σ', color=CONFIG['ACCENT_COLOR'], fontsize=10, linespacing=2.5)

    # Camera
    ax.view_init(elev=22, azim=240)

    # HUD
    fig.text(0.04, 0.94, "IMPLIED VOLATILITY SURFACE",
             fontsize=16, color='white', weight='bold', family='monospace')
    fig.text(0.04, 0.915, f"N={CONFIG['N_POINTS']} quotes  |  Grid: {CONFIG['GRID_RES']}×{CONFIG['GRID_RES']}",
             fontsize=9, color='#555555', family='monospace')

    # Stats box
    stats = (f"Base IV:  {CONFIG['BASE_IV']:.2f}\n"
             f"Smile:    {CONFIG['SMILE_COEFF']:.1f}\n"
             f"Skew:     {CONFIG['SKEW_COEFF']:.1f}\n"
             f"Z range:  [{Z.min():.3f}, {Z.max():.3f}]")
    props = dict(boxstyle='round', facecolor='#0d0d30', alpha=0.8, edgecolor='#1e6091')
    fig.text(0.88, 0.94, stats, fontsize=8, color='white', family='monospace',
             va='top', ha='left', bbox=props)

    # Formula
    formula = r"$\sigma(m,\tau) = \sigma_0 + \alpha \cdot m^2 e^{-\beta\tau} - \delta \cdot m \cdot e^{-\gamma\tau} + \lambda\tau$"
    fig.text(0.5, 0.025, formula, fontsize=15, color='white', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="#0d0d30", ec="#1e6091", alpha=0.9))

    # Save
    out_path = os.path.join(os.path.dirname(__file__), CONFIG['OUTPUT_IMAGE'])
    plt.savefig(out_path, dpi=150, facecolor=bg)
    plt.close(fig)
    log(f"[Visual] Dashboard saved to {out_path}")


# --- MAIN ---

def main():
    log("=== IV SURFACE PIPELINE ===")

    # 1. Data
    scatter_m, scatter_t, scatter_iv = fetch_data()

    # 2. Engine
    GM, GT, Z = fit_surface(scatter_m, scatter_t, scatter_iv)

    # 3. Visual
    render_static_dashboard(GM, GT, Z, scatter_m, scatter_t, scatter_iv)

    log("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
