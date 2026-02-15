"""
Wasserstein Pipeline.py
=======================
Project: Quant Trader Lab - Optimal Transport
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for analyzing regime shifts using 
    Wasserstein Distance and Optimal Transport (Sinkhorn).

    It calculates the geometric distance between two probability distributions
    (e.g., Backtest vs. Live returns) to detect drift.

    Pipeline Steps:
    1.  **Data Acquisition**: Generates synthetic regime data (User can plug in API).
    2.  **Transport Engine**: Solves the Earth Mover's Distance & Sinkhorn Divergence.
    3.  **Visualization**: Generates a high-fidelity static dashboard with transport vectors.

    NOTE: Video rendering removed for pipeline efficiency.

Dependencies:
    pip install numpy matplotlib scipy pot
    (pot is 'Python Optimal Transport')
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ot  # POT: Python Optimal Transport library
from scipy.stats import wasserstein_distance
from matplotlib.gridspec import GridSpec
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "SAMPLES": 500,
    "SEED": 42,
    
    # Transport Parameters
    "SINKHORN_REG": 0.1, # Regularization term (epsilon)
    "NUM_ARROWS": 40,    # Number of transport vectors to visualize
    
    # Output
    "OUTPUT_IMAGE": "wasserstein_analysis_static.png",
    
    # Aesthetics
    "COLOR_BG": '#0e0e0e', # Dark background
    "COLOR_BT": '#58a6ff', # Backtest Blue
    "COLOR_LIVE": '#f0883e', # Live Orange
    "COLOR_TEXT": '#b0b0b0',
    "FONT_MAIN": 14,
    "FONT_FORMULA": 16
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_data():
    """
    Generates or fetches market distributions.
    """
    log("[Data] Initializing data module...")
    
    # -------------------------------------------------------------------------
    # [USER INPUT REQUIRED] - API INTEGRATION TEMPLATE
    # -------------------------------------------------------------------------
    # To use your own data (e.g., yfinance, Alpaca, CCXT), uncomment and adapt:
    #
    # import yfinance as yf
    #
    # def get_returns(ticker, start, end):
    #     df = yf.download(ticker, start=start, end=end)
    #     return df['Close'].pct_change().dropna().values
    #
    # backtest_data = get_returns('BTC-USD', '2022-01-01', '2023-01-01')
    # live_data = get_returns('BTC-USD', '2023-01-01', '2024-01-01')
    # -------------------------------------------------------------------------

    log("[Data] Using Synthetic Regime Shift Data.")
    np.random.seed(CONFIG['SEED'])
    n = CONFIG['SAMPLES']
    
    # Synthetic Data Generation
    # 1D Returns
    bt_1d = np.random.normal(0, 0.02, n)
    live_1d = np.random.normal(0.005, 0.03, n) # Drifted
    
    # 2D Features (e.g., Returns vs Volatility)
    bt_2d = np.random.randn(n, 2)
    live_2d = np.random.randn(n, 2) + 1.5 # Significant shift
    
    return bt_1d, live_1d, bt_2d, live_2d

# --- MODULE 2: TRANSPORT ENGINE ---

def compute_optimal_transport(bt_1d, live_1d, bt_2d, live_2d):
    """
    Calculates Wasserstein metrics and Transport Plan.
    """
    log("[Engine] Computing Earth Mover's Distance...")
    
    # 1. Scalar Wasserstein (1D)
    w_dist = wasserstein_distance(bt_1d, live_1d)
    log(f"[Engine] 1D Wasserstein Distance: {w_dist:.5f}")
    
    # 2. Sinkhorn Divergence (2D)
    # Uniform weights
    n = len(bt_2d)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    
    # Cost matrix (Euclidean distance squared)
    M = ot.dist(bt_2d, live_2d)
    M /= M.max() # Normalization for stability
    
    log(f"[Engine] Solving Sinkhorn (reg={CONFIG['SINKHORN_REG']})...")
    sinkhorn_dist = ot.sinkhorn2(a, b, M, CONFIG['SINKHORN_REG'])
    log(f"[Engine] 2D Sinkhorn Divergence: {sinkhorn_dist:.5f}")
    
    # 3. Compute Transport Map (for visualization)
    # We solve exact EMD on a subset for clean arrows
    k = CONFIG['NUM_ARROWS']
    idx_b = np.random.choice(n, k, replace=False)
    idx_l = np.random.choice(n, k, replace=False)
    
    sub_b = bt_2d[idx_b]
    sub_l = live_2d[idx_l]
    
    M_sub = ot.dist(sub_b, sub_l)
    ab_sub = np.ones((k,)) / k
    G_sub = ot.emd(ab_sub, ab_sub, M_sub)
    
    # Extract connections
    arrows = []
    for i in range(k):
        for j in range(k):
            if G_sub[i, j] > 1e-5:
                arrows.append((sub_b[i], sub_l[j]))
                
    return w_dist, sinkhorn_dist, bt_2d, live_2d, arrows

# --- MODULE 3: VISUALIZATION ---

def render_static_dashboard(w_dist, sinkhorn, bt_2d, live_2d, arrows):
    """
    Generates the static high-fidelity image.
    """
    log("[Visual] Rendering static dashboard...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(CONFIG['COLOR_BG'])
    
    # Grid Layout
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(CONFIG['COLOR_BG'])
    
    # 1. Scatter Clusters
    ax.scatter(bt_2d[:, 0], bt_2d[:, 1], color=CONFIG['COLOR_BT'], alpha=0.5, s=20, label='Backtest Distribution')
    ax.scatter(live_2d[:, 0], live_2d[:, 1], color=CONFIG['COLOR_LIVE'], alpha=0.5, s=20, label='Live Distribution')
    
    # 2. Transport Vectors
    log(f"[Visual] Drawing {len(arrows)} transport vectors...")
    for start, end in arrows:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='white', alpha=0.6, lw=1, linestyle='--', zorder=1)
        # Add arrow head? simple plot is cleaner for "flow"
        
    # 3. Formulas & Metrics (Overlay)
    
    # Top: Wasserstein Formula
    formula_w = r"$W_p(\mu, \nu) = \dst\left( \inf_{\gamma} \int d(x, y)^p d\gamma(x, y) \right)^{1/p}$"
    # Note: \dst might not work without amsmath, assume standard latex
    formula_w = r"$W_p(\mu, \nu) = \left( \inf_{\gamma} \int d(x, y)^p d\gamma(x, y) \right)^{1/p}$"
    
    ax.text(0.5, 0.95, formula_w, transform=ax.transAxes, 
            ha='center', va='top', color=CONFIG['COLOR_TEXT'], fontsize=24, alpha=0.4)
            
    # Bottom: Sinkhorn Formula
    formula_s = r"$L_\lambda(P) = \langle C, P \rangle - \epsilon H(P)$"
    ax.text(0.5, 0.05, formula_s, transform=ax.transAxes, 
            ha='center', va='bottom', color=CONFIG['COLOR_TEXT'], fontsize=20, alpha=0.3)
            
    # Titles & Stats
    ax.set_title(f"Optimal Transport Map: Drift Detection", color='white', fontsize=18, pad=20)
    
    stats = (f"1D Wasserstein: {w_dist:.5f}\n"
             f"2D Sinkhorn:    {sinkhorn:.5f}")
    
    props = dict(boxstyle='round', facecolor='#1f1f1f', alpha=0.8, edgecolor='#333')
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', color='white', bbox=props, family='monospace')
            
    # Styling
    ax.legend(loc='upper right', facecolor='#1f1f1f', edgecolor='#333', fontsize=10)
    ax.axis('off') # Clean look
    
    # Save
    out_path = CONFIG['OUTPUT_IMAGE']
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=CONFIG['COLOR_BG'])
    plt.close(fig)
    log(f"[Visual] Dashboard saved to {out_path}")

# --- MAIN ---

def main():
    log("=== WASSERSTEIN PIPELINE ===")
    
    # 1. Data
    bt_1d, live_1d, bt_2d, live_2d = fetch_data()
    
    # 2. Engine
    w_dist, sinkhorn, bt_2d, live_2d, arrows = compute_optimal_transport(bt_1d, live_1d, bt_2d, live_2d)
    
    # 3. Visual
    render_static_dashboard(w_dist, sinkhorn, bt_2d, live_2d, arrows)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
