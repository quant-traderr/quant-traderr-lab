"""
Lempel-Ziv Pipeline.py
======================
Project: Quant Trader Lab - Market Phase Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for Lempel-Ziv (LZ) Complexity Analysis.

    It analyzes the algorithmic information content (complexity) of price 
    sequences to detect regime shifts (Predictable vs. Chaotic).

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data (BTC-USD).
    2.  **Engine**: Discretizes returns into binary sequences and computes Rolling LZ Complexity.
    3.  **Visualization**: Generates a high-quality 3D Phase Space snapshot.

    NOTE: Video rendering has been removed for pipeline efficiency.

Dependencies:
    pip install numpy pandas yfinance matplotlib
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER": "BTC-USD",
    "LOOKBACK_YEARS": 2,
    "INTERVAL": "1d",
    
    # Analysis
    "WINDOW_SIZE": 30,
    "SAMPLE_SIZE": 300, # Number of recent points to analyze
    
    # Output
    "OUTPUT_IMAGE": "lempel_ziv_static.png",
    
    # Aesthetics (Bloomberg Style)
    "COLOR_BG": '#000000',
    "COLOR_GRID": '#333333',
    "COLOR_PRICE": '#FF9800', # Orange
    "COLOR_ENTROPY": '#00BFFF', # Blue
    "COLOR_TEXT": '#E0E0E0'
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches historical market data.
    """
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['LOOKBACK_YEARS']}y)...")
    
    try:
        data = yf.download(CONFIG['TICKER'], period=f"{CONFIG['LOOKBACK_YEARS']}y", interval=CONFIG['INTERVAL'], progress=False)
        
        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex): 
            data = data.xs(CONFIG['TICKER'], axis=1, level=1)
            
        if data.empty:
            raise ValueError("No data returned from yfinance.")
            
        prices = data['Close'].values
        
        # Trim to desired sample size
        if len(prices) > CONFIG['SAMPLE_SIZE']:
            prices = prices[-CONFIG['SAMPLE_SIZE']:]
            
        log(f"[Data] Loaded {len(prices)} data points.")
        return prices
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        # Fallback dummy data
        log("[Warning] Using synthetic data.")
        np.random.seed(42)
        n_points = CONFIG['SAMPLE_SIZE']
        t = np.linspace(0, 10, n_points)
        periodic = np.sin(t * 2) + np.random.normal(0, 0.05, n_points//2)
        chaotic = np.random.normal(0, 0.5, n_points - n_points//2)
        return np.concatenate([periodic, chaotic]).cumsum()

# --- MODULE 2: ANALYSIS ENGINE ---

class LempelZivEngine:
    def __init__(self, prices):
        self.prices = prices
        self.window = CONFIG['WINDOW_SIZE']
        
    def _calculate_complexity(self, binary_sequence):
        """
        Calculates the LZ complexity count.
        """
        u, v, w = 0, 1, 1
        complexity = 1
        n = len(binary_sequence)
        while v + w <= n:
            if binary_sequence[v:v+w] in binary_sequence[u:v]:
                w += 1
            else:
                u, v, w = 0, v + w, 1
                complexity += 1
        return complexity

    def run(self):
        """
        Executes Rolling LZ Complexity Analysis.
        """
        log("[Engine] Starting LZ Complexity Analysis...")
        start_time = time.time()
        
        # 1. Discretize Returns -> Binary Sequence
        # 1 = Price Up, 0 = Price Down
        binary_seq = (np.diff(self.prices) > 0).astype(int).astype(str)
        full_binary_str = "".join(binary_seq.tolist())
        
        # 2. Rolling Window Calculation
        lz_scores = []
        # Pad initial window
        for _ in range(self.window):
            lz_scores.append(1) # Placeholder for initial points
            
        total_steps = len(full_binary_str) - self.window
        for i in range(total_steps):
            chunk = full_binary_str[i:i+self.window]
            c = self._calculate_complexity(chunk)
            lz_scores.append(c)
            
        # Align lengths (Diff reduces length by 1)
        # We want arrays to match self.prices length
        # Current lz_scores len ~= len(prices) - 1
        # Re-aligning roughly
        lz_array = np.array(lz_scores)
        
        # Pad to match price array length exactly if needed
        if len(lz_array) < len(self.prices):
            pad = len(self.prices) - len(lz_array)
            lz_array = np.pad(lz_array, (pad, 0), 'edge')
            
        duration = time.time() - start_time
        log(f"[Engine] Analysis complete in {duration:.2f}s.")
        
        return lz_array

# --- MODULE 3: VISUALIZATION ---

def plot_multicolor_line_3d(ax, x, y, z, c, cmap='plasma', lw=2):
    """
    Helper for 3D multicolor line.
    """
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(c.min(), c.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=lw)
    lc.set_array(c)
    ax.add_collection(lc)
    return lc

def analyze_and_visualize(prices, lz_scores):
    """
    Generates a static 3D phase space visualization.
    """
    log("[Visual] Generating static 3D snapshot...")
    
    # Setup Figure
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(CONFIG['COLOR_BG'])
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(CONFIG['COLOR_BG'])
    
    # Aesthetics
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.grid(True, color=CONFIG['COLOR_GRID'], linestyle=':', linewidth=0.5)
    
    # Data
    xs = np.arange(len(prices))
    ys = prices
    zs = lz_scores
    
    # Plot Ribbon
    plot_multicolor_line_3d(ax, xs, ys, zs, lz_scores, cmap='cool', lw=2)
    
    # Limits
    ax.set_xlim(0, len(prices))
    ax.set_ylim(prices.min()*0.99, prices.max()*1.01)
    ax.set_zlim(0, lz_scores.max() + 2)
    
    # Labels
    ax.set_xlabel('TIME', color=CONFIG['COLOR_TEXT'], fontsize=9, labelpad=10)
    ax.set_ylabel('PRICE', color=CONFIG['COLOR_PRICE'], fontsize=9, labelpad=10)
    ax.set_zlabel('LZ ENTROPY', color=CONFIG['COLOR_ENTROPY'], fontsize=9, labelpad=10)
    
    ax.tick_params(axis='x', colors=CONFIG['COLOR_TEXT'], labelsize=7)
    ax.tick_params(axis='y', colors=CONFIG['COLOR_PRICE'], labelsize=7)
    ax.tick_params(axis='z', colors=CONFIG['COLOR_ENTROPY'], labelsize=7)
    
    # Title
    ax.text2D(0.05, 0.95, f"LZ COMPLEXITY: {CONFIG['TICKER']}", transform=ax.transAxes, 
              color='white', fontsize=14, fontname='monospace', fontweight='bold')
    ax.text2D(0.05, 0.92, "Regime Detection (Predictable vs Chaotic)", transform=ax.transAxes,
              color='gray', fontsize=10, fontname='monospace')
    
    # View Angle
    ax.view_init(elev=25, azim=-60)
    
    # Save
    out_path = os.path.join(os.path.dirname(__file__), CONFIG['OUTPUT_IMAGE'])
    plt.savefig(out_path, facecolor=CONFIG['COLOR_BG'], dpi=150)
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")

# --- MAIN ---

def main():
    log("=== LEMPEL-ZIV PIPELINE ===")
    
    # 1. Data
    prices = fetch_market_data()
    
    # 2. Engine
    engine = LempelZivEngine(prices)
    lz_scores = engine.run()
    
    # 3. Output
    analyze_and_visualize(prices, lz_scores)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
