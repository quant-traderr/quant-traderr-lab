"""
Monte Carlo Pipeline.py
=======================
Project: Quant Trader Lab - Portfolio Optimization
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for Monte Carlo Asset Price Simulation.
    
    It uses a Geometric Brownian Motion (GBM) model calibrated on historical 
    volatility and drift to project future price paths.

    Note: This is proprietary info.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data (BTC-USD) via yfinance.
    2.  **Simulation Engine**: Runs vectorized Monte Carlo simulations.
    3.  **Analysis**: Computes VaR, Expected Returns, and Confidence Intervals.
    4.  **Static Visualization**: Generates a high-quality snapshot of the pathways.

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
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER": "BTC-USD",
    "LOOKBACK_YEARS": 1,
    "INTERVAL": "1d",
    
    # Simulation
    "SIMULATIONS": 25000,
    "DAYS_TO_PROJECT": 300,
    "START_CAPITAL": 10000,
    "RANDOM_SEED": None, # Set integer for reproducibility
    
    # Output
    "OUTPUT_IMAGE": "monte_carlo_static.png",
    
    # Aesthetics
    "COLOR_BG": '#0e0e0e',
    "COLOR_LINE": (0.0, 1.0, 1.0, 0.01), # Cyan, very low alpha
    "COLOR_MEAN": '#00aaff',
    "COLOR_BEST": '#00ff41',
    "COLOR_WORST": '#ff0055',
    "COLOR_GRID": '#1f1f1f'
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches historical market data to calibrate drift and volatility.
    """
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['LOOKBACK_YEARS']}y)...")
    
    try:
        data = yf.download(CONFIG['TICKER'], period=f"{CONFIG['LOOKBACK_YEARS']}y", interval=CONFIG['INTERVAL'], progress=False)
        
        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex): 
            data = data.xs(CONFIG['TICKER'], axis=1, level=1)
            
        if data.empty:
            raise ValueError("No data returned from yfinance.")
            
        prices = data['Close']
        returns = prices.pct_change().dropna().values
        
        log(f"[Data] Loaded {len(returns)} days of returns.")
        return returns
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        # Fallback dummy data for testing connectivity
        log("[Warning] Using dummy gaussian data.")
        return np.random.normal(0, 0.02, 252)

# --- MODULE 2: SIMULATION ENGINE ---

class MonteCarloEngine:
    def __init__(self, returns):
        self.returns = returns
        self.simulations = CONFIG['SIMULATIONS']
        self.days = CONFIG['DAYS_TO_PROJECT']
        self.start_capital = CONFIG['START_CAPITAL']
        
    def run(self):
        """
        Executes the Monte Carlo Simulation using Bootstrap Resampling 
        (sampling from historical returns).
        """
        log(f"[Simulation] Starting {self.simulations} simulations for {self.days} days...")
        
        start_time = time.time()
        
        # Vectorized Bootstrap Resampling
        # We sample random daily returns from history to project forward
        random_idx = np.random.randint(0, len(self.returns), size=(self.days, self.simulations))
        sim_returns = self.returns[random_idx]
        
        # Calculate Cumulative Returns
        # Eq_t = Eq_0 * product(1 + r_t)
        sim_paths = self.start_capital * (1 + sim_returns).cumprod(axis=0)
        
        # Add start point (Day 0)
        sim_paths = np.vstack([np.full((1, self.simulations), self.start_capital), sim_paths])
        
        duration = time.time() - start_time
        log(f"[Simulation] Completed in {duration:.2f}s.")
        
        return sim_paths

# --- MODULE 3: ANALYSIS & VISUALIZATION ---

def analyze_and_visualize(sim_paths):
    """
    Calculates stats and generates a static summary image.
    """
    days = sim_paths.shape[0]
    final_values = sim_paths[-1, :]
    
    # Statistics
    mean_path = np.mean(sim_paths, axis=1)
    upper_path = np.percentile(sim_paths, 95, axis=1) # 95th percentile (Best Case)
    lower_path = np.percentile(sim_paths, 5, axis=1)  # 5th percentile (VaR)
    
    median_profit = np.median(final_values) - CONFIG['START_CAPITAL']
    exp_return = (np.mean(final_values) / CONFIG['START_CAPITAL']) - 1
    
    log("=== RESULTS SUMMARY ===")
    log(f"Projected Median Profit: ${median_profit:,.2f}")
    log(f"Expected Return (Mean): {exp_return:.2%}")
    log(f"95% Best Case Equity: ${upper_path[-1]:,.2f}")
    log(f"95% Worst Case Equity: ${lower_path[-1]:,.2f}")
    
    # --- VISUALIZATION ---
    log("[Visual] Generating static snapshot...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(CONFIG['COLOR_BG'])
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])
    
    # 1. Trajectories Plot
    ax_main.set_facecolor(CONFIG['COLOR_BG'])
    ax_main.grid(True, color=CONFIG['COLOR_GRID'], linestyle='--')
    ax_main.set_title(f"Monte Carlo Simulation: {CONFIG['TICKER']} ({CONFIG['SIMULATIONS']} runs)", 
                      color='white', fontsize=16, weight='bold', pad=20)
    ax_main.set_ylabel("Projected Equity ($)", color='gray')
    ax_main.tick_params(colors='gray')
    for spine in ax_main.spines.values(): spine.set_edgecolor('#333')
    
    ax_main.set_xlim(0, days-1)
    ax_main.set_ylim(0, np.max(upper_path) * 1.1)
    
    # Plot a subset of paths to avoid memory overload if sims > 10k in matplotlib
    # Plotting 25k lines individually is heavy, so we plot a representative subset or use LineCollection
    # For a static image, plotting first 2000 is usually enough to show density without crashing
    step = max(1, CONFIG['SIMULATIONS'] // 2000)
    ax_main.plot(sim_paths[:, ::step], color=CONFIG['COLOR_LINE'], linewidth=1.0)
    
    # Stats Lines
    x_axis = np.arange(days)
    ax_main.plot(x_axis, mean_path, color=CONFIG['COLOR_MEAN'], linewidth=2.5, label='Average Projection')
    ax_main.plot(x_axis, upper_path, color=CONFIG['COLOR_BEST'], linewidth=1.5, linestyle='--', label='95% Best Case')
    ax_main.plot(x_axis, lower_path, color=CONFIG['COLOR_WORST'], linewidth=1.5, linestyle='--', label='95% VaR (Worst Case)')
    
    leg = ax_main.legend(loc='upper left', facecolor=CONFIG['COLOR_BG'], edgecolor='#333')
    for text in leg.get_texts(): text.set_color('white')
    
    # 2. Histogram Plot
    ax_hist.set_facecolor(CONFIG['COLOR_BG'])
    ax_hist.grid(True, color=CONFIG['COLOR_GRID'], linestyle=':')
    ax_hist.set_xlabel("Final Equity ($)", color='gray')
    ax_hist.set_ylabel("Frequency", color='gray')
    ax_hist.tick_params(colors='gray')
    for spine in ax_hist.spines.values(): spine.set_edgecolor('#333')
    ax_hist.set_title(f"Final Distribution @ Day {days-1}", color='white', fontsize=10)
    
    ax_hist.hist(final_values, bins=100, color='#5a7d9a', edgecolor=CONFIG['COLOR_BG'], alpha=0.9)
    
    # Save
    out_path = os.path.join(os.path.dirname(__file__), CONFIG['OUTPUT_IMAGE'])
    plt.savefig(out_path, facecolor=CONFIG['COLOR_BG'], dpi=100)
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")

# --- MAIN ---

def main():
    log("=== MONTE CARLO PIPELINE ===")
    
    # 1. Data
    returns = fetch_market_data()
    
    # 2. Simulation
    engine = MonteCarloEngine(returns)
    paths = engine.run()
    
    # 3. Analysis & Output
    analyze_and_visualize(paths)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
