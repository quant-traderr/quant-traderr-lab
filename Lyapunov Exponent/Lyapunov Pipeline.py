"""
Lyapunov_Pipeline.py
====================
Project: Quant Trader Lab - Chaos Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for analyzing financial market chaos using 
    Phase Space Reconstruction and the Method of Analogues.
    
    It verifies if current market microstructure resembles a historical period
    by embedding price series into a 3D Phase Space (Time Delay Embedding)
    and searching for "Nearest Neighbors" (Analogues).

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches strict real-market data (BTC-USD) via `yfinance`.
    2.  **Phase Space Reconstruction**: Embeds the time series into R^3 using Time Delay (Tau).
    3.  **Method of Analogues**: Uses KD-Tree Nearest Neighbor search to find historical matches.
    4.  **Visualization**: (Removed) The proprietary rendering pipeline has been removed. 
        This version outputs analysis metrics to the console only.

Dependencies:
    pip install numpy pandas yfinance scipy scikit-learn

Usage:
    python Lyapunov_Pipeline.py
"""

import os
import sys
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER": "BTC-USD",
    "PERIOD": "59d",       # Max for 5m data
    "INTERVAL": "5m",
    "LOOKBACK_POINTS": 3000, # How many points to analyze for the "Current" trajectory
    
    # Embedding (Phase Space)
    "TAU": 12,             # Delay (12 * 5m = 60m = 1 hour)
    "DIM": 3,              # Embedding Dimension
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches market data from YFinance.
    Returns: numpy array of normalized prices.
    """
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']}, {CONFIG['INTERVAL']})...")
    
    try:
        df = yf.download(CONFIG['TICKER'], period=CONFIG['PERIOD'], interval=CONFIG['INTERVAL'], progress=False)
        
        if df.empty:
            raise ValueError("No data returned from yfinance.")
            
        # Handle MultiIndex and Column extraction
        prices = None
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try Close
                prices = df.xs('Close', axis=1, level=0, drop_level=True)
                if prices.empty: raise KeyError
            except:
                prices = df.iloc[:, 0] # Fallback
        else:
            # Standard
            if 'Close' in df.columns:
                prices = df['Close']
            else:
                prices = df.iloc[:, 0]

        # Final cleanup
        if isinstance(prices, pd.DataFrame):
            # If multiple tickers, pick the one we asked for
            if CONFIG['TICKER'] in prices.columns:
                prices = prices[CONFIG['TICKER']]
            else:
                prices = prices.iloc[:, 0]

        prices = prices.dropna().values
        
        # Normalize (Z-Score) for shape comparison
        # We store the scaler params if we wanted to denormalize, but for phase space it's fine.
        prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-9)
        
        log(f"[Data] Loaded {len(prices_norm)} data points.")
        return prices_norm

    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        return None

# --- MODULE 2: MATH (CHAOS) ---

def embed_time_delay(series, dim, tau):
    """
    Creates a Time Delay Embedding of the series.
    Returns array of shape (N_vectors, dim).
    """
    N = len(series)
    M = N - (dim - 1) * tau
    
    if M <= 0:
        raise ValueError(f"Series too short ({N}) for Dim {dim} / Tau {tau}.")
        
    embedded = np.zeros((M, dim))
    for d in range(dim):
        # x(t), x(t+tau), x(t+2tau)...
        # We fill column 'd' with the shifted series
        st = d * tau
        ed = st + M
        embedded[:, d] = series[st:ed]
        
    return embedded

def perform_method_of_analogues(series):
    """
    1. Embeds the series.
    2. Identifies the 'Current' trajectory (recent history).
    3. Finds the 'Analogue' trajectory (nearest historical neighbor).
    """
    dim = CONFIG['DIM']
    tau = CONFIG['TAU']
    lookback = CONFIG['LOOKBACK_POINTS']
    
    # 1. Embed
    vectors = embed_time_delay(series, dim, tau)
    
    if len(vectors) < lookback * 2:
        log("[Error] Data insufficient for analysis length.")
        return None, None
        
    # 2. Define 'Subject' (Current Trajectory)
    # The last 'lookback' points
    subject_vectors = vectors[-lookback:]
    
    # 3. Define 'Library' (History)
    # Exclude the subject and a safety buffer (Theiler window) to prevent identifying 
    # the subject's immediate past as its own neighbor.
    safety_buffer = lookback 
    search_end_idx = len(vectors) - lookback - safety_buffer
    
    if search_end_idx < 100:
        log("[Error] History too short to find analogues.")
        return None, None
        
    search_space = vectors[:search_end_idx]
    
    # 4. Nearest Neighbor Search
    # We look for a match to the START of the subject trajectory
    query_point = subject_vectors[0].reshape(1, -1)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_space)
    distances, indices = nbrs.kneighbors(query_point)
    
    match_idx = indices[0][0]
    match_dist = distances[0][0]
    
    # Extract Analogue
    analogue_vectors = vectors[match_idx : match_idx + lookback]
    
    # Distance Metric (Log divergence not needed for simple visual analogue check)
    log(f"[Analysis] Found Analogue.")
    log(f"           Distance: {match_dist:.5f}")
    
    days_ago = (len(vectors) - match_idx) * 5 / 60 / 24
    log(f"           Time Index: {match_idx} (approx {days_ago:.1f} days ago)")
    
    return match_dist, match_idx

# --- MODULE 3: ANALYSIS OUTPUT ---
# NOTE: The proprietary rendering and visualization pipeline has been removed from this file.
# This pure-analysis version outputs metrics to the console only.

def report_findings(match_dist, match_idx, total_vectors):
    """
    Prints the analysis results to the console.
    """
    days_ago = (total_vectors - match_idx) * 5 / 60 / 24
    
    log("=== ANALYSIS RESULT ===")
    log(f"Current Market Microstructure (Last {CONFIG['LOOKBACK_POINTS']} points)")
    log(f"Most Similar Historical Period: Index {match_idx}")
    log(f"Time Delta: ~{days_ago:.2f} days ago")
    log(f"Euclidean Distance (Similarity Score): {match_dist:.5f}")
    
    if match_dist < 0.05:
         log(">> High Similarity Detected (Strong Analogue)")
    else:
         log(">> Low Similarity (Weak Analogue)")

# --- MAIN ---

def main():
    log("=== LYAPUNOV ANALYSIS PIPELINE ===")
    
    # 1. Fetch
    data = fetch_market_data()
    if data is None: return
    
    # 2. Analyze
    # We modified perform_method_of_analogues to return metrics instead of vectors if we don't need to plot
    # But let's just use the existing return and extract info
    match_dist, match_idx = perform_method_of_analogues(data)
    
    if match_dist is not None:
        report_findings(match_dist, match_idx, len(data))
        
    log("=== DONE ===")

if __name__ == "__main__":
    main()
