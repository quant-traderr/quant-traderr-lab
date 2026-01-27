"""
FFT Pipeline.py
===============
Project: Quant Trader Lab - Fourier Transform Time Series Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for decomposing financial time series using 
    Fast Fourier Transform (FFT) with a ROLLING WINDOW approach.

    This ensures NO LOOKAHEAD BIAS. The filtered signal at time t is calculated
    using ONLY data available up to time t (window [t-N : t]).

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches stock data via yfinance
    2.  **Rolling FFT**: Slides a window across the data
    3.  **Decomposition**: Inside each window, detrends and computes FFT
    4.  **Filtering**: Selects top N frequencies and reconstructs
    5.  **Point Estimation**: Takes the last point of the reconstruction as the value for time t
    6.  **Analysis**: Aggregates cycle statistics over the rolling windows

    NOTE: This is a Pure Analysis Pipeline. Visualization rendering has been removed.

Dependencies:
    pip install numpy pandas yfinance tqdm
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER": "NVDA", 
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-01-01",
    
    # Analysis
    "TOP_N_COMPONENTS": 10, # Number of frequency components to keep
    "WINDOW_SIZE": 126,     # Rolling window size in days (approx 6 months)
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data(ticker, start_date, end_date):
    """
    Fetches stock data and returns dates and closing prices.
    """
    log(f"[Data] Fetching {ticker} ({start_date} to {end_date})...")
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=True)
        
        if data.empty:
            raise ValueError("No data returned from yfinance.")
            
        # Robustly handle Dataframe to get just the Close series
        if 'Close' in data.columns:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.get_level_values(1): 
                     y = data['Close'][ticker].values
                else:
                    y = data['Close'].iloc[:, 0].values
            else:
                y = data['Close'].values
        else:
             y = data.iloc[:, 0].values 
             
        y = y.flatten()
        y = y[~np.isnan(y)]
        
        log(f"[Data] Successfully loaded {len(y)} data points")
        
        return y
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        return None

# --- MODULE 2: ROLLING FFT CORE ---

def compute_rolling_fft(y, window_size=126, top_n_components=10):
    """
    Performs Rolling FFT Analysis.
    For each time step t (from window_size to N), computes FFT on y[t-window:t].
    Returns the reconstructed time series.
    """
    log(f"[FFT] Starting Rolling Analysis (Window: {window_size}, Top-N: {top_n_components})...")
    
    n_samples = len(y)
    
    # Initialize output arrays with NaN for the startup period
    y_reconstructed_rolling = np.full(n_samples, np.nan)
    cycle_history = [] # To store dominant cycle per window
    
    start_time = time.time()
    
    # Iterate through the data
    # We can only compute for t >= window_size
    for t in tqdm(range(window_size, n_samples + 1), desc="[FFT] Rolling Window"):
        # 1. Slice Window
        # Window is from t-window_size to t (exclusive of t in slice arithmetic if we want index t-1, 
        # but here we want the window leading UP TO the current point)
        # Let's say we want to predict/filter the point at index t-1.
        # window indices: [t-window_size ... t-1]
        
        window = y[t-window_size : t]
        x_window = np.arange(len(window))
        
        # 2. Detrend
        poly_coeffs = np.polyfit(x_window, window, 1)
        trend_line = np.polyval(poly_coeffs, x_window)
        window_detrended = window - trend_line
        
        # 3. Compute FFT
        fft_coeffs = np.fft.fft(window_detrended)
        freqs = np.fft.fftfreq(len(window))
        
        # 4. Filter Top N
        amplitudes = np.abs(fft_coeffs)
        # Sort desc
        sorted_indices = np.argsort(amplitudes)[::-1]
        
        # Keep top N
        filtered_coeffs = np.zeros_like(fft_coeffs)
        filtered_coeffs[sorted_indices[:top_n_components]] = fft_coeffs[sorted_indices[:top_n_components]]
        
        # 5. Inverse FFT
        reconstructed_window_detrended = np.fft.ifft(filtered_coeffs).real
        reconstructed_window = reconstructed_window_detrended + trend_line
        
        # 6. Store Result
        # The result for index t-1 is the LAST point of this reconstructed window
        y_reconstructed_rolling[t-1] = reconstructed_window[-1]
        
        # 7. Store Dominant Cycle (for Analysis)
        # Find strongest non-zero frequency
        # exclude 0 (DC)
        valid_indices = [idx for idx in sorted_indices if freqs[idx] > 0]
        if valid_indices:
            top_idx = valid_indices[0] # Strongest frequency
            top_freq = freqs[top_idx]
            top_cycle = 1.0 / top_freq
            cycle_history.append(top_cycle)
        else:
            cycle_history.append(np.nan)

    duration = time.time() - start_time
    log(f"[FFT] Rolling computation complete in {duration:.2f}s")
    
    return {
        'original': y,
        'reconstructed': y_reconstructed_rolling,
        'noise': y - y_reconstructed_rolling, # Will be NaN for first window_size
        'cycle_history': np.array(cycle_history) # Aligned with t=window_size..end
    }

# --- MODULE 3: ANALYSIS ---

def analyze_fft_results(results, window_size):
    """
    Analyzes the rolling FFT output.
    """
    log("=== ROLLING FFT RESULTS ===")
    
    # 1. Quality Metrics (on valid data only)
    valid_mask = ~np.isnan(results['reconstructed'])
    y_valid = results['original'][valid_mask]
    recon_valid = results['reconstructed'][valid_mask]
    noise_valid = results['noise'][valid_mask]
    
    if len(y_valid) == 0:
        log("No valid data points generated.")
        return

    # Signal-to-Noise
    signal_power = np.mean(recon_valid ** 2)
    noise_power = np.mean(noise_valid ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Volatility Reduction
    orig_vol = np.std(y_valid)
    noise_vol = np.std(noise_valid)
    reduction = (1 - noise_vol/orig_vol)
    
    log(f"Process: Rolling Window ({window_size} days)")
    log(f"Analyzed Points: {len(y_valid)}")
    log(f"Signal-to-Noise Ratio: {snr_db:.2f} dB")
    log(f"Noise Reduction (Fit): {reduction:.2%}")
    
    # 2. Cycle Stability Analysis
    cycles = results['cycle_history']
    cycles = cycles[~np.isnan(cycles)]
    
    if len(cycles) > 0:
        avg_cycle = np.mean(cycles)
        median_cycle = np.median(cycles)
        std_cycle = np.std(cycles)
        
        log("\n=== DOMINANT CYCLE DYNAMICS ===")
        log(f"Average Dominant Cycle: {avg_cycle:.1f} days")
        log(f"Median Dominant Cycle:  {median_cycle:.1f} days")
        log(f"Cycle Stability (Std):  {std_cycle:.1f} days")
        log(f"Latest Detected Cycle:  {cycles[-1]:.1f} days")
    else:
        log("No cycles detected.")


# --- MAIN ---

def main():
    log("=== UNBIASED FFT PIPELINE (ROLLING WINDOW) ===")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Data Acquisition
    y = fetch_market_data(CONFIG['TICKER'], CONFIG['START_DATE'], CONFIG['END_DATE'])
    if y is None:
        log("[Error] Cannot proceed without data.")
        return
        
    # 2. Rolling FFT Analysis
    results = compute_rolling_fft(
        y, 
        window_size=CONFIG['WINDOW_SIZE'], 
        top_n_components=CONFIG['TOP_N_COMPONENTS']
    )
    
    # 3. Analysis & Metrics
    analyze_fft_results(results, CONFIG['WINDOW_SIZE'])
    
    log("\n=== PIPELINE FINISHED ===")
    
    return results

if __name__ == "__main__":
    main()
