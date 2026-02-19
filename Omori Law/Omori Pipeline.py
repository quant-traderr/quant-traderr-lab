"""
Omori Pipeline.py
=================
Project: Quant Trader Lab - Market Crash Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for analyzing financial market aftershocks using
    the Omori Law (originally from seismology).
    
    The pipeline identifies a market crash and models the decay of volatility
    (absolute returns exceeding thresholds) as a power-law process.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data (e.g., S&P 500, BTC) via yfinance.
    2.  **Event Detection**: Identifies the main crash event and subsequent "aftershocks".
    3.  **Rate Analysis**: Calculates the rate of volatility events over time using multiple thresholds.
    4.  **Model Fitting**: Fits the Omori Power Law n(t) = K * t^(-p) to the decay rate.
    5.  **Static Visualization**: Generates a high-quality dual-panel chart (Price + Omori Decay).

    Usage:
    - Configure the ASSET_TICKER and CRASH_DATE in the CONFIG dictionary below.
    - Run the script to generate a static PNG visualization.

Dependencies:
    pip install numpy pandas yfinance matplotlib scipy
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import os

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # --------------------------------------------------------------------------
    # [USER CONFIGURATION]
    # Change the Ticker and Crash Date to analyze different assets/events.
    # --------------------------------------------------------------------------
    
    # Asset to Analyze
    "ASSET_TICKER": "^GSPC",   # S&P 500: ^GSPC, Bitcoin: BTC-USD, Ethereum: ETH-USD
    
    # API Key Handling (Optional)
    # If using a data provider requiring an API key (e.g., AlphaVantage, Polygon), 
    # set it here and modify the fetch_market_data function accordingly.
    "API_KEY": None,           # e.g., "YOUR_API_KEY_HERE"
    
    # Analysis Parameters
    "START_DATE": '2019-06-01',
    "END_DATE": '2021-12-01',
    "CRASH_DATE": '2020-03-23', # The date of the main shock (lowest point or max volatility)
    
    # Volatility Thresholds (Absolute % Returns) for Aftershock Definition
    "THRESHOLDS": [0.5, 1.0, 1.5, 2.0, 3.0], 
    
    # Output
    "OUTPUT_FILENAME": "omori_analysis_static.png",
    
    # Aesthetics (Dark Mode Theme)
    "THEME": {
        "BG":       '#0a0a0a',
        "PANEL_BG": '#0e0e0e',
        "GRID":     '#1a1a1a',
        "TEXT":     '#c0c0c0',
        "ACCENT_1": '#ffb700', # Amber
        "ACCENT_2": '#00e5ff', # Cyan
        "WHITE":    '#ffffff',
        "COLORS":   ['#ff6d00', '#e91e63', '#ab47bc', '#29b6f6', '#00e676'], # For thresholds
        "MARKERS":  ['o',       's',       'D',       '^',       'x']
    }
}

# --- UTILS ---

def log(msg):
    """Simple logger with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches historical market data.
    """
    ticker = CONFIG["ASSET_TICKER"]
    start = CONFIG["START_DATE"]
    end = CONFIG["END_DATE"]
    
    log(f"[Data] Fetching {ticker} data from {start} to {end}...")
    
    try:
        # [USER CUSTOMIZATION]: If using a custom API, implement the request here
        # using CONFIG["API_KEY"].
        # Example: df = pd.read_csv(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={CONFIG['API_KEY']}...")
        
        # Default: yfinance
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Flatten MultiIndex if needed (common with new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            raise ValueError("No data returned.")
            
        # Calculate Returns
        df['Return'] = df['Close'].pct_change().abs() * 100
        df_clean = df.dropna()
        
        log(f"[Data] Successfully loaded {len(df_clean)} trading days.")
        return df, df_clean
        
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        # Return empty DataFrame or handle gracefully in production
        return pd.DataFrame(), pd.DataFrame()

# --- MODULE 2: ANALYSIS ENGINE ---

def analyze_omori_decay(df_clean):
    """
    Performs the Omori Law analysis:
    1. Filters aftershocks post-crash.
    2. bins events to calculate rates.
    3. Fits the Power Law model.
    """
    crash_date = pd.to_datetime(CONFIG["CRASH_DATE"])
    thresholds = CONFIG["THRESHOLDS"]
    
    log(f"[Analysis] Analyzing decay from crash date: {crash_date.date()}...")
    
    # 1. Filter Aftershocks
    aftershocks = df_clean[df_clean.index > crash_date].copy()
    if aftershocks.empty:
        log("[Error] No data found after crash date.")
        return None, None, None
        
    # Time t in days since crash (trading days)
    aftershocks['t'] = np.arange(1, len(aftershocks) + 1)
    
    # 2. Variable Binning for Rate Calculation
    # We use logarithmic binning to capture the rapid decay at t=small and slower decay at t=large
    t_max = aftershocks['t'].max()
    bin_edges = np.unique(np.logspace(0, np.log10(t_max), 25).astype(int))
    
    rates_data = {}
    
    for q in thresholds:
        events = aftershocks[aftershocks['Return'] > q]
        bin_centers = []
        event_rates = []
        
        for i in range(len(bin_edges) - 1):
            t0, t1 = bin_edges[i], bin_edges[i + 1]
            dt = t1 - t0
            if dt == 0: continue
            
            # Count events in this time window
            n_events = len(events[(events['t'] >= t0) & (events['t'] < t1)])
            rate = n_events / dt
            
            if rate > 0:
                bin_centers.append(np.sqrt(t0 * t1)) # Geometric mean for log plot
                event_rates.append(rate)
                
        rates_data[q] = (np.array(bin_centers), np.array(event_rates))
        
    # 3. Model Fitting (Power Law: n(t) = A * t^-p)
    def power_law(t, A, p):
        return A * t**(-p)

    # We typically fit on the 1.0% threshold or the most populated one
    fit_threshold = 1.0 if 1.0 in rates_data else thresholds[0]
    tc_fit, rate_fit = rates_data[fit_threshold]
    
    params = (0, 0) # (A, p)
    if len(tc_fit) > 2:
        try:
            popt, _ = curve_fit(power_law, tc_fit, rate_fit, p0=[1, 0.5], maxfev=5000)
            params = popt
            log(f"[Analysis] Power Law Fit (p-value): {params[1]:.4f}")
        except Exception as e:
            log(f"[Warning] Curve fit failed: {e}")
            params = (rate_fit[0], 0.5)
            
    return rates_data, params, aftershocks

# --- MODULE 3: VISUALIZATION ---

def draw_candlestick(ax, dates, opens, highs, lows, closes, alpha=0.35):
    """Draws custom high-performance candlesticks."""
    width = 0.6
    up_color = '#00c853'
    down_color = '#ff1744'

    # Vectorized color assignment would be faster, but loop is clear for logic
    for i in range(len(dates)):
        date = mdates.date2num(dates[i])
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = up_color if c >= o else down_color

        # High-Low Wick
        ax.plot([date, date], [l, h], color=color, linewidth=0.5, alpha=alpha)
        
        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.01: body_height = 0.01 # Min height for visibility
        
        rect = plt.Rectangle((date - width/2, body_bottom), width, body_height,
                             facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.3)
        ax.add_patch(rect)

def generate_static_plot(df, rates_data, fit_params, aftershocks):
    """
    Generates the final Omori Law visualization.
    """
    log("[Visual] Generating static visualization...")
    
    theme = CONFIG["THEME"]
    thresholds = CONFIG["THRESHOLDS"]
    A_fit, p_fit = fit_params
    
    # Setup Figure
    fig = plt.figure(figsize=(14, 12), facecolor=theme["BG"])
    
    # 1. Top Panel: Candlestick Chart
    ax_candle = fig.add_axes([0.08, 0.60, 0.88, 0.30]) # [left, bottom, width, height]
    ax_candle.set_facecolor(theme["PANEL_BG"])
    
    # 2. Bottom Panel: Omori Rate (Log-Log)
    ax_omori = fig.add_axes([0.08, 0.08, 0.88, 0.45])
    ax_omori.set_facecolor(theme["PANEL_BG"])
    
    # --- Plot Candlesticks ---
    # Filter for relevant range (e.g. from 2020 onwards)
    plot_start = pd.to_datetime(CONFIG["START_DATE"])
    candle_df = df[df.index >= plot_start].copy()
    
    dates = candle_df.index.to_pydatetime()
    draw_candlestick(ax_candle, dates, 
                     candle_df['Open'].values, candle_df['High'].values, 
                     candle_df['Low'].values, candle_df['Close'].values, alpha=0.7)
                     
    # Mark Crash Date
    crash_dt = pd.to_datetime(CONFIG["CRASH_DATE"])
    ax_candle.axvline(x=mdates.date2num(crash_dt), color=theme["ACCENT_1"], 
                      linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax_candle.text(mdates.date2num(crash_dt) + 5, candle_df['High'].max()*0.95, 
                   'CRASH EVENT', color=theme["ACCENT_1"], fontsize=10, fontweight='bold')

    # Formatting Top Panel
    ax_candle.set_ylabel(f"{CONFIG['ASSET_TICKER']} Price", color=theme["TEXT"], fontweight='bold')
    ax_candle.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_candle.tick_params(axis='both', colors=theme["TEXT"])
    ax_candle.grid(True, color=theme["GRID"], linestyle='-', linewidth=0.5, alpha=0.5)
    for spine in ax_candle.spines.values(): spine.set_color('#333')
    
    # --- Plot Omori Law ---
    for i, q in enumerate(thresholds):
        if q in rates_data:
            tc, rate = rates_data[q]
            if len(tc) > 0:
                ax_omori.plot(tc, rate, marker=theme["MARKERS"][i], color=theme["COLORS"][i],
                              markersize=7, markerfacecolor='none', markeredgewidth=1.5,
                              linewidth=1.5, linestyle='-', label=rf'$|r| > {q}\%$')
                              
    # Theoretical Fit Lines
    t_max = aftershocks['t'].max() if aftershocks is not None else 100
    t_ref = np.logspace(0, np.log10(t_max), 100)
    
    # Upper/Lower bound visual guides based on fit
    y_fit = A_fit * t_ref**(-p_fit)
    ax_omori.plot(t_ref, y_fit, '--', color=theme["WHITE"], linewidth=1.5, alpha=0.5, label=f'Fit $p={p_fit:.2f}$')
    
    # Axes Formatting
    ax_omori.set_xscale('log')
    ax_omori.set_yscale('log')
    ax_omori.set_xlabel('Trading Days Since Crash ($t$)', color=theme["TEXT"], fontsize=12, fontweight='bold')
    ax_omori.set_ylabel('Event Rate (events/day)', color=theme["TEXT"], fontsize=12, fontweight='bold')
    ax_omori.tick_params(axis='both', which='both', colors=theme["TEXT"])
    
    ax_omori.grid(True, which='major', color=theme["GRID"], linestyle='-', linewidth=0.6)
    ax_omori.grid(True, which='minor', color=theme["GRID"], linestyle=':', linewidth=0.4, alpha=0.3)
    for spine in ax_omori.spines.values(): spine.set_color('#333')
    
    legend = ax_omori.legend(fontsize=10, loc='upper right', facecolor=theme["PANEL_BG"], 
                             edgecolor='#444', labelcolor=theme["TEXT"])
                             
    # Title
    fig.text(0.08, 0.96, f"Omori Law Analysis: {CONFIG['ASSET_TICKER']}", 
             fontsize=18, fontweight='bold', color=theme["WHITE"])
    fig.text(0.08, 0.93, f"Power-law decay of volatility aftershocks (p={p_fit:.2f})", 
             fontsize=11, color=theme["TEXT"])
             
    # Save
    out_path = CONFIG["OUTPUT_FILENAME"]
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor=theme["BG"])
    plt.close(fig)
    log(f"[Visual] Saved static image to: {out_path}")

# --- MAIN ---

def main():
    log("=== OMORI LAW PIPELINE STARTED ===")
    
    # 1. Fetch Data
    df, df_clean = fetch_market_data()
    
    if not df.empty:
        # 2. Analyze
        rates_data, params, aftershocks = analyze_omori_decay(df_clean)
        
        if rates_data:
            # 3. Visualize
            generate_static_plot(df, rates_data, params, aftershocks)
        
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
