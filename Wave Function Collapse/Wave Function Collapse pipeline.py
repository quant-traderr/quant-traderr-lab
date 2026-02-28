"""
Wave Function Collapse Pipeline.py
========================
Project: Quant Trader Lab - Quantum Finance
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline modeling financial market prices as quantum states
    using the Schrödinger Equation.
    
    The pipeline models the probability density function (PDF) of future prices 
    as a coherent superposition state (a wave function). Upon a specified 
    "Measurement Date" (e.g., earnings report, macro event, or sudden crash),
    this probability wave "collapses" into a single realized price execution.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data (e.g., S&P 500, BTC) via yfinance.
    2.  **Quantum State Preparation**: Calculates the historical volatility (σ) and 
        momentum (drift) to construct the initial Gaussian wave packet ψ(x).
    3.  **Time Evolution**: Evolves the wave function using the Free-Particle 
        Schrödinger Equation (split-step Fourier method) up to the collapse date.
    4.  **Measurement/Collapse**: Identifies the realized asset price on the 
        measurement date, collapsing the probability wave |ψ|² into a definite state.
    5.  **Static Visualization**: Generates a high-quality dual-panel static chart
        showing Price Action (Top) and Wave Function Collapse (Bottom).

    Usage:
    - Configure ASSET_TICKER, START_DATE, and MEASUREMENT_DATE in the CONFIG below.
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
import os

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # --------------------------------------------------------------------------
    # [USER CONFIGURATION]
    # Change the Ticker and Measurement Dates to analyze different assets/events.
    # --------------------------------------------------------------------------
    
    # Asset to Analyze
    "ASSET_TICKER": "^GSPC",   # S&P 500: ^GSPC, Bitcoin: BTC-USD, Nvidia: NVDA
    
    # API Key Handling (Optional)
    # If using a custom data provider, set it here.
    "API_KEY": None,           
    
    # Analysis Parameters
    "START_DATE": '2019-06-01',
    "END_DATE": '2020-06-01',
    "MEASUREMENT_DATE": '2020-03-23', # The date of "Collapse" (e.g., sudden crash/news)
    
    # Quantum Model Tuning (Advanced)
    "MOMENTUM_LOOKBACK": 30, # Days to calculate drift/momentum prior to collapse
    "VOLATILITY_LOOKBACK": 60, # Days to calculate uncertainty (width of wave)
    
    # Output
    "OUTPUT_FILENAME": "schrodinger_analysis_static.png",
    
    # Aesthetics (Neon Dark Theme)
    "THEME": {
        "BG":       '#070714',
        "PANEL_BG": '#0a0a1a',
        "GRID":     '#1f1f3a',
        "TEXT":     '#c0c0d0',
        "WAVE_CORE":'#ffffff',
        "WAVE_GLOW":'#00f2ff', # Cyan
        "COLLAPSE": '#ff0055', # Magenta/Crimson for the measurement
        "CANDLE_UP":'#00e676',
        "CANDLE_DN":'#ff1744',
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
    Fetches historical market data via yfinance.
    """
    ticker = CONFIG["ASSET_TICKER"]
    start = CONFIG["START_DATE"]
    end = CONFIG["END_DATE"]
    
    log(f"[Data] Fetching {ticker} data from {start} to {end}...")
    
    try:
        # [USER CUSTOMIZATION]: Swap this out for API fetching if needed
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Flatten MultiIndex if needed (common with new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            raise ValueError("No data returned.")
            
        # Calculate daily returns
        df['Return'] = df['Close'].pct_change()
        df_clean = df.dropna()
        
        log(f"[Data] Successfully loaded {len(df_clean)} trading days.")
        return df, df_clean
        
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- MODULE 2: QUANTUM ANALYSIS ENGINE ---

def analyze_quantum_state(df):
    """
    Models the asset price as a quantum wave function leading up to the Measurement Date.
    """
    m_date_str = CONFIG["MEASUREMENT_DATE"]
    try:
        m_date = pd.to_datetime(m_date_str)
        # Find closest trading day if exact date isn't in index
        if m_date not in df.index:
            m_date = df.index[df.index.get_indexer([m_date], method='nearest')[0]]
    except Exception as e:
        log(f"[Error] Invalid measurement date: {e}")
        return None
        
    log(f"[Analysis] Initializing Quantum State prior to collapse: {m_date.date()}")
    
    # 1. Historical Data leading up to Collapse
    pre_collapse = df[df.index < m_date]
    if len(pre_collapse) < CONFIG["VOLATILITY_LOOKBACK"]:
        log("[Error] Not enough data prior to measurement date to build quantum state.")
        return None
        
    # Get parameters for the wave packet based on historical price action
    recent_prices = pre_collapse['Close'].tail(CONFIG["MOMENTUM_LOOKBACK"])
    vol_prices = pre_collapse['Close'].tail(CONFIG["VOLATILITY_LOOKBACK"])
    
    # Expectation value (Center of the wave) - using the last known price before collapse
    center_price = recent_prices.iloc[-1]
    
    # Uncertainty / Spread (Sigma) - based on true price volatility standard deviation
    price_std = vol_prices.std() * 1.5 # Multiplier for visual spread
    
    # Momentum (k-vector) - based on recent trend (drift)
    trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
    momentum = trend * 0.1 # Scaling factor for quantum phase
    
    # The Realized Measurement
    try:
        realized_price = df.loc[m_date, 'Close']
    except KeyError:
        realized_price = center_price # Fallback
        
    log(f"[Analysis] Quantum Parameters -> Center: {center_price:.2f}, Uncertainty: {price_std:.2f}, Drift: {trend:.2f}")
    log(f"[Analysis] Measurement Collapsed to: {realized_price:.2f}")

    # 2. Build the Grid (Price Space)
    # Ensure our grid covers the wave and the realized price
    x_min = min(center_price - price_std * 5, realized_price - price_std * 2)
    x_max = max(center_price + price_std * 5, realized_price + price_std * 2)
    
    # Force grid to be a power of 2 for FFT speed, though we do a static evolution here
    N = 2048
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    
    # 3. Initial Wave Function (Gaussian Wave Packet)
    # ψ(x) = A * exp(-(x - x0)^2 / (2 * sigma^2)) * exp(i * p * x / hbar)
    psi = np.exp(-0.5 * ((x - center_price) / price_std)**2) * np.exp(1j * momentum * x)
    
    # Normalize the wave function so that ∫|ψ|^2 dx = 1
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    
    # 4. "Time Evolution" 
    # For the static image, we simulate a slight "blur" or expansion mapping the uncertainty
    # just before the measurement happens.
    evolution_time = 0.5 # Arbitrary time steps
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5j * (k**2) * evolution_time)
    psi_evolved = np.fft.ifft(psi_k)
    
    # Final Probability Density
    prob_density = np.abs(psi_evolved)**2
    
    # Create the package to send to visualization
    quantum_data = {
        'x': x,
        'prob_density': prob_density,
        'center': center_price,
        'realized': realized_price,
        'm_date': m_date,
        'sigma': price_std
    }
    
    return quantum_data

# --- MODULE 3: VISUALIZATION ---

def draw_candlestick(ax, dates, opens, highs, lows, closes, theme, alpha=0.8):
    """Draws custom high-performance candlesticks."""
    width = 0.6
    up_color = theme["CANDLE_UP"]
    down_color = theme["CANDLE_DN"]

    for i in range(len(dates)):
        date = mdates.date2num(dates[i])
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = up_color if c >= o else down_color

        # High-Low Wick
        ax.plot([date, date], [l, h], color=color, linewidth=1.0, alpha=alpha)
        
        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.01: body_height = 0.01 
        
        rect = plt.Rectangle((date - width/2, body_bottom), width, body_height,
                             facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.5)
        ax.add_patch(rect)

def generate_static_plot(df, quantum_data):
    """
    Generates the final Schrödinger Pipeline visualization.
    """
    log("[Visual] Generating high-quality static dual visualization...")
    
    theme = CONFIG["THEME"]
    
    # Setup Figure Space
    fig = plt.figure(figsize=(16, 12), facecolor=theme["BG"], dpi=150)
    
    # 1. Top Panel: Candlestick Price History
    ax_price = fig.add_axes([0.08, 0.55, 0.88, 0.35]) # [left, bottom, width, height]
    ax_price.set_facecolor(theme["PANEL_BG"])
    
    # 2. Bottom Panel: The Quantum Probability Space
    ax_quant = fig.add_axes([0.08, 0.08, 0.88, 0.40])
    ax_quant.set_facecolor(theme["PANEL_BG"])
    
    # --- Top Panel: Plot Candlesticks ---
    dates = df.index.to_pydatetime()
    draw_candlestick(ax_price, dates, 
                     df['Open'].values, df['High'].values, 
                     df['Low'].values, df['Close'].values, theme=theme)
                     
    # Mark Collapse/Measurement Date on Price Chart
    m_date = quantum_data['m_date']
    ax_price.axvline(x=mdates.date2num(m_date), color=theme["COLLAPSE"], 
                     linestyle='--', linewidth=2.0, alpha=0.9, zorder=10)
    
    # Simple Price Moving Average
    df['SMA'] = df['Close'].rolling(20).mean()
    ax_price.plot(dates, df['SMA'], color=theme["WAVE_GLOW"], alpha=0.5, lw=1.5, label='Classical Trajectory (20 SMA)')
    
    ax_price.text(mdates.date2num(m_date) + 2, df['High'].max()*0.95, 
                   'WAVE COLLAPSE\n(Measurement)', color=theme["COLLAPSE"], fontsize=11, fontweight='bold')

    # Formatting Top Panel
    ax_price.set_ylabel(f"{CONFIG['ASSET_TICKER']} Price", color=theme["TEXT"], fontweight='bold', fontsize=12)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_price.set_xlim(mdates.date2num(dates[0]), mdates.date2num(dates[-1]))
    ax_price.tick_params(axis='both', colors=theme["TEXT"])
    ax_price.grid(True, color=theme["GRID"], linestyle='-', linewidth=0.5, alpha=0.7)
    for spine in ax_price.spines.values(): spine.set_color('#22223a')
    ax_price.legend(loc='lower left', facecolor=theme["PANEL_BG"], edgecolor='none', labelcolor=theme["TEXT"])
    
    # --- Bottom Panel: Quantum State Collapse ---
    x = quantum_data['x']
    prob = quantum_data['prob_density']
    center = quantum_data['center']
    realized = quantum_data['realized']
    sigma = quantum_data['sigma']
    
    # Normalize prob visually to fit graph nicely
    prob = prob / np.max(prob)
    
    # Draw Probability Wave (Superposition)
    glow_alphas = [0.05, 0.15, 0.4, 1.0]
    glow_lws = [15, 8, 4, 1.5]
    for alpha, lw in zip(glow_alphas, glow_lws):
        color = theme["WAVE_GLOW"] if alpha < 1.0 else theme["WAVE_CORE"]
        ax_quant.plot(x, prob, color=color, alpha=alpha, linewidth=lw, zorder=3)
        
    ax_quant.fill_between(x, prob, color=theme["WAVE_GLOW"], alpha=0.15, zorder=2)
    
    # Draw The Collapse (Measurement)
    collapse_alphas = [0.1, 0.3, 0.6, 1.0]
    for lw, alpha in zip(glow_lws, collapse_alphas):
        color = theme["COLLAPSE"] if alpha < 1.0 else theme["WAVE_CORE"]
        ax_quant.axvline(realized, color=color, alpha=alpha, linewidth=lw, zorder=4)

    # Annotations on Quantum Plot
    ax_quant.axvline(center, color=theme["WAVE_GLOW"], linestyle=':', alpha=0.5, label='Expected Center $\\langle x \\rangle$')
    
    # Spread Bracket
    ax_quant.plot([center - sigma, center + sigma], [0.5, 0.5], color=theme["WAVE_GLOW"], lw=1.5, marker='|')
    ax_quant.text(center, 0.53, f"Uncertainty ($1\\sigma$)", color=theme["WAVE_GLOW"], ha='center', fontsize=9)
    
    ax_quant.text(realized + (x[-1]-x[0])*0.01, 0.9, f"Realized State: ${realized:.2f}", color=theme["COLLAPSE"], fontweight='bold', fontsize=11)

    # Axe Formatting Bottom
    ax_quant.set_xlim(x[0], x[-1])
    ax_quant.set_ylim(-0.05, 1.2)
    ax_quant.set_xlabel('Price Space ($x$)', color=theme["TEXT"], fontsize=12, fontweight='bold')
    ax_quant.set_ylabel('Probability Density $|\\Psi(x)|^2$', color=theme["TEXT"], fontsize=12, fontweight='bold')
    ax_quant.tick_params(axis='both', colors=theme["TEXT"])
    ax_quant.grid(True, color=theme["GRID"], linestyle='-', linewidth=0.5, alpha=0.7)
    for spine in ax_quant.spines.values(): spine.set_color('#22223a')
    
    # Data summary box
    info_text = (
        f"Model: Free-Particle Schrödinger Evolution\n"
        f"Waveform: Gaussian Packet\n"
        f"Measurement Date: {m_date.strftime('%Y-%m-%d')}\n"
        f"Expected $\\langle x \\rangle$: ${center:.2f}\n"
        f"Realized $x_i$: ${realized:.2f}"
    )
    ax_quant.text(0.02, 0.85, info_text, transform=ax_quant.transAxes, color=theme["TEXT"], 
                  fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor=theme["BG"], alpha=0.8, edgecolor=theme["GRID"]))

    # Overall Title
    fig.suptitle(f"SCHRÖDINGER MARKET DECOHERENCE: {CONFIG['ASSET_TICKER']}", color=theme["WAVE_CORE"], fontsize=22, fontweight='heavy', y=0.96)
             
    # Save
    out_path = CONFIG["OUTPUT_FILENAME"]
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=theme["BG"])
    plt.close(fig)
    log(f"[Visual] Saved static image to: {out_path}")

# --- MAIN ---

def main():
    log("=== SCHRÖDINGER PIPELINE STARTED ===")
    
    # 1. Fetch Data
    df, df_clean = fetch_market_data()
    
    if not df_clean.empty:
        # 2. Analyze Quantum State
        quantum_data = analyze_quantum_state(df_clean)
        
        if quantum_data:
            # 3. Visualize Static Final Output
            generate_static_plot(df_clean, quantum_data)
        
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
