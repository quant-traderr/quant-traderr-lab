"""
3D Yield Curve Pipeline.py
==========================
Project: Quant Trader Lab
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for visualizing the 3D Yield Curve Evolution.
    
    It downloads historical treasury yields and macroeconomic data, interpolates
    the yield curve across maturities for each time step, and generates a
    high-quality static 3D surface plot.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical bond data and VIX via yfinance.
        (Includes commented sections for injecting custom DataFrame data).
    2.  **Processing Engine**: Interpolates the yield curve and aligns macro events.
    3.  **Visualization**: Generates a high-quality static snapshot of the 3D yield curve.

Dependencies:
    pip install numpy pandas yfinance matplotlib scipy
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "BOND_TICKERS": ['^IRX', '^FVX', '^TNX', '^TYX'], # 3 Mo, 5 Yr, 10 Yr, 30 Yr
    "MACRO_TICKERS": ['^VIX', '^GSPC'],
    "LOOKBACK_YEARS": 10,
    "INTERVAL": "1mo",
    "ACTUAL_MATURITIES": np.array([3, 60, 120, 360]), # Months
    
    # Interpolation
    "INTERP_KIND": 'quadratic',
    "INTERP_POINTS": 50,
    
    # Macro Shocks (Dates to highlight)
    "MACRO_SHOCKS": {
        "COVID-19 Crash": pd.to_datetime("2020-03-01"),
        "Fed Rate Hikes Begin": pd.to_datetime("2022-03-01")
    },
    
    # Output
    "OUTPUT_IMAGE": "3d_yield_curve_static.png",
    
    # Aesthetics
    "COLOR_BG": '#0e0e0e',
    "COLOR_AXIS_PLANE": (0.0, 0.0, 0.0, 1.0),
    "COLOR_GRID": '#333333',
    "COLOR_TEXT_NORMAL": '#5DA5DA',
    "COLOR_TEXT_STRESS": '#E15F99',
    "COLOR_Z_LABEL": '#FF7F0E'
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches historical bond and macro data.
    Provides a block to inject custom data if needed.
    """
    """
    # ==========================================
    # [CUSTOM DATA INJECTION]
    # To use your own data instead of yfinance, uncomment the block below
    # and load your data into 'close_data'. Note that columns must match the expected format.
    # Expected Index: DatetimeIndex
    # Expected Columns: ^IRX, ^FVX, ^TNX, ^TYX, ^VIX
    # ==========================================
    
    # Example Custom Data Load:
    # log("[Data] Loading custom data from CSV...")
    # close_data = pd.read_csv("my_yield_data.csv", index_col=0, parse_dates=True)
    # return close_data
    """
    
    log(f"[Data] Fetching Bond and Macro Data ({CONFIG['LOOKBACK_YEARS']}y)...")
    
    try:
        tickers = CONFIG['BOND_TICKERS'] + CONFIG['MACRO_TICKERS']
        data = yf.download(tickers, period=f"{CONFIG['LOOKBACK_YEARS']}y", interval=CONFIG['INTERVAL'], progress=False)
        
        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex): 
            close_data = data['Close'].dropna()
        else:
            close_data = data.dropna()
            
        if close_data.empty:
            raise ValueError("No data returned from yfinance.")
            
        log(f"[Data] Loaded {len(close_data)} months of data.")
        return close_data
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        raise

# --- MODULE 2: PROCESSING ENGINE ---

class YieldCurveEngine:
    def __init__(self, data):
        self.data = data
        self.bonds = self.data[CONFIG['BOND_TICKERS']].values
        self.vix = self.data['^VIX'].values if '^VIX' in self.data.columns else np.zeros(len(self.data))
        self.dates = self.data.index
        self.frames = len(self.data)
        self.maturities = np.linspace(1, 360, CONFIG['INTERP_POINTS'])
        self.time_steps = np.arange(self.frames)
        
        self.X, self.Y = np.meshgrid(self.maturities, self.time_steps)
        self.Z = np.zeros_like(self.X)
        self.shock_indices = {}

    def process(self):
        """
        Interpolates the yield curve and identifies macro shock indices.
        """
        log("[Processing] Interpolating yield curve across maturities...")
        for i in range(self.frames):
            interp_func = interp1d(
                CONFIG['ACTUAL_MATURITIES'], 
                self.bonds[i], 
                kind=CONFIG['INTERP_KIND'], 
                fill_value="extrapolate"
            )
            self.Z[i, :] = interp_func(self.maturities)

        log("[Processing] Aligning macro shock dates...")
        for name, shock_date in CONFIG['MACRO_SHOCKS'].items():
            distances = np.abs(self.dates - shock_date)
            closest_idx = np.argmin(distances)
            if distances[closest_idx].days < 60:
                self.shock_indices[name] = closest_idx

        return {
            'X': self.X, 'Y': self.Y, 'Z': self.Z,
            'dates': self.dates, 'vix': self.vix,
            'shock_indices': self.shock_indices
        }

# --- MODULE 3: VISUALIZATION ---

def generate_static_plot(processed_data):
    """
    Generates the final static 3D surface plot.
    """
    log("[Visual] Generating static 3D snapshot...")
    
    X = processed_data['X']
    Y = processed_data['Y']
    Z = processed_data['Z']
    dates = processed_data['dates']
    vix = processed_data['vix']
    shock_indices = processed_data['shock_indices']
    frames = len(dates)

    fig = plt.figure(figsize=(12, 9), facecolor=CONFIG['COLOR_BG'])
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(CONFIG['COLOR_BG'])

    # Dark theme axes
    ax.xaxis.set_pane_color(CONFIG['COLOR_AXIS_PLANE'])
    ax.yaxis.set_pane_color(CONFIG['COLOR_AXIS_PLANE'])
    ax.zaxis.set_pane_color(CONFIG['COLOR_AXIS_PLANE'])
    ax.tick_params(colors='white', labelsize=9)

    # Format time axis
    def format_time(x, pos):
        idx = int(np.clip(x, 0, len(dates)-1))
        return int(dates[idx].year)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_time))

    # Labels
    ax.set_xlabel('\nMaturity (Months)', color=CONFIG['COLOR_TEXT_NORMAL'], labelpad=15)
    ax.set_ylabel('\nTime (Years)', color=CONFIG['COLOR_TEXT_NORMAL'], labelpad=15)
    ax.set_zlabel('\nUSD Treasury Yield (%)', color=CONFIG['COLOR_Z_LABEL'], labelpad=15)

    title_date = dates[-1].strftime("%b %Y")
    current_vix = vix[-1]
    title_color = CONFIG['COLOR_TEXT_STRESS'] if current_vix > 25 else CONFIG['COLOR_TEXT_NORMAL']
    ax.set_title(f"US 3D YIELD CURVE - {title_date}\nStatic Snapshot", color=title_color, pad=20, fontweight='bold', fontsize=14)

    # Plot surface
    color_map = cm.plasma
    ax.plot_surface(X, Y, Z, cmap=color_map, edgecolor='none', alpha=0.85)

    # Plot Macro Shocks
    for name, idx in shock_indices.items():
        z_max = np.max(Z) + 1
        z_min = 0
        ax.plot([360, 360], [idx, idx], [z_min, z_max], 
                color='white', linestyle=':', linewidth=2, alpha=0.8)
        
        # Color red if in the recent past of the last frame, else white
        text_color = CONFIG['COLOR_TEXT_STRESS'] if frames - 1 <= idx + 3 else 'white'
        ax.text(360, idx, z_max + 0.2, name, color=text_color, 
                fontsize=10, fontweight='bold', ha='right')

    # Fix limits
    ax.set_xlim(0, 360)
    ax.set_ylim(0, frames)
    ax.set_zlim(np.min(Z), np.max(Z) + 1)

    # Cinematic camera angle for a great static view
    ax.view_init(elev=25, azim=235)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), CONFIG['OUTPUT_IMAGE'])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=CONFIG['COLOR_BG'])
    plt.close(fig)
    
    log(f"[Visual] Saved to {out_path}")

# --- MAIN ---

def main():
    log("=== 3D YIELD CURVE PIPELINE ===")
    
    # 1. Data
    data = fetch_market_data()
    
    # 2. Processing
    engine = YieldCurveEngine(data)
    processed_data = engine.process()
    
    # 3. Visualization
    generate_static_plot(processed_data)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
