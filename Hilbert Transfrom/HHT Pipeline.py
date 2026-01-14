"""
HHT_Pipeline.py
===============
Project: Quant Trader Lab - HHT Video Generation
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for visualizing the Hilbert-Huang Transform (HHT) 
    of financial time series. It decomposes the signal into Intrinsic Mode Functions (IMFs)
    and visualizes the Trend, Noise, and Residual components in a Bloomberg-styled 
    animation.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data via `yfinance`.
    2.  **Mathematical Logic**: Decomposes signal using EMD (Empirical Mode Decomposition).
    3.  **Rendering**: Parallelized frame generation using Matplotlib.
    4.  **Compilation**: Assembles frames into a high-quality MP4 video.

Dependencies:
    pip install numpy pandas yfinance scipy matplotlib emd-signal moviepy
    (Optional: pip install emd)

Usage:
    python "HHT Pipeline.py"
"""

import os
import sys
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
from datetime import datetime

# EMD Library Handling
try:
    import emd
    EMD_LIB = 'emd'
except ImportError:
    try:
        from PyEMD import EMD
        EMD_LIB = 'PyEMD'
    except ImportError:
        EMD_LIB = None

# Video Editing (MoviePy v1/v2 Compatibility)
try:
    from moviepy import ImageSequenceClip, ImageClip, CompositeVideoClip
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import ImageSequenceClip, ImageClip, CompositeVideoClip
    MOVIEPY_V2 = False

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    "ASSET": "SPY",
    "START_DATE": "2024-01-01",
    "FPS": 30,
    "DURATION_SEC": 14,
    "RESOLUTION": (1080, 1080), # Square format as per original Visualizer 12x12
    "TEMP_DIR": "temp_frames_hht_pipeline",
    "OUTPUT_FILE": "HHT_Bloomberg_Pipeline.mp4",
    "LOG_FILE": "hht_pipeline.log"
}

# Aesthetics ("Bloomberg Dark")
THEME = {
    "BG": "black",
    "GRID": "#333333",
    "PRICE": "#FFFFFF",
    "TREND": "#FF9800", # Bloomberg Orange
    "NOISE": "#00BFFF", # Deep Sky Blue
    "TEXT": "#B0B0B0",
    "FONT": "sans-serif"
}

# --- UTILS ---

def log(msg):
    """Centralized logger."""
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    try:
        with open(CONFIG["LOG_FILE"], "a") as f:
            f.write(formatted_msg + "\n")
    except:
        pass

# --- MODULE 1: DATA ---

def fetch_and_process_data():
    """
    Fetches market data and performs EMD decomposition.
    Returns: prices (Series), imfs (ndarray), components (dict)
    """
    asset = CONFIG["ASSET"]
    start_date = CONFIG["START_DATE"]
    log(f"[Data] Fetching {asset} from {start_date}...")

    try:
        # Fetch Data
        df = yf.download(asset, start=start_date, progress=False)
        
        if df.empty:
            log("[Error] Empty DataFrame returned.")
            return None, None, None

        # Clean Data (Handle MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try Close then Adj Close
                if 'Close' in df.columns.get_level_values(0):
                    df = df.xs('Close', level=0, axis=1)
                elif 'Close' in df.columns.get_level_values(1):
                     df = df.xs('Close', level=1, axis=1)
            except KeyError:
                pass
        
        # Extract Series
        if isinstance(df, pd.DataFrame):
            if 'Close' in df.columns:
                 price_series = df['Close']
            else:
                 price_series = df.iloc[:, 0]
        else:
            price_series = df

        price_series = price_series.dropna()
        if len(price_series) < 50:
            log(f"[Error] Not enough data points ({len(price_series)}).")
            return None, None, None

        # Perform EMD
        data = price_series.values
        imfs = None
        
        log(f"[Math] Performing EMD using {EMD_LIB}...")
        
        if EMD_LIB == 'emd':
            imf = emd.sift.sift(data)
            imfs = imf.T # emd returns (N, num_imfs) -> Transpose to (num_imfs, N) for consistency
        elif EMD_LIB == 'PyEMD':
            e = EMD()
            imfs = e.emd(data) # PyEMD returns (num_imfs, N)
        else:
            log("[Warning] No EMD library found. Using fallback MA decomposition.")
            # Fallback
            series = pd.Series(data)
            ema10 = series.ewm(span=10).mean().values
            ema50 = series.ewm(span=50).mean().values
            imf0 = data - ema10 # Noise
            imf1 = ema10 - ema50 # Cyclical
            imf2 = ema50 # Trend
            imfs = np.vstack([imf0, imf1, imf2])

        # Decompose into components
        # Logic from original visualizer
        num_imfs = imfs.shape[0]
        
        # Noise: First 2 IMFs or just first
        if num_imfs >= 2:
            noise = np.sum(imfs[:2], axis=0)
        else:
            noise = imfs[0]
            
        # Trend: Remainder
        split_idx = 2
        if num_imfs > split_idx:
            trend = np.sum(imfs[split_idx:], axis=0)
        else:
            trend = imfs[-1]
            
        components = {
            "noise": noise,
            "trend": trend,
            "residual": trend # In original logic, residual and trend were treated similarly for plotting
        }
        
        log(f"[Data] Processed {len(price_series)} points. IMFs: {num_imfs}")
        return price_series, imfs, components

    except Exception as e:
        log(f"[Error] Data/Math failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# --- MODULE 2: RENDERING ---

def render_worker(args):
    """
    Parallel worker to render a single Matplotlib frame.
    args: (frame_idx, cut_idx, prices, components, dates, config, theme, axis_limits)
    """
    frame_idx, i, prices, comps, dates, cfg, theme, limits = args
    
    try:
        # Slice Data
        curr_dates = dates[:i]
        curr_price = prices.iloc[:i]
        curr_trend = comps['trend'][:i]
        curr_noise = comps['noise'][:i]
        
        # Setup Figure
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = theme['FONT']
        
        fig = plt.figure(figsize=(12, 12), facecolor=theme['BG'])
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.15)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Formatting Helper
        def format_ax(ax):
            ax.set_facecolor(theme['BG'])
            ax.grid(True, color=theme['GRID'], linestyle=':', linewidth=0.5)
            ax.set_xlim(dates[0], dates[-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(theme['GRID'])
            ax.spines['left'].set_color(theme['GRID'])
            ax.tick_params(colors=theme['TEXT'])
        
        for ax in [ax1, ax2, ax3]: format_ax(ax)
        
        # --- Plot 1: Price + Trend ---
        ax1.plot(curr_dates, curr_price, color=theme['PRICE'], lw=1.5, label='Price')
        ax1.plot(curr_dates, curr_trend, color=theme['TREND'], lw=2.5, label='Trend')
        ax1.fill_between(curr_dates, curr_price, curr_trend, color=theme['TREND'], alpha=0.1)
        
        ax1.set_ylim(limits['p_min'], limits['p_max'])
        
        # Text/Boxes
        ax1.text(0.02, 0.92, f"{cfg['ASSET']} DECOMPOSED", transform=ax1.transAxes, 
                 color=theme['TREND'], fontsize=16, fontweight='bold')
        ax1.text(0.02, 0.86, "HILBERT-HUANG TRANSFORM", transform=ax1.transAxes, 
                 color='white', fontsize=10)
        
        curr_date_str = curr_dates[-1].strftime('%Y-%m-%d')
        ax1.text(0.35, 0.92, f"DATE: {curr_date_str}", transform=ax1.transAxes, 
                 color=theme['TEXT'], fontsize=12,
                 bbox=dict(facecolor='#1a1a1a', edgecolor=theme['TREND'], boxstyle='round,pad=0.3', alpha=0.8))
        
        ax1.legend(loc='upper right', frameon=True, facecolor='black', edgecolor=theme['GRID'], fontsize=8, labelcolor='white')
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        # --- Plot 2: Noise ---
        ax2.plot(curr_dates, curr_noise, color=theme['NOISE'], lw=1)
        ax2.fill_between(curr_dates, curr_noise, 0, color=theme['NOISE'], alpha=0.2)
        ax2.set_ylim(limits['n_min'], limits['n_max'])
        ax2.text(0.02, 0.88, "MARKET NOISE (High Freq)", transform=ax2.transAxes, 
                 color=theme['NOISE'], fontsize=10, fontweight='bold')
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        # --- Plot 3: Residual/Trend Low Freq ---
        ax3.plot(curr_dates, curr_trend, color=theme['TREND'], lw=2)
        ax3.set_ylim(limits['r_min'], limits['r_max'])
        ax3.text(0.02, 0.88, "UNDERLYING TREND (Signal)", transform=ax3.transAxes, 
                 color=theme['TREND'], fontsize=10, fontweight='bold')
        
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.xticks(rotation=0, ha='center')
        
        # Footer
        fig.text(0.82, 0.02, "SOURCE: YFINANCE", color=theme['TEXT'], fontsize=8, ha='right')
        
        # Save
        filename = os.path.join(cfg["TEMP_DIR"], f"frame_{frame_idx:04d}.png")
        fig.savefig(filename, dpi=100, bbox_inches='tight', facecolor=theme['BG'])
        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error rendering frame {frame_idx}: {e}")
        return False

def run_render_manager(prices, components):
    """
    Manages parallel rendering.
    """
    n_samples = len(prices)
    dates = prices.index
    
    # Pre-compute fixed axis limits
    pad_factor = 0.1
    
    p_min, p_max = prices.min(), prices.max()
    pad_p = (p_max - p_min) * pad_factor
    
    n_min, n_max = components['noise'].min(), components['noise'].max()
    pad_n = (n_max - n_min) * pad_factor
    
    # Residual limits (same as trend for now)
    r_min, r_max = components['trend'].min(), components['trend'].max()
    pad_r = (r_max - r_min) * pad_factor
    
    limits = {
        'p_min': p_min - pad_p, 'p_max': p_max + pad_p,
        'n_min': n_min - pad_n, 'n_max': n_max + pad_n,
        'r_min': r_min - pad_r, 'r_max': r_max + pad_r
    }
    
    # Frame Selection
    # Max frames based on Duration * FPS
    total_frames = CONFIG['DURATION_SEC'] * CONFIG['FPS']
    step = 1
    if n_samples > total_frames:
        step = n_samples // total_frames + 1
    
    # Create Indices: Start from sufficient data (e.g. 50) to end
    indices = range(50, n_samples, step)
    
    log(f"[Render] Generated {len(indices)} tasks for {CONFIG['DURATION_SEC']}s video.")
    
    # Setup Temp Dir
    if os.path.exists(CONFIG["TEMP_DIR"]):
        shutil.rmtree(CONFIG["TEMP_DIR"])
    os.makedirs(CONFIG["TEMP_DIR"], exist_ok=True)
    
    # Task construction
    tasks = []
    for frame_idx, i in enumerate(indices):
        tasks.append((frame_idx, i, prices, components, dates, CONFIG, THEME, limits))
        
    # Execution
    use_parallel = len(tasks) > 10
    start_time = time.time()
    
    if use_parallel:
        cores = max(1, cpu_count() - 2)
        log(f"[Render] Using Pool ({cores} cores).")
        try:
            with Pool(processes=cores) as pool:
                pool.map(render_worker, tasks)
        except Exception as e:
             log(f"[Error] Pool failed: {e}. Serial fallback.")
             for t in tasks: render_worker(t)
    else:
        log("[Render] Serial Processing.")
        for t in tasks:
            render_worker(t)
            
    log(f"[Render] Done in {time.time() - start_time:.2f}s")
    return True

# --- MODULE 3: COMPILATION ---

def compile_video():
    """Compiles frames into MP4 using MoviePy."""
    log("[Compile] Assembling Video...")
    
    frames = sorted([
        os.path.join(CONFIG["TEMP_DIR"], f) 
        for f in os.listdir(CONFIG["TEMP_DIR"]) 
        if f.endswith(".png")
    ])
    
    if not frames:
        log("[Error] No frames found!")
        return
        
    try:
        clip = ImageSequenceClip(frames, fps=CONFIG["FPS"])
        
        # Hold last frame
        try:
            last_frame = ImageClip(frames[-1], duration=2)
        except TypeError:
            last_frame = ImageClip(frames[-1]).set_duration(2)
            
        try:
            held_clip = last_frame.set_start(clip.duration)
        except AttributeError:
             # MoviePy v2 uses with_start
            held_clip = last_frame.with_start(clip.duration)

        final = CompositeVideoClip([clip, held_clip])

        output = os.path.abspath(CONFIG["OUTPUT_FILE"])
        final.write_videofile(output, codec='libx264', bitrate='8000k', audio=False, logger=None)
        
        log(f"[Success] Video saved to: {output}")
        
    except Exception as e:
        log(f"[Error] Compilation failed: {e}")

    # Cleanup
    try:
        shutil.rmtree(CONFIG["TEMP_DIR"])
        log("[Clean] Temp directory removed.")
    except:
        pass

# --- MAIN ---

def main():
    # 1. Init
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])
        
    log("=== HHT PIPELINE START ===")
    
    # 2. Data
    prices, imfs, components = fetch_and_process_data()
    
    if prices is not None:
        # 3. Render
        if run_render_manager(prices, components):
            # 4. Compile
            compile_video()
            
    log("=== PIPELINE END ===")

if __name__ == "__main__":
    main()
