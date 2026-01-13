"""
Wavelet_Pipeline.py
===================
Project: Quant Trader Lab - Wavelet Volatility Regimes
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for visualizing market volatility regimes using 
    Continuous Wavelet Transforms (CWT). It highlights time-frequency localization
    of volatility clusters in a Bloomberg-styled visualization.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data via `yfinance`.
    2.  **Mathematical Logic**: Computes Global CWT Power Spectrum using Morlet Wavelets.
    3.  **Rendering**: Parallelized frame generation using Matplotlib.
    4.  **Compilation**: Assembles frames into a high-quality MP4 video.

Dependencies:
    pip install numpy pandas yfinance scipy matplotlib pywavelets moviepy

Usage:
    python Wavelet_Pipeline.py
"""

import os
import sys
import shutil
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Video Editing (MoviePy v1/v2 Compatibility)
try:
    from moviepy import ImageSequenceClip, ImageClip, CompositeVideoClip, vfx
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import ImageSequenceClip, ImageClip, CompositeVideoClip
    import moviepy.video.fx.all as vfx
    MOVIEPY_V2 = False

# --- CONFIGURATION ---

CONFIG = {
    "ASSET": "QQQ",
    "PERIOD": "2y",
    "INTERVAL": "1d",
    "WINDOW_SIZE": 120,    # Trading days visible in one frame
    "FPS": 30,
    "DURATION_SEC": 15,
    "RESOLUTION": (1920, 1080), # 1080p
    "TEMP_DIR": "temp_wavelet_pipeline",
    "OUTPUT_FILE": "Wavelet_Pipeline_Output.mp4",
    "LOG_FILE": "wavelet_pipeline.log"
}

# Aesthetics ("Bloomberg Dark")
THEME = {
    "BG": "#121212",
    "GRID": "#333333",
    "ACCENT": "#FF9900", # Neon Amber
    "TEXT": "#E0E0E0",
    "CMAP": "inferno",
    "FONT": "monospace" # Matplotlib default compatible
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
    Fetches market data and calculates the CWT.
    Returns: prices (Series), z_score (Series), power (ndarray), widths (ndarray)
    """
    asset = CONFIG["ASSET"]
    log(f"[Data] Fetching {asset} ({CONFIG['PERIOD']}, {CONFIG['INTERVAL']})...")
    
    try:
        df = yf.download(asset, period=CONFIG['PERIOD'], interval=CONFIG['INTERVAL'], 
                         progress=False, auto_adjust=False)
        
        # Handle MultiIndex Columns (yfinance v0.2+)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                price_series = df['Adj Close'][asset]
            except KeyError:
                price_series = df['Close'][asset] if asset in df['Close'] else df.iloc[:, 0]
        else:
            price_series = df['Adj Close'] if 'Adj Close' in df else df['Close']

        price_series = price_series.dropna()
        if price_series.empty:
            log("[Error] No data returned.")
            return None, None, None, None

        # Log Returns & Normalization
        log_rets = np.log(price_series / price_series.shift(1)).dropna()
        z_score = (log_rets - log_rets.mean()) / log_rets.std()
        
        # Align prices to z_score indices (drop first)
        price_series = price_series.loc[z_score.index]

        # Compute CWT
        log("[Math] Computing CWT Power Spectrum...")
        widths = np.arange(1, 65)
        # 'cmor1.5-1.0' approximates standard Morlet
        cwtmatr, _ = pywt.cwt(z_score.values, widths, 'cmor1.5-1.0')
        power = np.abs(cwtmatr)**2  # Magnitude Squared
        
        log(f"[Data] Processed {len(z_score)} points.")
        return price_series, z_score, power, widths

    except Exception as e:
        log(f"[Error] Data/Math failed: {e}")
        return None, None, None, None

# --- MODULE 2: RENDERING ---

def render_worker(args):
    """
    Parallel worker to render a single Matplotlib frame.
    args: (frame_idx, start_idx, end_idx, prices, power, config, theme, vmax)
    """
    frame_idx, s, e, prices, power, cfg, theme, vmax = args
    
    try:
        # Data Slices
        segment_price = prices.iloc[s:e]
        segment_cwt = power[:, s:e]
        
        # If segment is smaller than window (start of data), pad? 
        # Alternatively, the manager logic ensures we only render full windows.
        # Assuming manager handles indices correctly.

        # Setup Figure
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 9), facecolor=theme['BG'])
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)
        
        ax_price = fig.add_subplot(gs[0])
        ax_cwt = fig.add_subplot(gs[1], sharex=ax_price)
        
        # Styling
        for ax in [ax_price, ax_cwt]:
            ax.set_facecolor(theme['BG'])
            ax.grid(True, color=theme['GRID'], linestyle='--', linewidth=0.5)
            ax.tick_params(colors=theme['TEXT'], labelcolor=theme['TEXT'])
            for spine in ax.spines.values():
                spine.set_color(theme['GRID'])

        # Plot Price
        x_axis = np.arange(len(segment_price))
        ax_price.plot(x_axis, segment_price.values, color=theme['ACCENT'], linewidth=2)
        
        # Dynamic Limits
        p_min, p_max = segment_price.min(), segment_price.max()
        pad = (p_max - p_min) * 0.05 if (p_max-p_min) > 0 else 1.0
        ax_price.set_ylim(p_min - pad, p_max + pad)
        ax_price.set_xlim(0, cfg['WINDOW_SIZE'])
        
        # Title Data
        d_start = segment_price.index[0].strftime('%Y-%m-%d')
        d_end = segment_price.index[-1].strftime('%Y-%m-%d')
        curr_price = segment_price.iloc[-1]
        
        ax_price.set_title(
            f"REF: {d_end} | PRICE: {curr_price:.2f}",
            loc='right', color=theme['TEXT'], fontsize=12, family='monospace'
        )
        
        # Plot Heatmap
        ax_cwt.imshow(
            segment_cwt, 
            aspect='auto', 
            cmap=theme['CMAP'], 
            origin='lower',
            extent=[0, cfg['WINDOW_SIZE'], 1, 64],
            vmax=vmax, vmin=0
        )
        
        # Labels
        ax_price.set_ylabel(f"{cfg['ASSET']} Price", color=theme['TEXT'])
        ax_cwt.set_ylabel("Freq Scale (Period)", color=theme['TEXT'])
        ax_cwt.set_xlabel("Trading Days (Rolling Window)", color=theme['TEXT'])
        
        # Main Title
        fig.suptitle(
            f"{cfg['ASSET']} VOLATILITY REGIMES | WAVELET TRANSFORM", 
            color=theme['TEXT'], fontsize=20, weight='bold', y=0.95
        )

        filename = os.path.join(cfg["TEMP_DIR"], f"frame_{frame_idx:04d}.png")
        fig.savefig(filename, dpi=100, bbox_inches='tight', facecolor=theme['BG'])
        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error rendering frame {frame_idx}: {e}")
        return False

def run_render_manager(prices, power):
    """
    Manages parallel rendering.
    """
    # Calculate Indices
    N = len(prices)
    W = CONFIG['WINDOW_SIZE']
    total_planned_frames = CONFIG['FPS'] * CONFIG['DURATION_SEC']
    
    # We scan from start to end-window
    max_start = N - W
    if max_start <= 0:
        log("[Error] Data length < Window Size. Cannot animate.")
        return

    step_size = max_start / total_planned_frames
    
    indices = []
    for f in range(total_planned_frames):
        idx = int(f * step_size)
        if idx > max_start: idx = max_start
        indices.append(idx)
        
    # Remove duplicates if any (though for smooth anim we usually keep them or interpolate)
    # We'll just render exactly what's requested for duration
    
    log(f"[Render] Generated {len(indices)} frames for {CONFIG['DURATION_SEC']}s video.")
    
    # Create Temp Dir
    if not os.path.exists(CONFIG["TEMP_DIR"]):
        os.makedirs(CONFIG["TEMP_DIR"])
        
    # Global Max Power for consistent heatmap scaling
    p_99 = np.percentile(power, 99)
    
    # Task construction
    tasks = []
    for i, start_idx in enumerate(indices):
        frame_path = os.path.join(CONFIG["TEMP_DIR"], f"frame_{i:04d}.png")
        if not os.path.exists(frame_path):
            end_idx = start_idx + W
            tasks.append((i, start_idx, end_idx, prices, power, CONFIG, THEME, p_99))
            
    if not tasks:
        log("[Render] All frames exist. Skipping.")
        return

    log(f"[Render] Processing {len(tasks)} frames...")
    
    # Execution
    use_parallel = len(tasks) > 5
    start_time = time.time()
    
    if use_parallel:
        cores = max(1, cpu_count() - 2)
        log(f"[Render] Using Pool ({cores} cores).")
        try:
            # Note: passing large dataframes/arrays via map can be slow due to pickling.
            # However, for 2y daily data it's manageable.
            with Pool(processes=cores) as pool:
                pool.map(render_worker, tasks)
        except Exception as e:
            log(f"[Error] Pool failed: {e}. Switching to serial.")
            for t in tasks: render_worker(t)
    else:
        log("[Render] Serial Processing.")
        for t in tasks:
            render_worker(t)
            
    log(f"[Render] Done in {time.time() - start_time:.2f}s")


# --- MODULE 3: COMPILATION ---

def compile_video():
    """Compiles generated frames into MP4."""
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
        
        # Hold last frame for 2 seconds
        # Compatibility Check
        try:
            last_frame = ImageClip(frames[-1], duration=2)
        except TypeError:
            last_frame = ImageClip(frames[-1]).set_duration(2)
            
        try:
            held_clip = last_frame.set_start(clip.duration)
        except AttributeError:
            held_clip = last_frame.with_start(clip.duration)
            
        final = CompositeVideoClip([clip, held_clip])
        
        output = os.path.abspath(CONFIG["OUTPUT_FILE"])
        final.write_videofile(output, codec='libx264', bitrate='15000k', audio=False, logger=None)
        
        log(f"[Success] Video saved to: {output}")
        
    except Exception as e:
        log(f"[Error] Compilation failed: {e}")

# --- MAIN ---

def main():
    # 1. Clean logs
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])
        
    log("=== WAVELET PIPELINE START ===")
    
    # 2. Pipeline
    prices, z_score, power, widths = fetch_and_process_data()
    
    if prices is not None:
        run_render_manager(prices, power)
        compile_video()
            
    log("=== PIPELINE END ===")

if __name__ == "__main__":
    main()
