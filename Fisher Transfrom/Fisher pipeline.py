"""
Fisher_Pipeline.py
==================
Project: Quant Trader Lab - Fisher Transform 3D Visualization
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for visualizing the Fisher Transform in 3D.
    It transforms raw financial price data into a Gaussian-like distribution,
    visualized as a 3D ribbon with a cinematic camera reveal.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data via `yfinance`.
    2.  **Mathematical Logic**: Computes the Fisher Transform (Normalization + Log Transform).
    3.  **3D Rendering**: Parallelized frame generation using Plotly & Kaleido.
    4.  **Cinematic Camera**: Implements a "Chaos to Clarity" camera transition.
    5.  **Compilation**: Assembles frames into a high-quality MP4 video.

Dependencies:
    pip install numpy pandas yfinance plotly kaleido moviepy

Usage:
    python Fisher_Pipeline.py
"""

import os
import sys
import shutil
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from multiprocessing import Pool, cpu_count

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
    "ASSET": "BTC-USD",
    "PERIOD": "1y",
    "INTERVAL": "1d",
    "FISHER_LEN": 10,
    "FPS": 30,
    "DURATION_SEC": 13,
    "REVEAL_DURATION": 5.0, # Last 5 seconds for the "Reveal"
    "RESOLUTION": (1920, 1080), # 1080p Full HD
    "TEMP_DIR": "temp_fisher_pipeline",
    "OUTPUT_FILE": "Fisher_Pipeline_Output.mp4",
    "LOG_FILE": "fisher_pipeline.log"
}

# Aesthetics ("Bloomberg Dark")
THEME = {
    "BG": "#0b0b0b",
    "GRID": "#1f1f1f",
    "ACCENT": "#00f2ff", # Neon Cyan
    "TEXT": "#ffffff",
    "FONT": "Roboto Mono"
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

def fetch_and_process_data(asset, period, interval):
    """
    Fetches market data and calculates the Fisher Transform.
    Returns a DataFrame with 'Close', 'Fisher', and 'Signal' columns.
    """
    log(f"[Data] Fetching {asset} ({period}, {interval})...")
    
    try:
        df = yf.download(asset, period=period, interval=interval, progress=False)
    except Exception as e:
        log(f"[Error] YF Download failed: {e}")
        return None

    # Handle MultiIndex Columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        log("[Data] Flattened MultiIndex columns.")

    if df.empty:
        log("[Error] No data returned.")
        return None

    # Processing
    try:
        # Extract Price (handle potential 2D array artifacts)
        price = df['Close'].values
        if len(price.shape) > 1:
            price = price.flatten()
            
        # Fisher Transform Algorithm
        n = len(price)
        len_ = CONFIG['FISHER_LEN']
        val1 = np.zeros(n)
        fish = np.zeros(n)
        
        for i in range(len_, n):
            # 1. Normalize price to range [-1, 1] over window
            min_val = np.min(price[i-len_+1:i+1])
            max_val = np.max(price[i-len_+1:i+1])
            
            range_val = max_val - min_val
            if range_val == 0: range_val = 0.001 # Prevent Div/0
                
            # "Value" = scaled price, smoothed
            val_curr = 0.33 * 2 * ((price[i] - min_val) / range_val - 0.5) + 0.67 * val1[i-1]
            val_curr = max(min(val_curr, 0.999), -0.999) # Clip to avoid log(neg)
            val1[i] = val_curr
            
            # 2. Fisher Transform
            fish[i] = 0.5 * np.log((1 + val_curr) / (1 - val_curr)) + 0.5 * fish[i-1]
            
        df['Fisher'] = fish
        df['Signal'] = df['Fisher'].shift(1)
        
        # Trim initial warmup period
        df = df.iloc[CONFIG['FISHER_LEN']:]
        log(f"[Data] Processed {len(df)} points of Fisher data.")
        return df
        
    except Exception as e:
        log(f"[Error] Fisher Calculation failed: {e}")
        return None

# --- MODULE 2: RENDERING ---

def get_camera_position(frame_idx, total_frames, fps):
    """
    Calculates the cinematic camera path.
    Phase 1: Orbit (Chaos) - Shows 3D depth.
    Phase 2: Reveal - Transitions to flat Side View for clarity.
    """
    reveal_frames = int(CONFIG["REVEAL_DURATION"] * fps)
    reveal_start = max(0, total_frames - reveal_frames)
    
    # Base Orbit (Phase 1 Logic)
    # Slow rotation around the data
    orbit_x = 1.8 + 0.5 * np.sin(frame_idx / 50)
    orbit_y = 1.8 + 0.5 * np.cos(frame_idx / 50)
    orbit_z = 0.8
    
    if frame_idx < reveal_start:
        # Phase 1: Pure Orbit
        return dict(x=orbit_x, y=orbit_y, z=orbit_z)
    else:
        # Phase 2: The Reveal
        # Target: Side View (x=0, y=2.5, z=0) looking down Y-axis
        target = np.array([0, 2.5, 0])
        
        # Interpolation progress
        frames_in_phase = frame_idx - reveal_start
        phase_len = total_frames - reveal_start
        if phase_len < 1: phase_len = 1
        
        t = frames_in_phase / phase_len
        
        # Ease-in-out curve
        t = t * t * (3 - 2 * t)
        
        new_x = orbit_x + (target[0] - orbit_x) * t
        new_y = orbit_y + (target[1] - orbit_y) * t
        new_z = orbit_z + (target[2] - orbit_z) * t
        
        return dict(x=new_x, y=new_y, z=new_z)

def render_worker(args):
    """
    Parallel worker to render a single Plotly frame.
    args: (frame_idx, data_subset_idx, df, total_frames, temp_dir)
    """
    frame_idx, idx, df, total_frames, temp_dir = args
    
    try:
        subset = df.iloc[:idx]
        if subset.empty: return False

        # Current Metadata
        curr_date = subset.index[-1].strftime('%Y-%m-%d')
        curr_price = subset['Close'].iloc[-1]
        curr_fish = subset['Fisher'].iloc[-1]
        
        if hasattr(curr_price, 'item'): curr_price = curr_price.item()
        if hasattr(curr_fish, 'item'): curr_fish = curr_fish.item()

        # Build Plot
        fig = go.Figure()
        
        # 1. Fisher Ribbon
        fig.add_trace(go.Scatter3d(
            x=np.arange(len(subset)),
            y=subset['Close'],
            z=subset['Fisher'],
            mode='lines',
            line=dict(color=subset['Fisher'], colorscale='IceFire', width=8),
            name='Fisher'
        ))
        
        # 2. Zero Plane (Reference)
        fig.add_trace(go.Scatter3d(
            x=np.arange(len(subset)),
            y=subset['Close'],
            z=np.zeros(len(subset)),
            mode='lines',
            line=dict(color='#333333', width=2),
            showlegend=False
        ))
        
        # 3. Current Head (Marker)
        fig.add_trace(go.Scatter3d(
            x=[len(subset)-1],
            y=[curr_price],
            z=[curr_fish],
            mode='markers',
            marker=dict(size=10, color='white', symbol='diamond'),
            showlegend=False
        ))
        
        # Styling
        camera_eye = get_camera_position(frame_idx, total_frames, CONFIG["FPS"])
        
        fig.update_layout(
            title=dict(
                text=f"<b>{CONFIG['ASSET']} // FISHER TRANSFORM</b><br><span style='font-size:22px;color:#888'>{curr_date} | PX: {curr_price:.2f} | FISHER: {curr_fish:.2f}</span>",
                font=dict(family=THEME['FONT'], size=34, color="white"),
                y=0.9, x=0.5, xanchor='center'
            ),
            width=CONFIG['RESOLUTION'][0],
            height=CONFIG['RESOLUTION'][1],
            scene=dict(
                xaxis=dict(title='', showgrid=True, gridcolor=THEME['GRID'], backgroundcolor=THEME['BG'], color="white", showticklabels=False),
                yaxis=dict(title='', showgrid=True, gridcolor=THEME['GRID'], backgroundcolor=THEME['BG'], color="white", showticklabels=False),
                zaxis=dict(title='FISHER', showgrid=True, gridcolor=THEME['GRID'], backgroundcolor=THEME['BG'], color="white", range=[-3, 3]),
                bgcolor=THEME['BG'],
                camera=dict(eye=camera_eye, center=dict(x=0,y=0,z=0), up=dict(x=0,y=0,z=1))
            ),
            paper_bgcolor=THEME['BG'],
            margin=dict(l=0, r=0, b=0, t=100),
            showlegend=False
        )
        
        filename = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        fig.write_image(filename)
        return True
        
    except Exception as e:
        print(f"Error rendering frame {frame_idx}: {e}")
        return False

def run_render_manager(df):
    """
    Manages the parallel rendering process.
    Supports 'Smart Resume'.
    """
    # Calculate indices to map data to video frames
    total_points = len(df)
    max_frames = CONFIG["FPS"] * CONFIG["DURATION_SEC"]
    min_window = 20
    
    step = max(1, (total_points - min_window) // max_frames)
    indices = range(min_window, total_points, step)
    total_frames = len(indices)

    # Setup Directory
    if not os.path.exists(CONFIG["TEMP_DIR"]):
        os.makedirs(CONFIG["TEMP_DIR"])
    
    # Smart Resume: Identify Missing Frames
    tasks = []
    for i, idx in enumerate(indices):
        frame_path = os.path.join(CONFIG["TEMP_DIR"], f"frame_{i:04d}.png")
        if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
            tasks.append((i, idx, df, total_frames, CONFIG["TEMP_DIR"]))
    
    if not tasks:
        log("[Render] All frames exist. Skipping.")
        return

    log(f"[Render] Starting processing for {len(tasks)} frames...")
    
    # Execution
    use_parallel = len(tasks) > 5
    
    start_time = time.time()
    
    if use_parallel:
        cores = max(1, cpu_count() - 2)
        log(f"[Render] Using Multiprocessing Pool ({cores} cores).")
        try:
             with Pool(processes=cores) as pool:
                pool.map(render_worker, tasks, chunksize=1)
        except Exception as e:
            log(f"[Error] Pool crashed: {e}. Falling back to serial.")
            for t in tasks: render_worker(t)
    else:
        log("[Render] Using Serial Processing.")
        for t in tasks:
            render_worker(t)
            print(f"Serialized Render: Frame {t[0]} done.")
            
    log(f"[Render] Completed in {time.time() - start_time:.2f}s")


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
        log("[Error] No frames found in temporary directory!")
        return
        
    log(f"[Compile] Found {len(frames)} frames.")
    
    try:
        clip = ImageSequenceClip(frames, fps=CONFIG["FPS"])
        
        # Hold last frame
        # Compatibility Check for MoviePy versions
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
        
    log("=== FISHER PIPELINE START ===")
    
    # 2. Pipeline
    df = fetch_and_process_data(CONFIG["ASSET"], CONFIG["PERIOD"], CONFIG["INTERVAL"])
    
    if df is not None:
        run_render_manager(df)
        compile_video()
            
    log("=== PIPELINE END ===")

if __name__ == "__main__":
    main()
