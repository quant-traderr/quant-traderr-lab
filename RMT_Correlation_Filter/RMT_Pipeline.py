
"""
RMT_Pipeline.py
===============
Project: Quant Trader Lab - RMT Signal Extraction
Author: @quant.traderr (Instagram)
License: MIT

Description:
    A complete end-to-end pipeline for visualizing Random Matrix Theory (RMT) signal extraction
    in financial time series. This script handles:
    
    1.  **Data Acquisition**: Fetches real market data via `yfinance` or generates synthetic correlation matrices.
    2.  **RMT Processing**: Computes PCA, Eigenvalues, and applies Marchenko-Pastur filtering.
    3.  **3D Rendering**: Generates a high-fidelity "Cyberpunk/Neon" animation using PyVista (GPU).
    4.  **Post-Production**: Compiles frames into a seamless looping video with speed-ramped intro.

Dependencies:
    pip install numpy pandas scipy pyvista moviepy yfinance
    
System Requirements:
    - GPU recommended for rendering (PyVista).
    - FK/FFMPEG installed (via moviepy).

Usage:
    python RMT_Pipeline.py
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd

# Visualization
import pyvista as pv

# Video Editing
try:
    from moviepy import ImageSequenceClip, ImageClip, CompositeVideoClip, vfx
except ImportError:
    from moviepy.editor import ImageSequenceClip, ImageClip, CompositeVideoClip
    import moviepy.video.fx.all as vfx

# Data
import yfinance as yf

# --- CONFIGURATION ---
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "FPS": 60,
    "DURATION_SEC": 18,
    "TEMP_DIR": "temp_pipeline_frames",
    "OUTPUT_FILE": "RMT_Signal_Extraction_v1.mp4",
    "USE_REAL_DATA": False, # Set to True to download data
    "TICKERS": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "INTC", "QCOM", 
                "CSCO", "NFLX", "ADBE", "CRM", "TXN", "AVGO", "PYPL", "ORCL", "IBM", "MU"], # Top 20 Tech
    "N_ASSETS": 27, # For synthetic generation if Real Data off
}

# Aesthetics ("Neon Quant Grid")
THEME = {
    "BG_TOP": "#000000",
    "BG_BOTTOM": "#000510",
    "FLOOR": "#004040", # Cyan
    "SCANNER": "#FF0000",
    "TEXT": "#FFFF00",
    "MESH_CMAP": "plasma"
}

# --- MODULE 1: DATA ---

def fetch_market_data(tickers):
    """Downloads 1 year of daily returns for specified tickers."""
    print(f"[Data] Fetching history for {len(tickers)} assets...")
    try:
        data = yf.download(tickers, period="1y", interval="1d", progress=False)['Adj Close']
        returns = data.pct_change().dropna()
        correlation = returns.corr().values
        print(f"[Data] Real Correlation Matrix Shape: {correlation.shape}")
        
        # We need a square matrix of size N*N. If data < synthetic size, pad or slice.
        # For this demo, we'll just return what we got.
        return correlation
    except Exception as e:
        print(f"[Data] Error fetching YF data: {e}. Falling back to synthetic.")
        return None

def generate_synthetic_data(n_assets):
    """Generates a synthetic correlation matrix with embedded signals."""
    print(f"[Data] Generating synthetic RMT matrix (Size: {n_assets}x{n_assets})...")
    np.random.seed(42)
    
    # 1. Create true signals (Spikes in eigenvalues)
    signals = np.array([4.5, 3.6, 2.8, 2.2]) 
    
    # 2. Create Noise (Bulk of the distribution)
    # Marchenko-Pastur distribution simulation
    noise = np.random.uniform(0.1, 1.2, n_assets - len(signals))
    noise = np.sort(noise)[::-1]
    
    all_evals = np.concatenate([signals, noise])
    
    # 3. Construct Matrix from Eigenvalues
    # M = Q * Lambda * Q^T
    H = np.random.randn(n_assets, n_assets)
    Q, _ = np.linalg.qr(H) # Random Orthogonal Matrix
    
    matrix = Q @ np.diag(all_evals) @ Q.T
    
    # Ensure symmetry / correlation properties (approx)
    np.fill_diagonal(matrix, 1.0)
    
    return matrix, all_evals

# --- MODULE 1.5: RMT LOGIC ---

def apply_rmt_filtering(correlation_matrix, T=252):
    """
    Applies Random Matrix Theory (RMT) filtering to the correlation matrix.
    Uses PCA (Eigen-decomposition) to separate Signal from Noise.
    """
    print("[RMT] Applying PCA & Marchenko-Pastur Filtering...")
    
    # 1. PCA / Eigen decomposition
    evals, evecs = np.linalg.eigh(correlation_matrix)
    
    # 2. Determine Noise Threshold (Marchenko-Pastur)
    # Q = T / N (Quality factor)
    N = correlation_matrix.shape[0]
    Q = T / N
    lambda_plus = (1 + np.sqrt(1/Q))**2
    
    print(f"      [RMT Setup] N={N}, T={T}, Q={Q:.2f}")
    print(f"      [Threshold] Max Expected Noise Eigenvalue: {lambda_plus:.4f}")
    
    # 3. Filter Eigenvalues
    # Replace noise eigenvalues (lambda < lambda_plus) with their average
    noise_indices = evals < lambda_plus
    if np.any(noise_indices):
        avg_noise = np.mean(evals[noise_indices])
        evals[noise_indices] = avg_noise
        print(f"      [Filter] Denoised {np.sum(noise_indices)} eigenvalues.")
    else:
        print("      [Filter] No noise detected (all evals > bound).")
        
    # 4. Reconstruct Matrix
    # Cleaned = V * Lambda_clean * V^T
    cleaned_matrix = evecs @ np.diag(evals) @ evecs.T
    
    # 5. Restore Correlation Properties (Diag=1)
    np.fill_diagonal(cleaned_matrix, 1.0)
    
    # Optional: Fix potential numerical drift causing values > 1 or < -1
    np.clip(cleaned_matrix, -1.0, 1.0, out=cleaned_matrix)
    
    return cleaned_matrix, lambda_plus

# --- MODULE 2: RENDER CAMERA LOGIC ---

def get_camera_path(t, n_assets, peak_pos):
    """
    Calculates camera position for the "Spiral Dive" narrative.
    
    Timeline:
    0-5s: Establishing Orbit (Wide)
    5-9s: Focus/Zoom (Corkscrew in)
    9-14s: Scan (Stabilized drift)
    14-18s: Result (Orbit Alpha)
    """
    center = np.array([n_assets/2, n_assets/2, 0])
    diag = np.sqrt(n_assets**2 + n_assets**2)
    
    # Default return
    pos = center + np.array([0, 0, 100])
    focus = center
    
    if t < 5:
        # ESTABLISHING
        p = t / 5.0
        dist = diag * 2.5
        elev = np.radians(45)
        angle = np.radians(45 + 15 * t) # Slow rotation
        
        z = dist * np.sin(elev)
        r_xy = dist * np.cos(elev)
        pos = center + np.array([np.cos(angle)*r_xy, np.sin(angle)*r_xy, z])
        focus = center

    elif 5 <= t < 9:
        # ZOOM
        p = (t - 5) / 4.0
        ease = p * p * (3 - 2 * p) # Smoothstep
        
        # Start state (matches end of phase 1)
        start_dist = diag * 2.5
        start_angle = np.radians(45 + 75) # 120 deg
        
        # End state
        end_dist = diag * 1.0
        end_angle = start_angle + np.radians(30)
        
        cur_dist = start_dist + (end_dist - start_dist) * ease
        cur_angle = start_angle + (end_angle - start_angle) * ease
        cur_elev = np.radians(45 + (30-45)*ease) # Drop to 30 deg
        
        z = cur_dist * np.sin(cur_elev)
        r_xy = cur_dist * np.cos(cur_elev)
        
        # Interpolate focal point from Center to Peak
        focus = (1 - ease) * center + ease * peak_pos
        
        pos = focus + np.array([np.cos(cur_angle)*r_xy, np.sin(cur_angle)*r_xy, z])
        # Correction: Keep elevation relative to floor for consistent look? 
        # Actually relative to focus Z is fine for closeups.
        
    elif 9 <= t < 14:
        # SCAN STABILIZED
        p = (t - 9) / 5.0
        focus = peak_pos
        dist = diag * 1.0
        elev = np.radians(30)
        angle = np.radians(150 + 10 * p) # Slow drift
        
        z = dist * np.sin(elev)
        r_xy = dist * np.cos(elev)
        pos = focus + np.array([np.cos(angle)*r_xy, np.sin(angle)*r_xy, z])

    else:
        # ORBIT ALPHA
        p = (t - 14) / 4.0
        focus = peak_pos
        dist = diag * 1.0
        elev = np.radians(30)
        angle = np.radians(160 + 360 * p) # Full spin
        
        z = dist * np.sin(elev)
        r_xy = dist * np.cos(elev)
        pos = focus + np.array([np.cos(angle)*r_xy, np.sin(angle)*r_xy, z])
        
    return pos, focus

# --- MODULE 3: RENDER ENGINE ---

def render_animation(raw_matrix, cleaned_matrix, lambda_plus):
    """Main rendering loop using PyVista."""
    print("[Render] Initializing GPU Engine...")
    
    # 1. Setup
    n_size = raw_matrix.shape[0]
    
    # Find Peak for camera target (use cleaned for final focus)
    max_idx = np.unravel_index(np.argmax(cleaned_matrix - np.eye(n_size)), cleaned_matrix.shape) # Exclude diagonal
    peak_pos = np.array([max_idx[0], max_idx[1], cleaned_matrix[max_idx]])
    
    # Grid
    x = np.arange(n_size)
    y = np.arange(n_size)
    gx, gy = np.meshgrid(x, y)
    
    # PyVista
    # Linux/Colab support (headless). Safe to skip on Windows.
    if os.name != 'nt':
        try:
            pv.start_xvfb()
        except:
            pass

    plotter = pv.Plotter(off_screen=True, window_size=CONFIG["RESOLUTION"])
    plotter.set_background(THEME["BG_TOP"], top=THEME["BG_BOTTOM"])
    plotter.enable_eye_dome_lighting() # Crucial for "Cyberpunk" look
    plotter.add_light(pv.Light(position=(60,60,80), color='white', intensity=1.1))

    # Calculate ranges for Z-scaling (Visual only)
    # Scale matrices for better 3D pop (Correlation 0-1 isn't very tall)
    Z_SCALE = 10.0
    raw_z = raw_matrix * Z_SCALE
    clean_z = cleaned_matrix * Z_SCALE
    
    # Meshes
    # We will initialize with Raw Data
    grid = pv.StructuredGrid(gx, gy, raw_z)
    grid["z_height"] = raw_z.ravel(order='F') 
    
    floor = pv.Plane(center=(n_size/2, n_size/2, 0), direction=(0,0,1), 
                     i_size=n_size*1.2, j_size=n_size*1.2)
    
    # Prepare Output
    if os.path.exists(CONFIG["TEMP_DIR"]):
        shutil.rmtree(CONFIG["TEMP_DIR"])
    os.makedirs(CONFIG["TEMP_DIR"])
    
    total_frames = int(CONFIG["FPS"] * CONFIG["DURATION_SEC"])
    print(f"[Render] Starting Render ({total_frames} frames)...")
    
    for i in range(total_frames):
        t = i / CONFIG["FPS"]
        plotter.clear()
        
        # 1. Draw Static Elements
        plotter.add_mesh(floor, style='wireframe', color=THEME["FLOOR"], opacity=0.6)
        
        # 2. Logic: Signal Scanning & RMT Interpolation
        # From t=9s to t=14s, interpolate from Raw -> Cleaned
        
        current_data = raw_z.copy()
        scan_progress = 0.0
        
        if 9 <= t < 14:
            scan_progress = (t - 9) / 5.0
            # Interpolate values
            current_data = (1 - scan_progress) * raw_z + scan_progress * clean_z
        elif t >= 14:
            current_data = clean_z
            scan_progress = 1.0
            
        # Update Grid Geometry
        grid.points[:, 2] = current_data.ravel(order='F')
        grid["z_height"] = current_data.ravel(order='F')
        
        # Draw Mesh
        # Threshold for visual clarity?
        # Just draw full mesh with opacity mapping
        plotter.add_mesh(grid, scalars="z_height", cmap=THEME["MESH_CMAP"], 
                         opacity=0.9, show_scalar_bar=False)
        
        # Wireframe overlay
        plotter.add_mesh(grid, style='wireframe', color='white', opacity=0.2)

        # Draw Laser Plane (The "Cleaning" Beam)
        if 9 <= t < 14:
            scan_x = n_size * scan_progress
            plane = pv.Plane(center=(scan_x, n_size/2, 5), direction=(1,0,0), 
                             i_size=15, j_size=n_size)
            plotter.add_mesh(plane, color=THEME["SCANNER"], opacity=0.3)
            
        # 3. UI
        if t >= 13.0:
             plotter.add_point_labels([peak_pos], ["ALPHA DETECTED\n[Noise Filtered]"],
                                      font_size=24, text_color=THEME["TEXT"], font_family='courier',
                                      shape_opacity=0, shadow=True)

        # 4. Camera
        pos, focus = get_camera_path(t, n_size, peak_pos)
        plotter.camera.position = pos
        plotter.camera.focal_point = focus
        plotter.camera.up = (0, 0, 1)
        plotter.camera.clipping_range = (0.1, 1000)
        
        # Save
        plotter.screenshot(os.path.join(CONFIG["TEMP_DIR"], f"frame_{i:04d}.png"))
        
        if i % 100 == 0:
            print(f"      Rendered {i}/{total_frames} | Mode: {'RAW' if t<9 else 'RMT OK' if t>=14 else 'CLEANING'}")

    plotter.close()
    print("[Render] Rendering Complete.")

# --- MODULE 4: COMPILATION ---

def compile_final_video():
    """Compiles frames with Speed Ramp (Intro) and Seamless Loop (Fade)."""
    print("[Compile] Assembling Video with Post-Processing...")
    
    file_list = sorted([os.path.join(CONFIG["TEMP_DIR"], f) for f in os.listdir(CONFIG["TEMP_DIR"]) if f.endswith(".png")])
    if not file_list:
        print("[Compile] No frames found!")
        return

    # Logic: 
    # Intro (0-9s) -> Speed up 300% (Take every 3rd frame)
    # Outro (9-18s) -> Normal speed
    
    split_idx = int(9 * CONFIG["FPS"]) # 540
    
    intro_frames = file_list[:split_idx:3]
    outro_frames = file_list[split_idx:]
    final_frames = intro_frames + outro_frames
    
    # 1. Main Clip
    clip = ImageSequenceClip(final_frames, fps=CONFIG["FPS"])
    
    # 2. Seamless Loop Trick (Crossfade to Frame 0 at end)
    # Extract frame 0
    fade_duration = 0.5
    first_frame = ImageClip(final_frames[0])
    
    # MoviePy v2 Syntax handling
    if hasattr(first_frame, "with_duration"):
        overlay = first_frame.with_duration(fade_duration).with_start(clip.duration - fade_duration)
        overlay = overlay.with_effects([vfx.CrossFadeIn(fade_duration)])
    else:
        # v1 Syntax
        overlay = first_frame.set_duration(fade_duration).set_start(clip.duration - fade_duration).crossfadein(fade_duration)
    
    video = CompositeVideoClip([clip, overlay])
    
    video.write_videofile(CONFIG["OUTPUT_FILE"], codec='libx264', bitrate='15000k', audio=False)
    print(f"[Done] Video saved to: {os.path.abspath(CONFIG['OUTPUT_FILE'])}")

# --- MAIN ---

def main():
    print("=== RMT SIGNAL PIPELINE START ===")
    
    # 1. Get Data
    matrix = None
    if CONFIG["USE_REAL_DATA"]:
        matrix = fetch_market_data(CONFIG["TICKERS"])
        
    if matrix is None:
        matrix, _ = generate_synthetic_data(CONFIG["N_ASSETS"])
    
    # 2. RMT Processing
    cleaned_matrix, lambda_plus = apply_rmt_filtering(matrix)
        
    # 3. Render
    render_animation(matrix, cleaned_matrix, lambda_plus)
    
    # 4. Compile
    compile_final_video()
    
    print("=== PIPELINE FINISHED SUCCESSFULY ===")

if __name__ == "__main__":
    main()
