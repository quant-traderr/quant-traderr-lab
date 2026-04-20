"""
RMT_Pipeline.py
===============
Project: Quant Trader Lab - RMT Signal Extraction (Static)
Author: @quant.traderr (Instagram)
License: MIT

Description:
    End-to-end pipeline for rendering a single cinematic static image of
    Random Matrix Theory (RMT) signal extraction on a correlation matrix.
    Neon Quant Grid aesthetic, PyVista-rendered 3D surface at 1920x1080.

    Stages:
    1. **Data Acquisition**: Fetches real market correlations via yfinance,
       or generates a synthetic matrix with embedded signal eigenvalues.
    2. **RMT Processing**: PCA eigen-decomposition, Marchenko-Pastur filter,
       eigenvalue clipping, spectral reconstruction.
    3. **3D Rendering**: Single PyVista frame of the cleaned surface with
       the alpha peak labelled, neon floor grid, eye-dome lighting.

Dependencies:
    pip install numpy pandas pyvista yfinance

System Requirements:
    - GPU recommended for PyVista rendering.

Usage:
    python RMT_Pipeline.py
"""

import os
import numpy as np
import pandas as pd

# Visualization
import pyvista as pv

# Data
import yfinance as yf


# --- CONFIGURATION ---
CONFIG = {
    "RESOLUTION":    (1920, 1080),
    "OUTPUT_FILE":   "RMT_Output.png",
    "USE_REAL_DATA": False,  # Set True to download data via yfinance
    "TICKERS": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
        "INTC", "QCOM", "CSCO", "NFLX", "ADBE", "CRM", "TXN", "AVGO",
        "PYPL", "ORCL", "IBM", "MU",
    ],
    "N_ASSETS":      27,     # For synthetic generation if USE_REAL_DATA off
    "Z_SCALE":       10.0,   # Visual elevation for 3D pop (correlation 0-1 isn't tall)
    "CAMERA_ANGLE":  210.0,  # Degrees — orbit angle around alpha peak
    "CAMERA_ELEV":   30.0,   # Degrees — elevation above floor
}

# Aesthetics ("Neon Quant Grid")
THEME = {
    "BG_TOP":    "#000000",
    "BG_BOTTOM": "#000510",
    "FLOOR":     "#004040",  # Cyan
    "SCANNER":   "#FF0000",
    "TEXT":      "#FFFF00",
    "MESH_CMAP": "plasma",
}


# --- MODULE 1: DATA ---

def fetch_market_data(tickers):
    """Downloads 1 year of daily returns for specified tickers."""
    print(f"[Data] Fetching history for {len(tickers)} assets...")
    try:
        data = yf.download(
            tickers, period="1y", interval="1d", progress=False,
        )['Adj Close']
        returns = data.pct_change().dropna()
        correlation = returns.corr().values
        print(f"[Data] Real Correlation Matrix Shape: {correlation.shape}")
        return correlation
    except Exception as e:
        print(f"[Data] Error fetching YF data: {e}. Falling back to synthetic.")
        return None


def generate_synthetic_data(n_assets):
    """Generates a synthetic correlation matrix with embedded signals."""
    print(f"[Data] Generating synthetic RMT matrix (Size: {n_assets}x{n_assets})...")
    np.random.seed(42)

    # 1. True signals (spikes in eigenvalues)
    signals = np.array([4.5, 3.6, 2.8, 2.2])

    # 2. Noise eigenvalues (bulk)
    noise = np.random.uniform(0.1, 1.2, n_assets - len(signals))
    noise = np.sort(noise)[::-1]

    all_evals = np.concatenate([signals, noise])

    # 3. Construct Matrix from Eigenvalues: M = Q * Lambda * Q^T
    H = np.random.randn(n_assets, n_assets)
    Q, _ = np.linalg.qr(H)  # Random orthogonal matrix
    matrix = Q @ np.diag(all_evals) @ Q.T

    # Enforce correlation-like properties
    np.fill_diagonal(matrix, 1.0)

    return matrix, all_evals


# --- MODULE 1.5: RMT LOGIC ---

def apply_rmt_filtering(correlation_matrix, T=252):
    """
    Random Matrix Theory (RMT) filtering on a correlation matrix via
    eigen-decomposition and Marchenko-Pastur bulk detection.
    """
    print("[RMT] Applying PCA & Marchenko-Pastur Filtering...")

    # 1. PCA / Eigen decomposition
    evals, evecs = np.linalg.eigh(correlation_matrix)

    # 2. Marchenko-Pastur upper edge
    N = correlation_matrix.shape[0]
    Q = T / N
    lambda_plus = (1 + np.sqrt(1 / Q))**2

    print(f"      [RMT Setup] N={N}, T={T}, Q={Q:.2f}")
    print(f"      [Threshold] Max Expected Noise Eigenvalue: {lambda_plus:.4f}")

    # 3. Replace noise eigenvalues with their mean
    noise_indices = evals < lambda_plus
    if np.any(noise_indices):
        avg_noise = np.mean(evals[noise_indices])
        evals[noise_indices] = avg_noise
        print(f"      [Filter] Denoised {np.sum(noise_indices)} eigenvalues.")
    else:
        print("      [Filter] No noise detected (all evals > bound).")

    # 4. Spectral reconstruction
    cleaned_matrix = evecs @ np.diag(evals) @ evecs.T

    # 5. Restore correlation properties
    np.fill_diagonal(cleaned_matrix, 1.0)
    np.clip(cleaned_matrix, -1.0, 1.0, out=cleaned_matrix)

    return cleaned_matrix, lambda_plus


# --- MODULE 2: STATIC RENDER ---

def render_static(cleaned_matrix, lambda_plus):
    """Render a single cinematic PNG of the cleaned RMT surface."""
    print("[Render] Initializing PyVista (off-screen)...")

    n_size = cleaned_matrix.shape[0]

    # Find the alpha peak (off-diagonal max) for camera focus + label
    max_idx = np.unravel_index(
        np.argmax(cleaned_matrix - np.eye(n_size)),
        cleaned_matrix.shape,
    )
    peak_val = cleaned_matrix[max_idx]

    # Scale Z for 3D pop
    z_scale = CONFIG["Z_SCALE"]
    clean_z = cleaned_matrix * z_scale
    peak_pos = np.array([max_idx[0], max_idx[1], peak_val * z_scale])

    # Grid
    x = np.arange(n_size)
    y = np.arange(n_size)
    gx, gy = np.meshgrid(x, y)
    grid = pv.StructuredGrid(gx, gy, clean_z)
    grid["z_height"] = clean_z.ravel(order='F')

    # Floor plane
    floor = pv.Plane(
        center=(n_size / 2, n_size / 2, 0),
        direction=(0, 0, 1),
        i_size=n_size * 1.2,
        j_size=n_size * 1.2,
    )

    # Headless setup (non-Windows)
    if os.name != 'nt':
        try:
            pv.start_xvfb()
        except Exception:
            pass

    plotter = pv.Plotter(off_screen=True, window_size=CONFIG["RESOLUTION"])
    plotter.set_background(THEME["BG_TOP"], top=THEME["BG_BOTTOM"])
    plotter.enable_eye_dome_lighting()
    plotter.add_light(
        pv.Light(position=(60, 60, 80), color='white', intensity=1.1),
    )

    # Floor wireframe (neon cyan grid)
    plotter.add_mesh(
        floor, style='wireframe', color=THEME["FLOOR"], opacity=0.6,
    )

    # Cleaned surface (plasma colormap)
    plotter.add_mesh(
        grid, scalars="z_height", cmap=THEME["MESH_CMAP"],
        opacity=0.9, show_scalar_bar=False,
    )
    # White wireframe overlay for structure
    plotter.add_mesh(
        grid, style='wireframe', color='white', opacity=0.2,
    )

    # Alpha peak label
    plotter.add_point_labels(
        [peak_pos], ["ALPHA DETECTED\n[Noise Filtered]"],
        font_size=24, text_color=THEME["TEXT"],
        font_family='courier', shape_opacity=0, shadow=True,
    )

    # Camera — cinematic oblique orbit around the alpha peak
    diag = np.sqrt(n_size**2 + n_size**2)
    dist = diag * 1.0
    elev = np.radians(CONFIG["CAMERA_ELEV"])
    angle = np.radians(CONFIG["CAMERA_ANGLE"])

    z_cam = dist * np.sin(elev)
    r_xy = dist * np.cos(elev)
    focus = peak_pos
    pos = focus + np.array([np.cos(angle) * r_xy, np.sin(angle) * r_xy, z_cam])

    plotter.camera.position = tuple(pos)
    plotter.camera.focal_point = tuple(focus)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.clipping_range = (0.1, 1000)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CONFIG["OUTPUT_FILE"],
    )
    plotter.screenshot(out_path)
    plotter.close()
    print(f"[Done] Image saved to: {out_path}")


# --- MAIN ---

def main():
    print("=== RMT STATIC PIPELINE START ===")

    # 1. Data
    matrix = None
    if CONFIG["USE_REAL_DATA"]:
        matrix = fetch_market_data(CONFIG["TICKERS"])
    if matrix is None:
        matrix, _ = generate_synthetic_data(CONFIG["N_ASSETS"])

    # 2. RMT processing
    cleaned_matrix, lambda_plus = apply_rmt_filtering(matrix)

    # 3. Single cinematic render
    render_static(cleaned_matrix, lambda_plus)

    print("=== PIPELINE FINISHED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
"""
RMT_Pipeline.py
===============
Project: Quant Trader Lab - RMT Signal Extraction (Static)
Author: @quant.traderr (Instagram)
License: MIT

Description:
    End-to-end pipeline for rendering a single cinematic static image of
    Random Matrix Theory (RMT) signal extraction on a correlation matrix.
    Neon Quant Grid aesthetic, PyVista-rendered 3D surface at 1920x1080.

    Stages:
    1. **Data Acquisition**: Fetches real market correlations via yfinance,
       or generates a synthetic matrix with embedded signal eigenvalues.
    2. **RMT Processing**: PCA eigen-decomposition, Marchenko-Pastur filter,
       eigenvalue clipping, spectral reconstruction.
    3. **3D Rendering**: Single PyVista frame of the cleaned surface with
       the alpha peak labelled, neon floor grid, eye-dome lighting.

Dependencies:
    pip install numpy pandas pyvista yfinance

System Requirements:
    - GPU recommended for PyVista rendering.

Usage:
    python RMT_Pipeline.py
"""

import os
import numpy as np
import pandas as pd

# Visualization
import pyvista as pv

# Data
import yfinance as yf


# --- CONFIGURATION ---
CONFIG = {
    "RESOLUTION":    (1920, 1080),
    "OUTPUT_FILE":   "RMT_Output.png",
    "USE_REAL_DATA": False,  # Set True to download data via yfinance
    "TICKERS": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
        "INTC", "QCOM", "CSCO", "NFLX", "ADBE", "CRM", "TXN", "AVGO",
        "PYPL", "ORCL", "IBM", "MU",
    ],
    "N_ASSETS":      27,     # For synthetic generation if USE_REAL_DATA off
    "Z_SCALE":       10.0,   # Visual elevation for 3D pop (correlation 0-1 isn't tall)
    "CAMERA_ANGLE":  210.0,  # Degrees — orbit angle around alpha peak
    "CAMERA_ELEV":   30.0,   # Degrees — elevation above floor
}

# Aesthetics ("Neon Quant Grid")
THEME = {
    "BG_TOP":    "#000000",
    "BG_BOTTOM": "#000510",
    "FLOOR":     "#004040",  # Cyan
    "SCANNER":   "#FF0000",
    "TEXT":      "#FFFF00",
    "MESH_CMAP": "plasma",
}


# --- MODULE 1: DATA ---

def fetch_market_data(tickers):
    """Downloads 1 year of daily returns for specified tickers."""
    print(f"[Data] Fetching history for {len(tickers)} assets...")
    try:
        data = yf.download(
            tickers, period="1y", interval="1d", progress=False,
        )['Adj Close']
        returns = data.pct_change().dropna()
        correlation = returns.corr().values
        print(f"[Data] Real Correlation Matrix Shape: {correlation.shape}")
        return correlation
    except Exception as e:
        print(f"[Data] Error fetching YF data: {e}. Falling back to synthetic.")
        return None


def generate_synthetic_data(n_assets):
    """Generates a synthetic correlation matrix with embedded signals."""
    print(f"[Data] Generating synthetic RMT matrix (Size: {n_assets}x{n_assets})...")
    np.random.seed(42)

    # 1. True signals (spikes in eigenvalues)
    signals = np.array([4.5, 3.6, 2.8, 2.2])

    # 2. Noise eigenvalues (bulk)
    noise = np.random.uniform(0.1, 1.2, n_assets - len(signals))
    noise = np.sort(noise)[::-1]

    all_evals = np.concatenate([signals, noise])

    # 3. Construct Matrix from Eigenvalues: M = Q * Lambda * Q^T
    H = np.random.randn(n_assets, n_assets)
    Q, _ = np.linalg.qr(H)  # Random orthogonal matrix
    matrix = Q @ np.diag(all_evals) @ Q.T

    # Enforce correlation-like properties
    np.fill_diagonal(matrix, 1.0)

    return matrix, all_evals


# --- MODULE 1.5: RMT LOGIC ---

def apply_rmt_filtering(correlation_matrix, T=252):
    """
    Random Matrix Theory (RMT) filtering on a correlation matrix via
    eigen-decomposition and Marchenko-Pastur bulk detection.
    """
    print("[RMT] Applying PCA & Marchenko-Pastur Filtering...")

    # 1. PCA / Eigen decomposition
    evals, evecs = np.linalg.eigh(correlation_matrix)

    # 2. Marchenko-Pastur upper edge
    N = correlation_matrix.shape[0]
    Q = T / N
    lambda_plus = (1 + np.sqrt(1 / Q))**2

    print(f"      [RMT Setup] N={N}, T={T}, Q={Q:.2f}")
    print(f"      [Threshold] Max Expected Noise Eigenvalue: {lambda_plus:.4f}")

    # 3. Replace noise eigenvalues with their mean
    noise_indices = evals < lambda_plus
    if np.any(noise_indices):
        avg_noise = np.mean(evals[noise_indices])
        evals[noise_indices] = avg_noise
        print(f"      [Filter] Denoised {np.sum(noise_indices)} eigenvalues.")
    else:
        print("      [Filter] No noise detected (all evals > bound).")

    # 4. Spectral reconstruction
    cleaned_matrix = evecs @ np.diag(evals) @ evecs.T

    # 5. Restore correlation properties
    np.fill_diagonal(cleaned_matrix, 1.0)
    np.clip(cleaned_matrix, -1.0, 1.0, out=cleaned_matrix)

    return cleaned_matrix, lambda_plus


# --- MODULE 2: STATIC RENDER ---

def render_static(cleaned_matrix, lambda_plus):
    """Render a single cinematic PNG of the cleaned RMT surface."""
    print("[Render] Initializing PyVista (off-screen)...")

    n_size = cleaned_matrix.shape[0]

    # Find the alpha peak (off-diagonal max) for camera focus + label
    max_idx = np.unravel_index(
        np.argmax(cleaned_matrix - np.eye(n_size)),
        cleaned_matrix.shape,
    )
    peak_val = cleaned_matrix[max_idx]

    # Scale Z for 3D pop
    z_scale = CONFIG["Z_SCALE"]
    clean_z = cleaned_matrix * z_scale
    peak_pos = np.array([max_idx[0], max_idx[1], peak_val * z_scale])

    # Grid
    x = np.arange(n_size)
    y = np.arange(n_size)
    gx, gy = np.meshgrid(x, y)
    grid = pv.StructuredGrid(gx, gy, clean_z)
    grid["z_height"] = clean_z.ravel(order='F')

    # Floor plane
    floor = pv.Plane(
        center=(n_size / 2, n_size / 2, 0),
        direction=(0, 0, 1),
        i_size=n_size * 1.2,
        j_size=n_size * 1.2,
    )

    # Headless setup (non-Windows)
    if os.name != 'nt':
        try:
            pv.start_xvfb()
        except Exception:
            pass

    plotter = pv.Plotter(off_screen=True, window_size=CONFIG["RESOLUTION"])
    plotter.set_background(THEME["BG_TOP"], top=THEME["BG_BOTTOM"])
    plotter.enable_eye_dome_lighting()
    plotter.add_light(
        pv.Light(position=(60, 60, 80), color='white', intensity=1.1),
    )

    # Floor wireframe (neon cyan grid)
    plotter.add_mesh(
        floor, style='wireframe', color=THEME["FLOOR"], opacity=0.6,
    )

    # Cleaned surface (plasma colormap)
    plotter.add_mesh(
        grid, scalars="z_height", cmap=THEME["MESH_CMAP"],
        opacity=0.9, show_scalar_bar=False,
    )
    # White wireframe overlay for structure
    plotter.add_mesh(
        grid, style='wireframe', color='white', opacity=0.2,
    )

    # Alpha peak label
    plotter.add_point_labels(
        [peak_pos], ["ALPHA DETECTED\n[Noise Filtered]"],
        font_size=24, text_color=THEME["TEXT"],
        font_family='courier', shape_opacity=0, shadow=True,
    )

    # Camera — cinematic oblique orbit around the alpha peak
    diag = np.sqrt(n_size**2 + n_size**2)
    dist = diag * 1.0
    elev = np.radians(CONFIG["CAMERA_ELEV"])
    angle = np.radians(CONFIG["CAMERA_ANGLE"])

    z_cam = dist * np.sin(elev)
    r_xy = dist * np.cos(elev)
    focus = peak_pos
    pos = focus + np.array([np.cos(angle) * r_xy, np.sin(angle) * r_xy, z_cam])

    plotter.camera.position = tuple(pos)
    plotter.camera.focal_point = tuple(focus)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.clipping_range = (0.1, 1000)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        CONFIG["OUTPUT_FILE"],
    )
    plotter.screenshot(out_path)
    plotter.close()
    print(f"[Done] Image saved to: {out_path}")


# --- MAIN ---

def main():
    print("=== RMT STATIC PIPELINE START ===")

    # 1. Data
    matrix = None
    if CONFIG["USE_REAL_DATA"]:
        matrix = fetch_market_data(CONFIG["TICKERS"])
    if matrix is None:
        matrix, _ = generate_synthetic_data(CONFIG["N_ASSETS"])

    # 2. RMT processing
    cleaned_matrix, lambda_plus = apply_rmt_filtering(matrix)

    # 3. Single cinematic render
    render_static(cleaned_matrix, lambda_plus)

    print("=== PIPELINE FINISHED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
