"""
Ornstein-Uhlenbeck Pipeline.py
==============================
Project: Quant Trader Lab - Mean Reversion Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for Ornstein-Uhlenbeck (O-U) Process Simulation.
    
    The O-U process is a mean-reverting stochastic model widely used in 
    quantitative finance for modeling interest rates, volatility, and 
    pairs trading strategies.

    The Stochastic Differential Equation (SDE):
        dX_t = θ(μ - X_t)dt + σdW_t

    Where:
        θ (theta) = Speed of mean reversion
        μ (mu)    = Long-term mean (equilibrium level)
        σ (sigma) = Volatility of the process
        W_t       = Standard Brownian motion (Wiener process)

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches historical data via yfinance for calibration.
    2.  **Parameter Estimation**: Calibrates O-U parameters from real market data.
    3.  **Simulation Engine**: Runs vectorized O-U path simulations.
    4.  **Analytical Bounds**: Computes theoretical expectation and variance.
    5.  **Static Visualization**: Generates a high-quality snapshot with:
        - Multiple sample paths (probability cloud)
        - Theoretical mean reversion trajectory
        - ±2σ confidence bands (funnel effect)
        - A single "hero" realized path

    NOTE: Video rendering has been removed for pipeline efficiency.
          See render_timelapse.py for animated output.

Dependencies:
    pip install numpy pandas yfinance matplotlib
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data Acquisition
    "TICKER": "GLD",            # Gold ETF - classic mean-reverting asset
    "LOOKBACK_YEARS": 1,        # Historical data for calibration
    "INTERVAL": "1d",           # Daily data
    
    # O-U Process Parameters (will be overwritten by calibration if USE_CALIBRATION=True)
    "USE_CALIBRATION": True,    # If True, estimate params from data; else use manual
    "THETA": 1.5,               # Mean reversion speed (manual override)
    "MU": 100.0,                # Long-term mean (manual override)
    "SIGMA": 3.0,               # Volatility (manual override)
    "X0": None,                 # Initial value (None = use latest price)
    
    # Simulation
    "DT": 1/252,                # Time step (1 day in years)
    "N_STEPS": 252,             # Number of time steps (1 year of trading days)
    "N_PATHS": 50,              # Number of sample paths to simulate
    "RANDOM_SEED": 42,          # Set integer for reproducibility, None for random
    
    # Output
    "OUTPUT_IMAGE": "ou_static.png",
    
    # Aesthetics
    "COLOR_BG": '#0e1117',
    "COLOR_PATHS": '#00ff41',       # Matrix green for ghost paths
    "COLOR_MEAN": 'cyan',           # Theoretical mean line
    "COLOR_BOUNDS": 'red',          # ±2σ confidence bands
    "COLOR_HERO": '#ffffff',        # Realized "hero" path
    "PATH_ALPHA": 0.05,             # Ghost path transparency
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ACQUISITION ---

def fetch_market_data():
    """
    Fetches historical market data for O-U parameter calibration.
    
    Returns:
        prices (pd.Series): Historical closing prices
    """
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['LOOKBACK_YEARS']}y, {CONFIG['INTERVAL']})...")
    
    try:
        data = yf.download(
            CONFIG['TICKER'], 
            period=f"{CONFIG['LOOKBACK_YEARS']}y", 
            interval=CONFIG['INTERVAL'], 
            progress=False
        )
        
        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG['TICKER'], axis=1, level=1)
        
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        
        prices = data['Close'].dropna()
        log(f"[Data] Loaded {len(prices)} data points. Latest: ${prices.iloc[-1]:.2f}")
        
        return prices
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        log("[Warning] Using synthetic data for demonstration.")
        # Fallback: Generate synthetic mean-reverting data
        np.random.seed(42)
        n = 252
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] + 0.5 * (100 - prices[-1]) * (1/252) + 2 * np.random.normal() * np.sqrt(1/252))
        return pd.Series(prices)

# --- MODULE 2: PARAMETER ESTIMATION ---

def estimate_ou_parameters(prices):
    """
    Estimates O-U parameters from historical price data using OLS regression.
    
    Method: Discretized O-U process → AR(1) regression
        X_{t+1} - X_t = θ(μ - X_t)Δt + σ√Δt * ε
        
    Rearranged as:
        ΔX = a + b*X_t + noise
        
    Where:
        b = -θΔt     → θ = -b/Δt
        a = θμΔt     → μ = a/(-b) = a/(θΔt)
        σ = std(residuals) / √Δt
    
    Returns:
        theta, mu, sigma, x0
    """
    log("[Calibration] Estimating O-U parameters from data...")
    
    dt = CONFIG['DT']
    
    # Calculate price changes
    X = prices.values
    X_t = X[:-1]
    X_tp1 = X[1:]
    dX = X_tp1 - X_t
    
    # OLS Regression: dX = a + b * X_t + residual
    # Using least squares: [a, b] = (X'X)^(-1) X'y
    ones = np.ones_like(X_t)
    A = np.column_stack([ones, X_t])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, dX, rcond=None)
    a, b = coeffs
    
    # Extract O-U parameters
    theta = -b / dt
    mu = a / (-b) if b != 0 else X.mean()
    
    # Estimate sigma from residual standard deviation
    predicted = a + b * X_t
    residual_std = np.std(dX - predicted)
    sigma = residual_std / np.sqrt(dt)
    
    # Initial value = latest price
    x0 = X[-1]
    
    # Validate parameters
    if theta <= 0:
        log("[Warning] Negative theta detected (no mean reversion). Using absolute value.")
        theta = abs(theta) if abs(theta) > 0.01 else 0.5
    
    log(f"[Calibration] θ = {theta:.4f} (half-life: {np.log(2)/theta:.1f} days)")
    log(f"[Calibration] μ = {mu:.2f} (long-term mean)")
    log(f"[Calibration] σ = {sigma:.4f} (volatility)")
    log(f"[Calibration] X₀ = {x0:.2f} (current price)")
    
    return theta, mu, sigma, x0

# --- MODULE 3: SIMULATION ENGINE ---

class OUEngine:
    """
    Ornstein-Uhlenbeck Process Simulation Engine.
    
    Uses discretized Euler-Maruyama method for SDE simulation.
    """
    
    def __init__(self, theta, mu, sigma, x0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.dt = CONFIG['DT']
        self.n_steps = CONFIG['N_STEPS']
        self.n_paths = CONFIG['N_PATHS']
        
    def run(self):
        """
        Executes the O-U process simulation.
        
        Returns:
            t (ndarray): Time array of shape (n_steps,)
            paths (ndarray): Simulated paths of shape (n_steps, n_paths)
        """
        log(f"[Simulation] Generating {self.n_paths} O-U paths ({self.n_steps} steps)...")
        
        start_time = time.time()
        
        if CONFIG['RANDOM_SEED'] is not None:
            np.random.seed(CONFIG['RANDOM_SEED'])
        
        # Time array (in years)
        t = np.linspace(0, self.n_steps * self.dt, self.n_steps)
        
        # Initialize paths
        paths = np.zeros((self.n_steps, self.n_paths))
        paths[0, :] = self.x0
        
        # Vectorized noise generation
        noise = np.random.normal(0, np.sqrt(self.dt), (self.n_steps, self.n_paths))
        
        # Euler-Maruyama discretization
        for i in range(1, self.n_steps):
            dx = self.theta * (self.mu - paths[i-1, :]) * self.dt + self.sigma * noise[i, :]
            paths[i, :] = paths[i-1, :] + dx
        
        duration = time.time() - start_time
        log(f"[Simulation] Completed in {duration:.3f}s.")
        
        return t, paths

# --- MODULE 4: ANALYTICAL BOUNDS ---

def compute_theoretical_bounds(t, theta, mu, sigma, x0):
    """
    Computes the theoretical expectation and variance bounds for the O-U process.
    """
    log("[Analysis] Computing theoretical bounds...")
    
    # Expectation: E[X_t] = μ + (X_0 - μ) * exp(-θt)
    exp_val = mu + (x0 - mu) * np.exp(-theta * t)
    
    # Variance: Var[X_t] = (σ² / 2θ) * (1 - exp(-2θt))
    variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))
    std_dev = np.sqrt(variance)
    
    asymptotic_std = sigma / np.sqrt(2 * theta)
    log(f"[Analysis] Asymptotic std: ${asymptotic_std:.2f}")
    log(f"[Analysis] Half-life: {np.log(2)/theta:.1f} days")
    
    return exp_val, exp_val + 2*std_dev, exp_val - 2*std_dev

# --- MODULE 5: VISUALIZATION ---

def visualize(t, paths, exp_val, upper_bound, lower_bound, ticker):
    """
    Generates a static visualization of the O-U process.
    """
    log("[Visual] Generating static snapshot...")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=CONFIG['COLOR_BG'])
    ax.set_facecolor(CONFIG['COLOR_BG'])
    
    # Ghost Paths
    ax.plot(t * 252, paths[:, 1:], color=CONFIG['COLOR_PATHS'], 
            alpha=CONFIG['PATH_ALPHA'], linewidth=1)
    
    # Bounds
    ax.plot(t * 252, upper_bound, color=CONFIG['COLOR_BOUNDS'], 
            linestyle='--', linewidth=1.5, alpha=0.7, label='±2σ Bounds')
    ax.plot(t * 252, lower_bound, color=CONFIG['COLOR_BOUNDS'], 
            linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Mean
    ax.plot(t * 252, exp_val, color=CONFIG['COLOR_MEAN'], 
            linestyle=':', linewidth=1.5, alpha=0.8, label='Theoretical Mean')
    
    # Hero Path
    ax.plot(t * 252, paths[:, 0], color=CONFIG['COLOR_HERO'], 
            linewidth=2.25, alpha=0.9, label='Realized Path')
    
    # Styling
    ax.set_title(f"Ornstein-Uhlenbeck: {ticker} Projection", 
                 fontsize=18, color='white', weight='bold', pad=20)
    ax.set_xlabel("Days", color='gray')
    ax.set_ylabel("Price ($)", color='gray')
    ax.legend(loc='upper right', frameon=True, facecolor='#161b22', edgecolor='white')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.1)
    
    # Formula Overlay
    formula = r"$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$"
    plt.figtext(0.5, 0.05, formula, ha="center", fontsize=22, color='white',
                bbox=dict(boxstyle="round,pad=0.4", fc="#161b22", ec="white", alpha=0.9))
    
    plt.subplots_adjust(bottom=0.18)
    
    out_path = os.path.join(os.path.dirname(__file__), CONFIG['OUTPUT_IMAGE'])
    plt.savefig(out_path, facecolor=CONFIG['COLOR_BG'], dpi=150)
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")

# --- MAIN ---

def main():
    log("=== ORNSTEIN-UHLENBECK PIPELINE ===")
    
    # 1. Data Acquisition
    prices = fetch_market_data()
    
    # 2. Parameter Estimation
    if CONFIG['USE_CALIBRATION']:
        theta, mu, sigma, x0 = estimate_ou_parameters(prices)
    else:
        theta = CONFIG['THETA']
        mu = CONFIG['MU']
        sigma = CONFIG['SIGMA']
        x0 = CONFIG['X0'] if CONFIG['X0'] else prices.iloc[-1]
    
    log(f"[Config] Using: θ={theta:.4f}, μ={mu:.2f}, σ={sigma:.4f}, X₀={x0:.2f}")
    
    # 3. Simulation
    engine = OUEngine(theta, mu, sigma, x0)
    t, paths = engine.run()
    
    # 4. Analytical Bounds
    exp_val, upper_bound, lower_bound = compute_theoretical_bounds(t, theta, mu, sigma, x0)
    
    # 5. Visualization
    visualize(t, paths, exp_val, upper_bound, lower_bound, CONFIG['TICKER'])
    
    # Summary Stats
    log("=== RESULTS SUMMARY ===")
    final_values = paths[-1, :]
    log(f"Final Mean (Simulated): ${np.mean(final_values):.2f}")
    log(f"Final Mean (Theoretical): ${exp_val[-1]:.2f}")
    log(f"Final Std (Simulated): ${np.std(final_values):.2f}")
    log(f"Convergence to μ: {100 * abs(np.mean(final_values) - mu) / abs(x0 - mu):.1f}% remaining")
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
