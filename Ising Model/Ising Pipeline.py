"""
Ising Pipeline.py
================
Project: Quant Trader Lab - Phase Transition Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for analyzing Financial Market Phase Transitions 
    using the Ising Model of Ferromagnetism.

    It maps market microstructure (volatility/sentiment) to Spin Dynamics,
    simulating the transition from High Temperature (Chaos) to Low Temperature (Order)
    to identify Critical Points (Phase Transitions).

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches strict real-market data (BTC-USD) for context.
    2.  **Physics Simulation**: Runs a Metropolis-Hastings Monte Carlo simulation 
        on a 3D Lattice to model "Social Sentiment" dynamics.
    3.  **Analysis**: Calculates System Metrics (Magnetization, Energy, Susceptibility)
        to detect Criticality.

    NOTE: This is a Pure Analysis Pipeline. Visualization rendering has been removed.

Dependencies:
    pip install numpy pandas yfinance scipy
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER": "BTC-USD",
    "PERIOD": "365d",     # 1 Year context
    "INTERVAL": "1d",
    
    # Physics (Ising Model)
    "GRID_SIZE": 12,      # N x N x N Lattice
    "STEPS_PER_TEMP": 100, # Monte Carlo steps per temperature
    "TEMP_RANGE": (10.0, 0.1), # Annealing from Chaos (10) to Order (0.1)
    "TEMP_STEPS": 50,     # Number of temperature increments
    
    # Control
    "TARGET_SENTIMENT": 0.75, # Target "Bullish" ratio (Blue/Up spins)
    "CONTROL_GAIN": 5.0,      # Strength of the constraint field
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches market data to establish the 'Temperature' context.
    In a full rigorous model, Volatility would map to Temperature.
    Here we fetch it to confirm market connectivity and regime.
    """
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")
    
    try:
        df = yf.download(CONFIG['TICKER'], period=CONFIG['PERIOD'], interval=CONFIG['INTERVAL'], progress=False)
        
        if df.empty:
            raise ValueError("No data returned from yfinance.")
            
        # Handle MultiIndex
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
             prices = prices.iloc[:, 0]
             
        # Calculate Volatility (proxy for System Temperature)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(365)
        
        log(f"[Data] Loaded {len(prices)} days.")
        log(f"[Data] Annualized Volatility: {volatility:.2%}")
        
        return volatility
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        return None

# --- MODULE 2: PHYSICS (ISING CORE) ---

class IsingSystem:
    def __init__(self, N, target_sentiment, gain):
        self.N = N
        self.target = target_sentiment
        self.gain = gain
        
        # Initialize Random Chaos (High Temp)
        np.random.seed(42)
        self.lattice = np.random.choice([-1, 1], size=(N, N, N))
        
    def _get_neighbor_sum(self, x, y, z):
        """Periodic Boundary Conditions (Toroidal Topology)"""
        N = self.N
        lat = self.lattice
        return (
            lat[(x+1)%N, y, z] + lat[(x-1)%N, y, z] +
            lat[x, (y+1)%N, z] + lat[x, (y-1)%N, z] +
            lat[x, y, (z+1)%N] + lat[x, y, (z-1)%N]
        )

    def metropolis_step(self, temp):
        """
        Performs one Monte Carlo sweep over the entire lattice.
        Includes a 'Social Field' bias to enforce target sentiment.
        """
        N = self.N
        lat = self.lattice
        
        # Dynamic Field (Bias) Calculation
        # Maps to an external magnetic field H proportional to the deviation from target
        # H(t) = K * (Target - Current_Magnetization)
        # Note: spins are -1, 1. target is 0.75 ratio of +1s.
        # Ratio 0.75 means Magnetization M = (0.75 * 1) + (0.25 * -1) = 0.5
        
        current_up_ratio = np.mean(lat == 1)
        bias = self.gain * (self.target - current_up_ratio)
        
        # Vectorized updates are hard with nearest neighbors due to dependency,
        # but for N=12, a simple loop is fast enough (~1700 sites)
        change_count = 0
        
        # To avoid correlation, we pick random sites or checkerboard. 
        # For simplicity in this pipeline, we iterate random sites N^3 times.
        for _ in range(N**3):
            x, y, z = np.random.randint(0, N, 3)
            spin = lat[x, y, z]
            
            neighbor_sum = self._get_neighbor_sum(x, y, z)
            
            # Hamiltonian H = -J * sum(si*sj) - H_field * sum(si)
            # dE for flip s -> -s is 2*s*(neighbors + H_field)
            
            dE = 2 * spin * (neighbor_sum + bias)
            
            # Metropolis Criterion
            if dE <= 0 or np.random.rand() < np.exp(-dE / temp):
                lat[x, y, z] *= -1
                change_count += 1
                
        return change_count

    def get_observables(self):
        """Calculate system metrics."""
        M = np.mean(self.lattice) # Magnetization [-1, 1]
        
        # Energy per spin approx (neglecting the bias term for pure Ising comparison)
        # E = -0.5 * sum(s_i * s_j) / N^3
        # Since we don't have a neighbor list pre-calc, we can just do a quick sum
        # shift array to get neighbors
        E = 0
        for axis in range(3):
            E += np.sum(self.lattice * np.roll(self.lattice, 1, axis=axis))
        E = -E / (self.N**3)
        
        return M, E

# --- MODULE 3: ANALYSIS ---

def run_phase_transition_analysis(volatility_context):
    """
    Simulates the market cooling down from Chaos to Order.
    Tracks Susceptibility (Variance of Magnetization) to find Critical Temp.
    """
    log("=== INITIALIZING PHYSICS ENGINE ===")
    
    sim = IsingSystem(
        N=CONFIG['GRID_SIZE'], 
        target_sentiment=CONFIG['TARGET_SENTIMENT'], 
        gain=CONFIG['CONTROL_GAIN']
    )
    
    log(f"Grid: {sim.N}x{sim.N}x{sim.N} ({sim.N**3} Agents)")
    log(f"Target Sentiment: {CONFIG['TARGET_SENTIMENT']:.0%}")
    
    # Annealing Schedule
    temps = np.linspace(CONFIG['TEMP_RANGE'][0], CONFIG['TEMP_RANGE'][1], CONFIG['TEMP_STEPS'])
    
    results = []
    
    log("=== STARTING SIMULATION (ANNEALING) ===")
    
    start_time = time.time()
    
    for T in temps:
        # Thermalsize (Run steps to reach equilibrium at T)
        changes = 0
        magnetizations = []
        
        for _ in range(CONFIG['STEPS_PER_TEMP']):
            c = sim.metropolis_step(T)
            changes += c
            # Sample observables last 20% of steps
            if _ > CONFIG['STEPS_PER_TEMP'] * 0.8:
                m, _ = sim.get_observables()
                magnetizations.append(m)
        
        # Calculate Susceptibility (Chi) = Variance(M) / T
        # High Chi indicates Phase Transition
        avg_m = np.mean(magnetizations) if magnetizations else 0
        var_m = np.var(magnetizations) if magnetizations else 0
        chi = var_m / T
        
        results.append({
            "Temp": T,
            "Magnetization": avg_m,
            "Susceptibility": chi,
            "Activity": changes
        })
        
        # Sparse logging
        if T == temps[0] or T == temps[-1] or (int(T*10) % 20 == 0):
             log(f"T={T:.2f} | M={avg_m:.3f} | Chi={chi:.4f}")
             
    duration = time.time() - start_time
    log(f"Simulation Complete in {duration:.2f}s")
    
    return pd.DataFrame(results)

def report_findings(df):
    """
    Identifies Critical Phase Transition.
    """
    log("=== CRITICALITY REPORT ===")
    
    # Critical Point is max Susceptibility
    critical_point = df.loc[df['Susceptibility'].idxmax()]
    
    Tc = critical_point['Temp']
    Mc = critical_point['Magnetization']
    
    log(f"Critical Temperature (Tc): {Tc:.2f}")
    log(f"Magnetization at Tc: {Mc:.3f}")
    
    # Current Market Assessment (Low Temp regime)
    final_state = df.iloc[-1]
    
    log(f"Final State (Order): M={final_state['Magnetization']:.3f}")
    
    if abs(final_state['Magnetization']) > 0.5:
        log(">> Market Phase: ORDERED (Strong Trend)")
        if final_state['Magnetization'] > 0:
            log(">> Direction: BULLISH")
        else:
            log(">> Direction: BEARISH")
    else:
        log(">> Market Phase: DISORDERED (Chop/Volatile)")

# --- MAIN ---


def main():

    
    log("=== ISING MARKET PIPELINE ===")
    
    # 1. Context
    vol = fetch_market_data()
    if vol is None:
        log("[Warning] Proceeding with simulation without live volatility context.")
    
    # 2. Physics & Analysis
    results_df = run_phase_transition_analysis(vol)
    
    # 3. Report
    report_findings(results_df)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
