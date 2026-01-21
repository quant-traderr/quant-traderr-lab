"""
Sandpile Pipeline.py
====================
Project: Quant Trader Lab - Self-Organized Criticality
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for analyzing Financial Market Criticality
    using the Bak-Tang-Wiesenfeld (BTW) Sandpile Model.

    It maps market stress (volatility/returns) to "Sand Grains",
    simulating the accumulation of stress and resulting "Avalanches" (Market Crashes/Rallies).
    
    This allows us to detect Self-Organized Criticality (SOC) and potential
    phase transitions in the market state.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches real-market data (BTC-USD) for stress input.
        *NOTE*: The visualization counterpart to this pipeline used a specific local dataset 
        ('BTCUSDT_1M_CLEAN.csv') with 1-minute resolution. This pipeline fetches daily data 
        via yfinance for portability, but for high-fidelity reproduction, users should 
        supply their own high-frequency data.
    2.  **Physics Simulation**: Runs a Cellular Automaton (Sandpile) on a 2D Grid.
    3.  **Analysis**: Tracks Avalanche sizes and System Energy (Total Grains) to 
        identify Critical States.

    NOTE: This is a Pure Analysis Pipeline. Visualization rendering has been removed.

Dependencies:
    pip install numpy pandas yfinance
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKER": "BTC-USD",
    "PERIOD": "365d",     # 1 Year context
    "INTERVAL": "1d",
    
    # Physics (Sandpile Model)
    "GRID_SIZE": 50,      # N x N Lattice
    "CRITICAL_MASS": 4,   # Topple threshold
    "GRAIN_SCALE": 1500,  # Scaling factor for volatility -> grains
    
    # Analysis
    "CRITICAL_THRESHOLD": 50, # Avalanche size considered "Critical"
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches market data to drive the Sandpile stress accumulation.
    
    NOTE: For the visual demonstration, we used 'BTCUSDT_1M_CLEAN.csv' (1-minute data).
    Here we use daily data for portability. Users should replace this with their 
    own high-frequency data source for more granular analysis.
    """
    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")
    log("[Data] NOTE: Using yfinance for portability. For reproduction of visuals, use high-freq local data.")
    
    try:
        df = yf.download(CONFIG['TICKER'], period=CONFIG['PERIOD'], interval=CONFIG['INTERVAL'], progress=False)
        
        if df.empty:
            raise ValueError("No data returned from yfinance.")
            
        # Handle MultiIndex if present
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
             prices = prices.iloc[:, 0]
        
        # Calculate Returns (Stress)
        returns = prices.pct_change().fillna(0)
        
        # Calculate Volatility/Stress Metric
        # In this model, absolute return size determines number of grains dropped
        stress_metric = np.abs(returns)
        
        log(f"[Data] Loaded {len(prices)} periods.")
        log(f"[Data] Max Stress (Abs Return): {stress_metric.max():.2%}")
        
        return stress_metric, prices.index
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        return None, None

# --- MODULE 2: PHYSICS (SANDPILE CORE) ---

class SandpileSystem:
    def __init__(self, N, critical_mass=4):
        self.N = N
        self.critical_mass = critical_mass
        
        # Initialize Empty Grid
        self.grid = np.zeros((N, N), dtype=int)
        
    def add_sand(self, amount):
        """Drops 'amount' grains onto random locations."""
        if amount <= 0:
            return
            
        xs = np.random.randint(0, self.N, amount)
        ys = np.random.randint(0, self.N, amount)
        np.add.at(self.grid, (xs, ys), 1)

    def step(self):
        """
        Evolves the system until stability (or for a fixed number of relaxation steps).
        Returns the total number of topples (avalanche size).
        """
        avalanche_size = 0
        
        # Relaxation loop
        # We limit the loop to avoid infinite loops in extreme cases, 
        # though standard BTW model is guaranteed to stabilize on open boundaries.
        max_sub_steps = 100 
        
        for _ in range(max_sub_steps):
            unstable = self.grid >= self.critical_mass
            if not np.any(unstable):
                break
                
            topple_count = np.sum(unstable)
            avalanche_size += topple_count
            
            # Topple: Decrease unstable sites by 4
            self.grid[unstable] -= 4
            
            # Distribute to neighbors (Vectorized)
            topple_mask = unstable.astype(int)
            
            # Spread (North, South, East, West)
            # CAREFUL: Boundary conditions (Open Boundaries - grains fall off edge)
            
            # Down (i+1)
            self.grid[:-1, :] += topple_mask[1:, :]
            # Up (i-1)
            self.grid[1:, :] += topple_mask[:-1, :]
            # Right (j+1)
            self.grid[:, :-1] += topple_mask[:, 1:]
            # Left (j-1)
            self.grid[:, 1:] += topple_mask[:, :-1]
            
            # Edges are "Open" (sinks), so we just don't add grains that fall off.
            # The slicing above naturally handles this by only adding to valid inner coordinates.
            # Example: grid[:-1, :] refers to rows 0 to N-2. It receives from topple_mask[1:, :] which is rows 1 to N-1.
            # So a topple at row 0 (top) does NOT add to any row above it. Correct.
            
        return avalanche_size

    def get_system_energy(self):
        """Total grains in the system."""
        return np.sum(self.grid)

# --- MODULE 3: ANALYSIS ---

def run_sandpile_analysis(stress_series, dates):
    """
    Feeds market stress into the sandpile and records critical events.
    """
    if stress_series is None:
        return pd.DataFrame()

    log("=== INITIALIZING PHYSICS ENGINE ===")
    
    sim = SandpileSystem(
        N=CONFIG['GRID_SIZE'],
        critical_mass=CONFIG['CRITICAL_MASS']
    )
    
    log(f"Grid: {sim.N}x{sim.N} ({sim.N**2} Cells)")
    log(f"Grain Scale: {CONFIG['GRAIN_SCALE']}")
    
    results = []
    
    log("=== STARTING SIMULATION ===")
    
    start_time = time.time()
    
    # Calculate grains for all steps upfront
    grains_series = (stress_series * CONFIG['GRAIN_SCALE']).astype(int)
    grains_series = np.maximum(grains_series, 1) # Min 1 grain per step
    
    for i, (date, grains) in enumerate(zip(dates, grains_series)):
        
        # 1. Perturb System (Drop Grains)
        sim.add_sand(grains)
        
        # 2. Relax System (Avalanche)
        avalanche_size = sim.step()
        
        # 3. Record State
        energy = sim.get_system_energy()
        
        results.append({
            "Date": date,
            "Input_Stress": stress_series.iloc[i],
            "Grains_Dropped": grains,
            "Avalanche_Size": avalanche_size,
            "System_Energy": energy
        })
        
        # Sparse logging
        if i % 50 == 0:
             log(f"[{date.date()}] Stress={stress_series.iloc[i]:.4f} | Avalanche={avalanche_size}")
             
    duration = time.time() - start_time
    log(f"Simulation Complete in {duration:.2f}s")
    
    return pd.DataFrame(results)

def report_findings(df):
    """
    Identifies Critical Avalanches.
    """
    log("=== CRITICALITY REPORT ===")
    
    if df.empty:
        log("No results to report.")
        return
        
    # Define Critical Event
    critical_events = df[df['Avalanche_Size'] > CONFIG['CRITICAL_THRESHOLD']]
    
    log(f"Total Time Steps: {len(df)}")
    log(f"Critical Events Found: {len(critical_events)}")
    log(f"Criticality Ratio: {len(critical_events)/len(df):.2%}")
    
    if not critical_events.empty:
        log("\nTop 5 Critical Market Events (Predicted by Physics):")
        top_events = critical_events.sort_values(by='Avalanche_Size', ascending=False).head(5)
        
        for _, row in top_events.iterrows():
            log(f">> {row['Date'].date()} | Avalanche: {row['Avalanche_Size']} | Input Stress: {row['Input_Stress']:.2%}")
            
    # Correlation Check
    # Does Stress correlate with Avalanche size?
    # In SOC, avalanches can happen with small triggers if system is critical.
    corr = df['Input_Stress'].corr(df['Avalanche_Size'])
    log(f"\nStress-Avalanche Correlation: {corr:.3f}")
    if corr < 0.5:
        log(">> Note: Low correlation implies Endogenous Instability (SOC) dominates Exogenous Shock.")
    else:
        log(">> Note: High correlation implies Market Driven dynamics.")

# --- MAIN ---

def main():
    log("=== SANDPILE MARKET PIPELINE ===")
    
    # 1. Context
    stress, dates = fetch_market_data()
    
    # 2. Physics & Analysis
    if stress is not None:
        results_df = run_sandpile_analysis(stress, dates)
        
        # 3. Report
        report_findings(results_df)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
