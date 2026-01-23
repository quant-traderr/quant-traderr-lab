"""
MST Pipeline.py
===============
Project: Quant Trader Lab - Market Correlation Structure Analysis
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for analyzing Financial Market Correlation Structure
    using Minimum Spanning Tree (MST) algorithms from Graph Theory.
    
    It maps stock correlations to a network graph, then applies Kruskal's/Prim's algorithm
    to extract the Market Skeleton - the most critical connections that hold the
    market together.
    
    Pipeline Steps:
    1.  **Data Acquisition**: Fetches real S&P 500 stock data via yfinance
    2.  **Correlation Analysis**: Computes distance matrix from correlation matrix
    3.  **Graph Construction**: Builds complete graph with distance-weighted edges
    4.  **MST Computation**: Extracts minimum spanning tree using NetworkX
    5.  **Analysis**: Identifies central hubs, sector clusters, and network metrics

    NOTE: This is a Pure Analysis Pipeline. Visualization rendering has been removed.
    
    # Rendering Note:
    # The original visual.py contains a 3D PyVista-based rendering pipeline.
    # It creates a 12-second narrative video showing the "Market Skeleton"
    # with cyberpunk aesthetics, sector-colored nodes, and emission glows.
    # Rendering code has been intentionally removed for portability and
    # to keep this pipeline focused on the core MST computation.
    
    # Lookahead Bias Note:
    # ⚠️  IMPORTANT: This pipeline computes the MST over a FULL PERIOD (e.g., 6 months).
    # It answers: "What was the correlation structure DURING this period?"
    # This is VALID for:
    #   - Static historical analysis
    #   - Understanding market structure in hindsight
    #   - Research and visualization
    #
    # This is NOT VALID for:
    #   - Live trading strategies (without modification)
    #   - Backtesting (as-is, would have lookahead bias)
    #
    # For trading applications, you would need to use EXPANDING or ROLLING windows:
    #   - At time T, only use data up to T-1 to compute correlations
    #   - Recompute MST at each rebalancing period with historical data only
    #   - See compute_correlation_distance() function for implementation notes

Dependencies:
    pip install numpy pandas yfinance networkx scipy
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
from datetime import datetime

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Data
    "TICKERS": [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'ADBE', 'CRM', 'CSCO', 'ACN',
        # Financials
        'JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SCHW', 'AXP', 'SPGI',
        # Healthcare
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        # Consumer
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
    ],
    "PERIOD": "6mo",
    "INTERVAL": "1d",
    
    # Sector Mapping (for analysis)
    "SECTORS": {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'NVDA': 'Tech', 'META': 'Tech',
        'AVGO': 'Tech', 'ADBE': 'Tech', 'CRM': 'Tech', 'CSCO': 'Tech', 'ACN': 'Tech',
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'MS': 'Financials',
        'GS': 'Financials', 'BLK': 'Financials', 'C': 'Financials', 'SCHW': 'Financials',
        'AXP': 'Financials', 'SPGI': 'Financials',
        'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
        'MRK': 'Healthcare', 'PFE': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
        'DHR': 'Healthcare', 'BMY': 'Healthcare',
        'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'MCD': 'Consumer',
        'NKE': 'Consumer', 'SBUX': 'Consumer', 'LOW': 'Consumer', 'TJX': 'Consumer',
        'BKNG': 'Consumer', 'CMG': 'Consumer',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
        'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
        'OXY': 'Energy', 'HAL': 'Energy',
    },
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def fetch_market_data():
    """
    Fetches S&P 500 stock data and computes returns.
    Returns cleaned price DataFrame.
    """
    log(f"[Data] Fetching {len(CONFIG['TICKERS'])} stocks ({CONFIG['PERIOD']})...")
    
    try:
        data = yf.download(
            CONFIG['TICKERS'], 
            period=CONFIG['PERIOD'], 
            interval=CONFIG['INTERVAL'],
            progress=False,
            threads=True
        )['Close']
        
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        
        # Remove tickers with missing data
        data = data.dropna(axis=1, how='any')
        valid_tickers = data.columns.tolist()
        
        log(f"[Data] Successfully loaded {len(valid_tickers)} stocks")
        log(f"[Data] Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data, valid_tickers
    
    except Exception as e:
        log(f"[Error] Data fetch failed: {e}")
        return None, []

# --- MODULE 2: CORRELATION & DISTANCE ---

def compute_correlation_distance(data):
    """
    Computes correlation matrix and transforms to distance matrix.
    Distance metric: d = sqrt(2(1 - correlation))
    
    This ensures:
    - Perfect correlation (ρ=1) → distance = 0
    - No correlation (ρ=0) → distance = sqrt(2)
    - Perfect anti-correlation (ρ=-1) → distance = 2
    
    ⚠️  LOOKAHEAD BIAS WARNING:
    This function computes correlation over the ENTIRE dataset provided.
    For static analysis, this is acceptable.
    
    For TRADING/BACKTESTING, you must:
    1. Use expanding window: df.iloc[:t].corr() at each time t
    2. Or rolling window: df.rolling(window=252).corr() 
    3. Never use future data to compute past correlations
    
    Example for bias-free implementation:
        # Bad (lookahead bias):
        corr = returns.corr()  # Uses all data
        
        # Good (no lookahead):
        corr_at_t = returns.iloc[:t].corr()  # Only use data up to time t
    """
    log("[Analysis] Computing log returns...")
    returns = np.log(data / data.shift(1)).dropna()
    
    log("[Analysis] Computing correlation matrix...")
    corr_matrix = returns.corr()  # ⚠️ Uses entire period - OK for static analysis only
    
    log("[Analysis] Transforming to distance matrix...")
    # Distance metric from correlation
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    
    # Statistics
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    log(f"[Analysis] Average pairwise correlation: {avg_corr:.3f}")
    
    return corr_matrix, dist_matrix

# --- MODULE 3: GRAPH & MST ---

def build_complete_graph(valid_tickers, dist_matrix):
    """
    Constructs a complete graph where each stock is a node
    and edge weights are correlation distances.
    """
    log("[Graph] Building complete graph...")
    
    G = nx.Graph()
    
    # Add edges with distance weights
    for i, stock1 in enumerate(valid_tickers):
        for j, stock2 in enumerate(valid_tickers):
            if i < j:  # Avoid duplicate edges
                distance = dist_matrix.loc[stock1, stock2]
                G.add_edge(stock1, stock2, weight=distance)
    
    log(f"[Graph] Nodes: {G.number_of_nodes()}")
    log(f"[Graph] Edges: {G.number_of_edges()}")
    
    return G

def compute_mst(G):
    """
    Computes Minimum Spanning Tree using NetworkX.
    Uses Kruskal's/Prim's algorithm under the hood.
    """
    log("[MST] Computing Minimum Spanning Tree...")
    
    start_time = time.time()
    mst = nx.minimum_spanning_tree(G, weight='weight')
    duration = time.time() - start_time
    
    log(f"[MST] Computation complete in {duration:.3f}s")
    log(f"[MST] Tree edges: {mst.number_of_edges()}")
    
    # MST should have exactly N-1 edges for N nodes
    assert mst.number_of_edges() == mst.number_of_nodes() - 1, "MST property violated!"
    
    return mst

# --- MODULE 4: ANALYSIS ---

def analyze_mst_structure(mst, sectors):
    """
    Analyzes MST topology and identifies key features:
    - Central hub (most connected node)
    - Degree distribution
    - Sector homophily (do sectors cluster?)
    """
    log("=== MST STRUCTURAL ANALYSIS ===")
    
    # 1. Degree Analysis
    degree_dict = dict(mst.degree())
    hub_node = max(degree_dict, key=degree_dict.get)
    hub_degree = degree_dict[hub_node]
    hub_sector = sectors.get(hub_node, 'Unknown')
    
    log(f"Central Hub: {hub_node} (Sector: {hub_sector})")
    log(f"Hub Degree: {hub_degree} connections")
    
    # Average degree
    avg_degree = sum(degree_dict.values()) / len(degree_dict)
    log(f"Average Degree: {avg_degree:.2f}")
    
    # 2. Sector Clustering
    log("\n=== SECTOR CLUSTERING ===")
    
    sector_edges = {}  # Count edges within same sector
    cross_sector_edges = 0
    
    for u, v in mst.edges():
        sector_u = sectors.get(u, 'Unknown')
        sector_v = sectors.get(v, 'Unknown')
        
        if sector_u == sector_v:
            sector_edges[sector_u] = sector_edges.get(sector_u, 0) + 1
        else:
            cross_sector_edges += 1
    
    total_edges = mst.number_of_edges()
    homophily_ratio = (total_edges - cross_sector_edges) / total_edges
    
    log(f"Intra-sector edges: {total_edges - cross_sector_edges}")
    log(f"Cross-sector edges: {cross_sector_edges}")
    log(f"Sector Homophily: {homophily_ratio:.2%}")
    
    # 3. Tree Diameter (longest path)
    # For large graphs this can be slow, so we use approximation
    if mst.number_of_nodes() < 100:
        diameter = nx.diameter(mst)
        log(f"\nTree Diameter: {diameter} hops")
    
    return hub_node, hub_sector, hub_degree, homophily_ratio

def analyze_sector_connectivity(mst, sectors):
    """
    Analyzes which sectors are most interconnected in the MST.
    """
    log("\n=== SECTOR CONNECTIVITY MATRIX ===")
    
    # Get unique sectors
    unique_sectors = set(sectors.values())
    
    # Build connectivity matrix
    connectivity = {s: {s2: 0 for s2 in unique_sectors} for s in unique_sectors}
    
    for u, v in mst.edges():
        sector_u = sectors.get(u, 'Unknown')
        sector_v = sectors.get(v, 'Unknown')
        
        # Symmetric
        connectivity[sector_u][sector_v] += 1
        connectivity[sector_v][sector_u] += 1
    
    # Print matrix
    for s1 in sorted(unique_sectors):
        connections = [f"{s2[:4]}:{connectivity[s1][s2]}" for s2 in sorted(unique_sectors) if connectivity[s1][s2] > 0]
        if connections:
            log(f"{s1:12s} → {', '.join(connections)}")

# --- MODULE 5: RISK METRICS ---

def compute_mst_risk_metrics(mst, dist_matrix):
    """
    Computes risk-related metrics from the MST structure.
    """
    log("\n=== RISK METRICS ===")
    
    # 1. MST Length (total edge weight)
    # Shorter MST = more correlated market = higher systemic risk
    mst_length = sum(dist_matrix.loc[u, v] for u, v in mst.edges())
    avg_edge_weight = mst_length / mst.number_of_edges()
    
    log(f"MST Total Length: {mst_length:.3f}")
    log(f"Average Edge Weight: {avg_edge_weight:.3f}")
    
    # Interpretation
    if avg_edge_weight < 0.5:
        log("⚠️  HIGH SYSTEMIC RISK: Market is highly correlated")
    elif avg_edge_weight < 1.0:
        log("⚙️  MODERATE RISK: Normal correlation structure")
    else:
        log("✅ LOW SYSTEMIC RISK: Market shows diversification")
    
    return mst_length, avg_edge_weight

# --- MAIN ---

def main():
    log("=== MINIMUM SPANNING TREE PIPELINE ===")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Data Acquisition
    data, valid_tickers = fetch_market_data()
    if data is None or len(valid_tickers) == 0:
        log("[Error] Cannot proceed without data.")
        return
    
    # 2. Correlation & Distance
    corr_matrix, dist_matrix = compute_correlation_distance(data)
    
    # 3. Graph Construction
    G = build_complete_graph(valid_tickers, dist_matrix)
    
    # 4. MST Computation
    mst = compute_mst(G)
    
    # 5. Structural Analysis
    hub, hub_sector, hub_deg, homophily = analyze_mst_structure(mst, CONFIG['SECTORS'])
    
    # 6. Sector Analysis
    analyze_sector_connectivity(mst, CONFIG['SECTORS'])
    
    # 7. Risk Metrics
    mst_len, avg_weight = compute_mst_risk_metrics(mst, dist_matrix)
    
    log("\n=== PIPELINE FINISHED ===")
    
    # Return results for programmatic use
    return {
        'mst': mst,
        'correlation_matrix': corr_matrix,
        'distance_matrix': dist_matrix,
        'hub_node': hub,
        'hub_sector': hub_sector,
        'hub_degree': hub_deg,
        'sector_homophily': homophily,
        'mst_length': mst_len,
        'avg_edge_weight': avg_weight,
    }

if __name__ == "__main__":
    results = main()

