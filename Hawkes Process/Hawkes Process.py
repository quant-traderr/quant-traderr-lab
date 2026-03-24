"""
Hawkes_Pipeline.py
==================
Project : Quant Trader Lab – Market Microstructure Simulation
Author  : quant.traderr (Instagram)
License : MIT

Description
-----------
A production-ready pipeline for simulating and visualizing market
microstructure dynamics using a **Hawkes process** — a self-exciting
point process where each trade arrival increases the probability of
subsequent trades.

The Hawkes process captures the clustering behavior observed in real
high-frequency order flow: volatility breeds volatility.

Pipeline Steps
--------------
    1. **Simulation**    : Generates trade arrivals via Ogata's thinning
                           algorithm, then builds mid-price, spread, and
                           order-flow imbalance on top.
    2. **Event Detection**: Identifies key microstructure events
                           (order cascades, liquidity droughts, etc.).
    3. **Rendering**     : Single static matplotlib frame (final state).

Dependencies
------------
    pip install numpy matplotlib

Usage
-----
    python Hawkes_Pipeline.py
"""

import os
import sys
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Hawkes Process ────────────────────────────────────────────────────
    "MU":       0.8,       # Baseline intensity  (events per unit time)
    "ALPHA":    0.65,      # Excitation amplitude
    "BETA":     1.5,       # Decay rate           (branching ratio α/β ≈ 0.43)
    "T_MAX":    500,       # Simulation time horizon

    # ── Microstructure ────────────────────────────────────────────────────
    "IMPACT_FACTOR":  0.02,   # Price impact per trade  (κ)
    "SPREAD_BASE":    0.5,    # Base bid-ask spread
    "SPREAD_GAMMA":   0.3,    # Spread sensitivity to intensity excess

    # ── Output ───────────────────────────────────────────────────────────
    "RESOLUTION":     (1920, 1080),
    "SEED":           42,
    "OUTPUT_FILE":    os.path.join(os.path.dirname(__file__), "Hawkes_Output.png"),
    "LOG_FILE":       os.path.join(os.path.dirname(__file__), "hawkes_pipeline.log"),
}

THEME = {
    "BG":        "#0e0e0e",
    "GRID":      "#1f1f1f",
    "SPINE":     "#333333",
    "TEXT":      "#c0c0c0",
    "WHITE":     "#ffffff",
    "FONT":      "DejaVu Sans",

    # Data colors
    "PRICE":     "#00d4ff",    # Cyan
    "BUY":       "#00ff41",    # Neon green
    "SELL":      "#ff0055",    # Neon pink
    "INTENSITY": "#ff9800",    # Orange
    "SPREAD":    "#ff1493",    # Deep pink
    "IMBALANCE": "#ffcc00",    # Yellow
    "TRADE_RATE":"#bb66ff",    # Purple
}

# ─── UTILS ───────────────────────────────────────────────────────────────────

def log(msg):
    """Timestamped console + file logger."""
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    try:
        print(formatted)
    except UnicodeEncodeError:
        print(formatted.encode("ascii", errors="replace").decode())
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 :  HAWKES SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_hawkes():
    """
    Simulate a univariate Hawkes process using Ogata's thinning algorithm.

    Returns
    -------
    sim : dict with keys:
        event_times   – array of trade arrival times
        event_signs   – +1 (buy) / -1 (sell) for each trade
        t_grid        – regular time grid for plotting
        intensity     – λ(t) evaluated on t_grid
        mid_price     – mid-price path on t_grid
        spread        – bid-ask spread on t_grid
        imbalance     – cumulative order-flow imbalance on t_grid
        trade_rate    – rolling trade count on t_grid
        events        – list of (time, label) annotation events
    """
    log("[Simulation] Starting Hawkes process (Ogata thinning)...")

    mu    = CONFIG["MU"]
    alpha = CONFIG["ALPHA"]
    beta  = CONFIG["BETA"]
    T     = CONFIG["T_MAX"]

    np.random.seed(CONFIG["SEED"])

    # ── Ogata's Thinning Algorithm ────────────────────────────────────────
    event_times = []
    t = 0.0

    while t < T:
        # Current intensity (sum of all previous contributions)
        lam = mu + alpha * sum(
            np.exp(-beta * (t - ti)) for ti in event_times
        )

        # Upper bound for thinning
        lam_bar = lam

        # Next candidate event (exponential inter-arrival)
        u1 = np.random.rand()
        dt = -np.log(u1) / lam_bar
        t += dt

        if t >= T:
            break

        # Recalculate exact intensity at candidate time
        lam_t = mu + alpha * sum(
            np.exp(-beta * (t - ti)) for ti in event_times
        )

        # Accept / reject
        u2 = np.random.rand()
        if u2 <= lam_t / lam_bar:
            event_times.append(t)

    event_times = np.array(event_times)
    n_events = len(event_times)
    log(f"[Simulation] Generated {n_events} trade events over [0, {T}].")

    # ── Trade Signs ───────────────────────────────────────────────────────
    # Slight buy-bias (realistic order flow)
    event_signs = np.where(np.random.rand(n_events) < 0.52, 1, -1)

    # ── Regular Time Grid ─────────────────────────────────────────────────
    n_grid = 2000
    t_grid = np.linspace(0, T, n_grid)

    # ── Intensity on Grid ─────────────────────────────────────────────────
    log("[Simulation] Computing intensity on time grid...")
    intensity = np.full(n_grid, mu, dtype=float)
    for ti in event_times:
        mask = t_grid > ti
        intensity[mask] += alpha * np.exp(-beta * (t_grid[mask] - ti))

    # ── Mid-Price ─────────────────────────────────────────────────────────
    kappa = CONFIG["IMPACT_FACTOR"]
    mid_price = np.zeros(n_grid)
    mid_price[0] = 100.0  # Starting price

    # Assign each trade to nearest grid point
    trade_grid_idx = np.searchsorted(t_grid, event_times, side='right') - 1
    trade_grid_idx = np.clip(trade_grid_idx, 0, n_grid - 1)

    # Accumulate trade impacts at grid points
    price_impacts = np.zeros(n_grid)
    for k in range(n_events):
        idx = trade_grid_idx[k]
        local_intensity = intensity[idx]
        noise = 1.0 + 0.3 * np.random.randn()
        price_impacts[idx] += event_signs[k] * kappa * np.sqrt(local_intensity) * noise

    # Walk the price forward
    for i in range(1, n_grid):
        mid_price[i] = mid_price[i - 1] + price_impacts[i]

    # ── Spread ────────────────────────────────────────────────────────────
    base   = CONFIG["SPREAD_BASE"]
    gamma  = CONFIG["SPREAD_GAMMA"]
    spread = base + gamma * np.maximum(intensity - mu, 0)
    # Add small noise
    spread += 0.05 * np.random.randn(n_grid)
    spread = np.maximum(spread, 0.05)

    # ── Order Flow Imbalance ──────────────────────────────────────────────
    cum_imbalance = np.zeros(n_grid)
    for k in range(n_events):
        idx = trade_grid_idx[k]
        if idx < n_grid:
            cum_imbalance[idx] += event_signs[k]
    cum_imbalance = np.cumsum(cum_imbalance)
    # Normalize to [-1, 1] range for plotting
    max_abs = max(np.abs(cum_imbalance).max(), 1)
    imbalance = cum_imbalance / max_abs

    # ── Rolling Trade Rate ────────────────────────────────────────────────
    window = 50  # grid points
    trade_count_grid = np.zeros(n_grid)
    for idx in trade_grid_idx:
        if idx < n_grid:
            trade_count_grid[idx] += 1
    trade_rate = np.convolve(trade_count_grid, np.ones(window), mode='same') / window
    # Normalize
    trade_rate_max = max(trade_rate.max(), 1)
    trade_rate = trade_rate / trade_rate_max

    # ── Event Detection ───────────────────────────────────────────────────
    log("[Simulation] Detecting microstructure events...")
    events = _detect_events(t_grid, intensity, mid_price, spread, mu)

    log(f"[Simulation] Detected {len(events)} annotation events.")

    return {
        "event_times":  event_times,
        "event_signs":  event_signs,
        "t_grid":       t_grid,
        "intensity":    intensity,
        "mid_price":    mid_price,
        "spread":       spread,
        "imbalance":    imbalance,
        "trade_rate":   trade_rate,
        "events":       events,
        "trade_grid_idx": trade_grid_idx,
    }


def _detect_events(t_grid, intensity, mid_price, spread, mu):
    """
    Scan the simulation for notable microstructure events.

    Returns list of (grid_index, label) tuples.
    """
    events = []
    n = len(t_grid)
    used_zones = []  # Prevent overlapping annotations

    def _zone_clear(idx, min_gap=80):
        return all(abs(idx - z) > min_gap for z in used_zones)

    # 1. Order Cascades — intensity > 3× baseline
    cascade_mask = intensity > 3 * mu
    if cascade_mask.any():
        # Find peaks of intensity in cascade regions
        diff = np.diff(cascade_mask.astype(int))
        starts = np.where(diff == 1)[0]
        for s in starts:
            end = s + np.argmax(~cascade_mask[s:]) if not cascade_mask[s:].all() else n - 1
            peak_idx = s + np.argmax(intensity[s:end + 1])
            if _zone_clear(peak_idx):
                events.append((peak_idx, "Order Cascade"))
                used_zones.append(peak_idx)
                if len(events) >= 2:
                    break  # Max 2 cascades

    # 2. Flash Dislocation — sharp price move
    price_diff = np.abs(np.diff(mid_price))
    threshold = np.percentile(price_diff, 99.5)
    flash_indices = np.where(price_diff > threshold)[0]
    for idx in flash_indices:
        if _zone_clear(idx):
            events.append((idx, "Flash Dislocation"))
            used_zones.append(idx)
            break

    # 3. Liquidity Drought — spread spike
    spread_mean = spread.mean()
    drought_mask = spread > 2.0 * spread_mean
    if drought_mask.any():
        drought_indices = np.where(drought_mask)[0]
        # Pick the peak spread
        peak_idx = drought_indices[np.argmax(spread[drought_indices])]
        if _zone_clear(peak_idx):
            events.append((peak_idx, "Liquidity Drought"))
            used_zones.append(peak_idx)

    # 4. Calm Regime — intensity drops near baseline after a cluster
    calm_mask = intensity < 1.1 * mu
    calm_after_storm = np.diff(calm_mask.astype(int))
    calm_starts = np.where(calm_after_storm == 1)[0]
    for idx in calm_starts:
        if idx > n // 4 and _zone_clear(idx):
            events.append((idx, "Calm Regime"))
            used_zones.append(idx)
            break

    # 5. Mean Reversion — price returns toward local mean after dislocation
    if len(mid_price) > 200:
        rolling_mean = np.convolve(mid_price, np.ones(100) / 100, mode='same')
        deviation = np.abs(mid_price - rolling_mean)
        revert_candidates = np.where(
            (deviation[1:] < deviation[:-1]) & (deviation[:-1] > np.percentile(deviation, 95))
        )[0]
        for idx in revert_candidates:
            if _zone_clear(idx):
                events.append((idx, "Mean Reversion"))
                used_zones.append(idx)
                break

    # Sort by time
    events.sort(key=lambda x: x[0])
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 :  RENDERING  (matplotlib, multi-panel timelapse)
# ═══════════════════════════════════════════════════════════════════════════════

def render_static(sim):
    """Renders a single static image showing the full simulation."""
    try:
        # Use all data
        t      = sim["t_grid"]
        price  = sim["mid_price"]
        inten  = sim["intensity"]
        spread = sim["spread"]
        imbal  = sim["imbalance"]
        trate  = sim["trade_rate"]

        if len(t) < 5:
            return False

        # Full axis limits (fixed across all frames for smooth timelapse)
        t_full      = sim["t_grid"]
        price_full  = sim["mid_price"]
        inten_full  = sim["intensity"]

        # ── Figure Setup ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs  = gridspec.GridSpec(
            3, 1,
            height_ratios=[3, 1, 1],
            hspace=0.10,
            left=0.07, right=0.95, top=0.90, bottom=0.08,
        )
        ax_price  = fig.add_subplot(gs[0])
        ax_inten  = fig.add_subplot(gs[1], sharex=ax_price)
        ax_micro  = fig.add_subplot(gs[2], sharex=ax_price)

        axes = [ax_price, ax_inten, ax_micro]

        # ── Common Styling ────────────────────────────────────────────────
        for ax in axes:
            ax.set_facecolor(THEME["BG"])
            ax.tick_params(axis="both", which="major", labelsize=12,
                           colors=THEME["TEXT"], direction="in", length=5)
            ax.tick_params(axis="both", which="minor",
                           colors=THEME["TEXT"], direction="in", length=3)
            for spine in ax.spines.values():
                spine.set_color(THEME["SPINE"])
                spine.set_linewidth(0.6)
            ax.yaxis.grid(True, linewidth=0.3, alpha=0.45, color=THEME["GRID"])
            ax.xaxis.grid(False)
            ax.set_xlim(0, t_full[-1])

        # Hide x-tick labels on upper panels
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_inten.get_xticklabels(), visible=False)

        # ── Panel 1: Mid-Price ────────────────────────────────────────────
        ax_price.plot(t, price, color=THEME["PRICE"], linewidth=2.0, zorder=3)

        # Buy / sell markers (subsample for performance)
        current_trades = sim["event_times"][sim["event_times"] <= t[-1]]
        current_signs  = sim["event_signs"][:len(current_trades)]
        current_idx    = sim["trade_grid_idx"][:len(current_trades)]

        # Subsample markers for visual clarity
        step = max(1, len(current_trades) // 150)
        show_idx    = current_idx[::step]
        show_signs  = current_signs[::step]
        valid = show_idx < data_end_idx

        buys  = show_idx[valid & (show_signs == 1)]
        sells = show_idx[valid & (show_signs == -1)]

        if len(buys) > 0:
            ax_price.scatter(t_full[buys], price_full[buys], c=THEME["BUY"],
                             s=18, alpha=0.8, zorder=4, label="Buys")
        if len(sells) > 0:
            ax_price.scatter(t_full[sells], price_full[sells], c=THEME["SELL"],
                             s=18, alpha=0.8, zorder=4, label="Sells")

        price_margin = (price_full.max() - price_full.min()) * 0.08
        ax_price.set_ylim(price_full.min() - price_margin,
                          price_full.max() + price_margin)
        ax_price.set_ylabel("Mid-Price", color=THEME["TEXT"], fontsize=14,
                            fontweight="bold")

        # ── Panel 2: Hawkes Intensity ─────────────────────────────────────
        ax_inten.fill_between(t, 0, inten, color=THEME["INTENSITY"],
                              alpha=0.55, zorder=2)
        ax_inten.plot(t, inten, color=THEME["INTENSITY"], linewidth=1.4,
                      alpha=0.95, zorder=3)
        ax_inten.axhline(CONFIG["MU"], color=THEME["WHITE"], linewidth=1.0,
                         linestyle="--", alpha=0.6, zorder=2)

        ax_inten.set_ylim(0, inten_full.max() * 1.15)
        ax_inten.set_ylabel(r"Intensity  $\lambda(t)$", color=THEME["TEXT"],
                            fontsize=14, fontweight="bold")

        # Baseline label
        ax_inten.text(t_full[-1] * 0.98, CONFIG["MU"] * 1.12,
                      r"$\mu$ baseline",
                      color=THEME["WHITE"], fontsize=11, ha="right", alpha=0.7)

        # ── Panel 3: Microstructure Metrics ───────────────────────────────
        # Normalize spread to [0,1] range for co-plotting
        spread_norm = (spread - sim["spread"].min()) / max(
            sim["spread"].max() - sim["spread"].min(), 1e-6
        )

        ax_micro.plot(t, spread_norm, color=THEME["SPREAD"], linewidth=1.6,
                      alpha=0.95, label="Spread")
        ax_micro.plot(t, (imbal + 1) / 2, color=THEME["IMBALANCE"],
                      linewidth=1.6, alpha=0.95, label="Order Imbalance")
        ax_micro.plot(t, trate, color=THEME["TRADE_RATE"], linewidth=1.6,
                      alpha=0.95, label="Trade Rate")

        ax_micro.set_ylim(-0.1, 1.35)
        ax_micro.set_ylabel("Metrics  (normalized)", color=THEME["TEXT"],
                            fontsize=14, fontweight="bold")
        ax_micro.set_xlabel("Time", color=THEME["TEXT"], fontsize=14)

        # ── Annotations ──────────────────────────────────────────────────
        for evt_idx, label in sim["events"]:
            if evt_idx < data_end_idx:
                evt_t = t_full[evt_idx]
                evt_p = price_full[evt_idx]

                # Annotation on price panel
                ax_price.annotate(
                    label,
                    xy=(evt_t, evt_p),
                    xytext=(evt_t, evt_p + price_margin * 0.7),
                    fontsize=11, color=THEME["WHITE"], fontweight="bold",
                    ha="center",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=THEME["WHITE"],
                        lw=1.3,
                        connectionstyle="arc3,rad=0.15"
                    ),
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=THEME["BG"],
                              edgecolor=THEME["SPINE"], alpha=0.9),
                    zorder=10,
                )

        # ── Legend (bottom of figure) ─────────────────────────────────────
        # Collect handles from all panels
        h1, l1 = ax_price.get_legend_handles_labels()
        h3, l3 = ax_micro.get_legend_handles_labels()
        all_h = h1 + h3
        all_l = l1 + l3

        if all_h:
            leg = fig.legend(
                all_h, all_l,
                loc="lower center", ncol=len(all_l),
                fontsize=12, frameon=True, fancybox=False,
                borderpad=0.5, handlelength=2.5, columnspacing=2.0,
                bbox_to_anchor=(0.52, 0.002),
                edgecolor=THEME["SPINE"],
                facecolor=THEME["BG"],
            )
            for txt in leg.get_texts():
                txt.set_color(THEME["WHITE"])

        # ── Title / HUD ──────────────────────────────────────────────────
        fig.suptitle(
            "Hawkes Process  //  Market Microstructure Simulation",
            fontsize=20, fontweight="bold", color=THEME["WHITE"],
            y=0.965,
        )
        fig.text(
            0.95, 0.965,
            f"t = {t[-1]:.0f}",
            fontsize=13, color=THEME["TEXT"], ha="right",
            fontfamily="monospace",
        )

        # Formula annotation (top-left, subtle)
        fig.text(
            0.07, 0.940,
            r"$\lambda(t) = \mu + \sum_{t_i < t} \alpha \, e^{-\beta(t - t_i)}$",
            fontsize=13, color="#999999", fontfamily="serif",
        )

        # ── Save ──────────────────────────────────────────────────────────
        out_path = CONFIG["OUTPUT_FILE"]
        fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)
        log(f"[Success] Image saved to: {out_path}")
        return True

    except Exception as e:
        print(f"[Error] Render: {e}")
        plt.close("all")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Clean log
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    log("=== HAWKES PROCESS PIPELINE START ===")
    log(f"    mu={CONFIG['MU']}, alpha={CONFIG['ALPHA']}, beta={CONFIG['BETA']}")
    log(f"    Branching ratio alpha/beta = {CONFIG['ALPHA']/CONFIG['BETA']:.3f}")

    # 1. Simulate
    sim = simulate_hawkes()

    # 2. Render static image
    render_static(sim)

    log("=== PIPELINE END ===")


if __name__ == "__main__":
    main()
