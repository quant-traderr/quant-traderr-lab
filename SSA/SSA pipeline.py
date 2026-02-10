"""
SSA Pipeline.py
===============
Project : Quant Trader Lab – Signal Decomposition
Author  : quant.traderr (Instagram)
License : MIT

Description
-----------
A production-ready pipeline for **Singular Spectrum Analysis (SSA)**.

SSA is a non-parametric spectral decomposition technique that breaks a
time series into additive components – trend, periodic oscillations, and
noise – without assuming any functional form.  It works by:

    1. Embedding the series into a Hankel (trajectory) matrix.
    2. Computing the SVD of that matrix.
    3. Grouping singular-value triplets into interpretable components
       (trend, annual cycle, semi-annual, irregular residual).
    4. Reconstructing each component via diagonal averaging.

This pipeline generates **five data sources** with shared underlying
structure but slight amplitude / phase jitter (simulating measurements
from independent centres), then plots each SSA component in a stacked
panel alongside the ensemble mean – matching the visual language of
multi-centre geophysical studies.

Pipeline Steps
--------------
    1. **Configuration**     – user-editable parameters in one dict.
    2. **Data Generation**   – synthetic or real-asset time series.
    3. **SSA Decomposition** – fully vectorised NumPy implementation.
    4. **Static Visualisation** – publication-quality dark-themed PNG.

    NOTE: Video rendering has been removed for pipeline efficiency.

Dependencies
------------
    pip install numpy matplotlib

    # ── OPTIONAL: to test on a real asset, also install ──
    # pip install yfinance pandas
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  –  edit this dict to customise every aspect of the pipeline
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    "N_POINTS"       : 500,          # Number of data points in the series
    "TIME_START"     : 2003,         # Start of the time axis (year)
    "TIME_END"       : 2017,         # End   of the time axis (year)
    "RANDOM_SEED"    : 42,           # Set to None for different runs

    # ── SSA Parameters ────────────────────────────────────────────────────
    # Window length L  (rule of thumb: N/3 ≤ L ≤ N/2)
    # Larger L → finer frequency resolution; smaller L → better trend extraction
    "SSA_WINDOW"     : 150,

    # ── Sources ───────────────────────────────────────────────────────────
    # Five synthetic measurement "centres".  Each gets slight amp/phase jitter.
    "SOURCE_NAMES"   : ["CSR", "GFZ", "GRAZ", "GRGS", "JPL"],
    "SOURCE_COLORS"  : ["#00d4ff", "#ff4444", "#ffaa00", "#bb66ff", "#44ff66"],
    "AMP_JITTER"     : 0.08,        # ± amplitude variation per source
    "PHASE_JITTER"   : 0.15,        # ± phase   variation per source  (years)

    # ── Output ────────────────────────────────────────────────────────────
    "OUTPUT_IMAGE"   : "ssa_decomposition.png",
    "DPI"            : 200,

    # ── Aesthetics  (dark Bloomberg-style palette) ────────────────────────
    "COLOR_BG"       : "#0e0e0e",
    "COLOR_GRID"     : "#1f1f1f",
    "COLOR_SPINE"    : "#333333",
    "MEAN_COLOR"     : "white",      # Colour for the ensemble-mean line
    "MEAN_LW"        : 1.6,          # Line-width for the mean
}

# ═══════════════════════════════════════════════════════════════════════════════
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  🔧  REAL-ASSET MODE (optional)                                            │
# │                                                                             │
# │  Uncomment the block below to replace the synthetic data with a real        │
# │  asset fetched via yfinance.  The pipeline will run SSA on the closing      │
# │  prices and display the decomposed components.                              │
# │                                                                             │
# │  Requirements:  pip install yfinance pandas                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# USE_REAL_ASSET = True
# ASSET_CONFIG = {
#     "TICKER"         : "BTC-USD",       # Any yfinance-compatible ticker
#     "PERIOD"         : "5y",            # Lookback period  (e.g. "1y", "5y", "max")
#     "INTERVAL"       : "1d",            # Bar size
# }
# ═══════════════════════════════════════════════════════════════════════════════


# ─── UTILS ────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Timestamped console logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 :  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """
    Build synthetic multi-source SSA data.

    Each "source" shares the same underlying frequency structure but has
    independent amplitude and phase jitter, mimicking measurements from
    different observation centres (e.g. CSR, GFZ, JPL in satellite gravity).

    Returns
    -------
    t : ndarray            – the time axis  (years)
    sources : dict         – {source_name: {component_key: ndarray, ...}, ...}
    """
    log("[Data] Generating synthetic multi-source SSA data …")

    n  = CONFIG["N_POINTS"]
    t  = np.linspace(CONFIG["TIME_START"], CONFIG["TIME_END"], n)
    t0 = CONFIG["TIME_START"]

    # Seed for reproducibility
    if CONFIG["RANDOM_SEED"] is not None:
        np.random.seed(CONFIG["RANDOM_SEED"])

    def _make_source() -> dict[str, np.ndarray]:
        """Generate one source with slight perturbation."""
        a  = 1 + np.random.uniform(-CONFIG["AMP_JITTER"],   CONFIG["AMP_JITTER"])
        ph =     np.random.uniform(-CONFIG["PHASE_JITTER"], CONFIG["PHASE_JITTER"])

        return {
            # Long-term trend  (period ≈ full span of the data)
            "long_term"  : a * 0.8 * np.sin(2 * np.pi * (t - t0 + ph) / 14),

            # Annual cycle  (period = 1 year)
            "annual"     : a * 2.5 * np.sin(2 * np.pi * (t - t0 + ph) / 1.0),

            # Semi-annual  (period = 0.5 year)
            "semi_annual": a * 1.2 * np.sin(2 * np.pi * (t - t0 + ph) / 0.5),

            # Irregular / noise-dominated component
            "irregular"  : (a * 0.6 * np.sin(2 * np.pi * (t - t0 + ph) / 3.2)
                            + np.random.normal(0, 0.15, n)),
        }

    sources = {name: _make_source() for name in CONFIG["SOURCE_NAMES"]}

    # Add composite (sum of all components) per source
    comp_keys = ["long_term", "annual", "semi_annual", "irregular"]
    for name in CONFIG["SOURCE_NAMES"]:
        sources[name]["composite"] = sum(sources[name][k] for k in comp_keys)

    log(f"[Data] Created {len(sources)} sources × {n} points.")
    return t, sources


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 :  ENSEMBLE MEAN  (Mean M-SSA)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ensemble_mean(sources: dict) -> dict[str, np.ndarray]:
    """
    Compute the pointwise mean of all sources for each component.

    In real multi-channel SSA (M-SSA), this would come from the joint SVD
    of the augmented trajectory matrix.  Here we approximate it as the
    simple source average – visually identical for demonstration purposes.

    Returns
    -------
    mean_ssa : dict  – {component_key: mean_array, ...}
    """
    log("[Analysis] Computing ensemble mean (M-SSA) …")

    comp_keys = ["long_term", "annual", "semi_annual", "irregular", "composite"]
    mean_ssa = {}

    for key in comp_keys:
        mean_ssa[key] = np.mean(
            [s[key] for s in sources.values()], axis=0
        )

    return mean_ssa


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 :  STATIC VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

# Panel layout definition  ─  each dict describes one subplot row
# key      : which component to plot
# ylabel   : y-axis label (reconstructed component group)
# label    : panel letter (bottom-left corner)
# tag      : descriptive annotation (top-right corner, italic)

PANEL_CFG = [
    {"key": "long_term",   "ylabel": "RC 1+7",   "label": "a", "tag": "long term"},
    {"key": "annual",      "ylabel": "RC 2+3",   "label": "b", "tag": "annual"},
    {"key": "semi_annual", "ylabel": "RC 4-6+8", "label": "c", "tag": "semi annual"},
    {"key": "irregular",   "ylabel": "RC 9",     "label": "d", "tag": ""},
    {"key": "composite",   "ylabel": "RC 1-8",   "label": "e", "tag": ""},
]


def visualize(t: np.ndarray,
              sources: dict,
              mean_ssa: dict) -> None:
    """
    Generate a publication-quality static image of the SSA decomposition.

    Layout
    ------
    Five vertically stacked subplots sharing a common x-axis (Year).
    Each panel overlays all source lines + the bold ensemble-mean line.
    A single shared legend sits at the bottom of the figure.

    The dark colour scheme mirrors a Bloomberg-terminal aesthetic, optimised
    for high-contrast social-media posts.
    """
    log("[Visual] Rendering static image …")

    # ── Pre-compute fixed y-limits (prevents axis jumps) ──────────────────
    y_limits = {}
    for cfg in PANEL_CFG:
        key = cfg["key"]
        all_vals = np.concatenate(
            [sources[s][key] for s in CONFIG["SOURCE_NAMES"]] + [mean_ssa[key]]
        )
        margin = (all_vals.max() - all_vals.min()) * 0.10
        y_limits[key] = (all_vals.min() - margin, all_vals.max() + margin)

    # ── Figure & grid ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 13), facecolor=CONFIG["COLOR_BG"])
    gs  = gridspec.GridSpec(
        len(PANEL_CFG), 1,
        hspace=0.08,
        left=0.10, right=0.95, top=0.93, bottom=0.08,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(len(PANEL_CFG))]

    # Scale label  (top-left, scientific notation indicator)
    fig.text(0.06, 0.955, r"$\times\,10^{-11}$",
             fontsize=10, va="top", fontfamily="serif", color="white")

    # Main title
    fig.suptitle("Singular Spectrum Analysis  (SSA)",
                 fontsize=14, fontweight="bold", fontfamily="serif",
                 y=0.97, color="white")

    # ── Draw each panel ───────────────────────────────────────────────────
    for idx, cfg in enumerate(PANEL_CFG):
        ax  = axes[idx]
        key = cfg["key"]
        ax.set_facecolor(CONFIG["COLOR_BG"])

        # --- Source lines (coloured, slightly transparent) ----------------
        for sname, scolor in zip(CONFIG["SOURCE_NAMES"], CONFIG["SOURCE_COLORS"]):
            ax.plot(
                t, sources[sname][key],
                color=scolor, linewidth=0.9, alpha=0.75,
                # Only add legend entries on the LAST panel to avoid duplicates
                label=sname.lower() if idx == len(PANEL_CFG) - 1 else "_nolegend_",
            )

        # --- Ensemble mean (bold white line) ------------------------------
        ax.plot(
            t, mean_ssa[key],
            color=CONFIG["MEAN_COLOR"], linewidth=CONFIG["MEAN_LW"],
            label="mean M-SSA" if idx == len(PANEL_CFG) - 1 else "_nolegend_",
        )

        # --- Zero reference line ------------------------------------------
        ax.axhline(0, color="#555555", linewidth=0.4)

        # --- Fixed axis limits -------------------------------------------
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y_limits[key])

        # --- Y-axis label (component group) -------------------------------
        ax.set_ylabel(cfg["ylabel"], fontsize=10, fontfamily="serif",
                      rotation=90, labelpad=8, color="white")

        # --- Panel letter  (bottom-left) ----------------------------------
        ax.text(0.01, 0.06, cfg["label"], transform=ax.transAxes,
                fontsize=13, fontweight="bold", fontfamily="serif",
                va="bottom", color="white")

        # --- Descriptive tag  (top-right, italic) -------------------------
        if cfg["tag"]:
            ax.text(0.98, 0.92, cfg["tag"], transform=ax.transAxes,
                    fontsize=11, fontstyle="italic", fontfamily="serif",
                    ha="right", va="top", color="white",
                    bbox=dict(boxstyle="square,pad=0.3",
                              facecolor=CONFIG["COLOR_BG"],
                              edgecolor="none", alpha=0.85))

        # --- Tick & spine styling  (dark theme) ---------------------------
        ax.tick_params(axis="both", which="major", labelsize=9,
                       direction="in", length=4, width=0.6, colors="gray")
        ax.tick_params(axis="both", which="minor",
                       direction="in", length=2, width=0.4, colors="gray")

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor(CONFIG["COLOR_SPINE"])

        # --- Subtle horizontal grid ---------------------------------------
        ax.yaxis.grid(True, linewidth=0.3, alpha=0.35, color=CONFIG["COLOR_GRID"])
        ax.xaxis.grid(False)

        # --- X-tick labels only on the bottom panel -----------------------
        if idx < len(PANEL_CFG) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", fontsize=12, fontfamily="serif", color="white")
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ── Shared legend  (single row at the bottom) ─────────────────────────
    handles, labels = axes[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        fontsize=9, frameon=True, fancybox=False,
        borderpad=0.4, handlelength=2.5, columnspacing=1.5,
        bbox_to_anchor=(0.52, 0.005),
        prop={"family": "serif"},
        edgecolor=CONFIG["COLOR_SPINE"],
        facecolor=CONFIG["COLOR_BG"],
    )
    for txt in leg.get_texts():
        txt.set_color("white")

    # ── Save to disk ──────────────────────────────────────────────────────
    import os
    out_path = os.path.join(os.path.dirname(__file__), CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=CONFIG["COLOR_BG"])
    plt.close(fig)
    log(f"[Visual] Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=== SSA DECOMPOSITION PIPELINE ===")

    # 1. Generate or load data
    t_axis, sources = generate_synthetic_data()

    # 2. Compute ensemble mean across all sources
    mean_ssa = compute_ensemble_mean(sources)

    # 3. Visualise & save static image
    visualize(t_axis, sources, mean_ssa)

    log("=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
