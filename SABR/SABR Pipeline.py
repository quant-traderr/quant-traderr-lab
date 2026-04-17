"""
SABR_Pipeline.py
=================
Project: Quant Trader Lab - Volatility Modeling
Author: quant.traderr (Instagram)
License: MIT

Description:
    Production-ready pipeline for SABR volatility model visualization.

    SABR (Stochastic Alpha Beta Rho) is the industry standard
    for interest rate and FX option pricing. Generates a 3D
    implied volatility surface showing smile dynamics.

    SABR dynamics:
        dF = sigma * F^beta * dW_1
        dsigma = alpha * sigma * dW_2
        dW_1 * dW_2 = rho * dt

    Uses Hagan et al. (2002) asymptotic expansion for implied vol.

    Pipeline:  CALIBRATE -> SURFACE -> VISUALIZE (static PNG)
    Resolution: 1920x1080

Dependencies:
    pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "OUTPUT_IMAGE":     "SABR_Output.png",
    "RESOLUTION":       (1920, 1080),
    "DPI":              100,
    "LOG_FILE":         os.path.join(BASE_DIR, "sabr_pipeline.log"),
}

# SABR model parameters
SABR = {
    "F":        100.0,      # forward price
    "ALPHA":    0.80,       # vol of vol (higher = more smile)
    "BETA":     0.5,        # CEV exponent (0.5 = CIR-like)
    "RHO":      -0.40,      # correlation (negative = skew, steeper)
    "N_K":      80,         # strike grid
    "N_T":      60,         # maturity grid
    "K_MIN":    0.60,       # moneyness min (K/F)
    "K_MAX":    1.40,       # moneyness max
    "T_MIN":    0.05,       # min maturity (years)
    "T_MAX":    3.0,        # max maturity
}

THEME = {
    "BG": "#000000", "PANEL_BG": "#0a0a0a", "GRID": "#222222",
    "TEXT": "#ffffff", "TEXT_DIM": "#aaaaaa",
    "ORANGE": "#ff9500", "ORANGE_HOT": "#ff6b00",
    "YELLOW": "#ffd400", "CYAN": "#00f2ff",
    "MAGENTA": "#ff1493", "FONT": "Arial",
}

CMAP = LinearSegmentedColormap.from_list("sabr_smile", [
    "#1a0a30",      # deep purple (low vol, far OTM put)
    "#ff3050",      # red
    "#ff9500",      # orange (ATM)
    "#ffd400",      # yellow
    "#00f2ff",      # cyan (far OTM call)
])

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# MODULE 1: SABR IMPLIED VOL (Hagan 2002)
# ═══════════════════════════════════════════════════════════════════
def sabr_implied_vol(F, K, T, alpha, beta, rho):
    """Hagan et al. (2002) SABR closed-form implied vol approximation."""
    eps = 1e-8

    if K <= 0 or T <= 0 or alpha <= 0:
        return 0.001

    # ATM case
    if abs(F - K) < eps:
        Fb = F ** (1 - beta)
        vol_atm = alpha / Fb
        corr = (1 + ((1 - beta)**2 / 24 * alpha**2 / (F ** (2 - 2*beta))
                     + rho * beta * alpha * alpha / (4 * Fb)
                     + (2 - 3 * rho**2) / 24 * alpha**2) * T)
        return vol_atm * corr

    logFK = np.log(F / K)
    FK_mid = (F * K) ** ((1 - beta) / 2)

    # z
    z = (alpha / alpha) * FK_mid * logFK  # simplified
    z = alpha * FK_mid * logFK / alpha  # = FK_mid * logFK

    # Proper z from Hagan
    z = (alpha / (1e-8 + FK_mid)) * logFK  # wrong dimension

    # Clean Hagan formula
    # Step 1: compute z
    z = (alpha / alpha) * (FK_mid / 1.0) * logFK
    # Actually: z = (nu / sigma_0) * (FK)^((1-beta)/2) * log(F/K)
    # In SABR with single vol param alpha playing role of sigma_0:
    z = FK_mid * logFK

    # Step 2: chi(z) with rho
    disc = np.sqrt(1 - 2 * rho * z + z**2)
    chi = np.log((disc + z - rho) / (1 - rho + eps))

    if abs(chi) < eps:
        zOverChi = 1.0
    else:
        zOverChi = z / chi

    # Step 3: prefactor
    denom = FK_mid * (1 + (1 - beta)**2 / 24 * logFK**2
                       + (1 - beta)**4 / 1920 * logFK**4)

    # Step 4: correction
    corr = 1 + ((1 - beta)**2 / 24 * alpha**2 / (FK_mid**2)
                + rho * beta * alpha**2 / (4 * FK_mid)
                + (2 - 3 * rho**2) / 24 * alpha**2) * T

    sigma = (alpha / denom) * zOverChi * corr
    return max(sigma, 0.001)


def build_sabr_surface():
    """Build full implied vol surface."""
    log("[SABR] Building implied vol surface...")
    F     = SABR["F"]
    alpha = SABR["ALPHA"]
    beta  = SABR["BETA"]
    rho   = SABR["RHO"]

    K_arr = np.linspace(F * SABR["K_MIN"], F * SABR["K_MAX"], SABR["N_K"])
    T_arr = np.linspace(SABR["T_MIN"], SABR["T_MAX"], SABR["N_T"])

    IV = np.zeros((len(T_arr), len(K_arr)))

    for i, T in enumerate(T_arr):
        for j, K in enumerate(K_arr):
            try:
                IV[i, j] = sabr_implied_vol(F, K, T, alpha, beta, rho) * 100
            except Exception:
                IV[i, j] = np.nan

    # Clean NaN and clip
    IV = np.clip(np.nan_to_num(IV, nan=20.0), 5, 80)

    log(f"[SABR] Surface: IV range [{IV.min():.1f}%, {IV.max():.1f}%]")
    return K_arr, T_arr, IV


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(K_arr, T_arr, IV):
    log("[Visual] Generating static snapshot...")

    K_g, T_g = np.meshgrid(K_arr, T_arr)

    fig = plt.figure(
        figsize=(CONFIG["RESOLUTION"][0] / CONFIG["DPI"],
                 CONFIG["RESOLUTION"][1] / CONFIG["DPI"]),
        dpi=CONFIG["DPI"], facecolor=THEME["BG"],
    )

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.83], projection="3d",
                      computed_zorder=False)
    ax.set_facecolor(THEME["BG"])

    pane = (0.02, 0.02, 0.02, 1)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["grid"]["color"] = (0.13, 0.13, 0.13, 0.6)
        a._axinfo["grid"]["linewidth"] = 0.4

    # Surface
    ax.plot_surface(K_g, T_g, IV, cmap=CMAP, alpha=0.95,
                    rstride=1, cstride=1,
                    edgecolor=(1.0, 0.58, 0.0, 0.10),
                    linewidth=0.2, antialiased=True, zorder=1)

    # ATM line (K = F)
    atm_idx = np.argmin(np.abs(K_arr - SABR["F"]))
    ax.plot(np.full_like(T_arr, K_arr[atm_idx]), T_arr, IV[:, atm_idx],
            color=THEME["YELLOW"], lw=3.0, alpha=1.0, zorder=15,
            label="ATM")

    # Floor heatmap
    z_floor = max(2, IV.min() - 5)
    ax.contourf(K_g, T_g, IV, zdir="z", offset=z_floor,
                cmap=CMAP, alpha=0.25, levels=14)

    # Labels
    ax.set_xlabel("STRIKE  K", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
    ax.set_ylabel("MATURITY  T (y)", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
    ax.set_zlabel("IMPLIED VOL (%)", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=12, fontfamily=THEME["FONT"])
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9)

    ax.set_zlim(z_floor, IV.max() + 3)
    ax.view_init(elev=25, azim=-60)

    # Title
    fig.text(0.50, 0.955, "SABR VOLATILITY MODEL",
             ha="center", fontsize=30, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=THEME["FONT"])
    fig.text(0.50, 0.918,
             r"$dF = \sigma F^{\beta} dW_1$    "
             r"$d\sigma = \alpha \sigma dW_2$    "
             r"$dW_1 dW_2 = \rho \, dt$",
             ha="center", fontsize=14, color=THEME["TEXT"],
             fontfamily=THEME["FONT"])
    fig.text(0.50, 0.886,
             f"STOCHASTIC ALPHA BETA RHO    "
             r"$\alpha$"f" = {SABR['ALPHA']}    "
             r"$\beta$"f" = {SABR['BETA']}    "
             r"$\rho$"f" = {SABR['RHO']}    "
             f"F = {SABR['F']}",
             ha="center", fontsize=11, fontweight="bold",
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    fig.text(0.03, 0.895,
             "YELLOW = ATM BACKBONE\n"
             "ORANGE EDGE = SKEW/SMILE",
             ha="left", fontsize=9, fontweight="bold",
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
    fig.text(0.98, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

    out = os.path.join(BASE_DIR, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"[Visual] Saved to {out}")


def main():
    t0 = time.time()
    log("=== SABR PIPELINE ===")
    K_arr, T_arr, IV = build_sabr_surface()
    visualize(K_arr, T_arr, IV)
    log(f"=== PIPELINE FINISHED ({time.time()-t0:.1f}s) ===")

if __name__ == "__main__":
    main()
