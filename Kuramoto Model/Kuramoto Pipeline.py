"""
Kuramoto_Pipeline.py
=====================
Project: Quant Trader Lab - Complex Systems
Author: quant.traderr (Instagram)
License: MIT

Description:
    Static snapshot of the Kuramoto model: N coupled phase oscillators
    whose natural frequencies are drawn from a Gaussian. As the coupling
    K ramps from 0 past the critical threshold K_c, phases lock. Quant
    reframe: each oscillator is one asset's return phase; when systemic
    coupling crosses K_c, everything moves together — diversification
    dies.

        dtheta_i/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i)
        order parameter:  r * exp(i*psi) = (1/N) sum_j exp(i*theta_j)

    Renders the full sim as a time-cylinder (z = time, polar = phase),
    each oscillator a hue-mapped thread fanning from incoherent bottom
    into a synchronized braid at the top. Gold arrow at the top is the
    final mean-field order parameter.

    Pipeline:  SIMULATE -> VISUALIZE (static PNG)
    Resolution: 1920x1080

Dependencies:
    pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "OUTPUT_IMAGE":  "Kuramoto_Output.png",
    "RESOLUTION":    (1920, 1080),
    "DPI":           100,
    "LOG_FILE":      os.path.join(BASE_DIR, "kuramoto_pipeline.log"),
    "SEED":          42,
}

KURAMOTO = {
    "N":            100,     # number of oscillators
    "OMEGA_STD":    0.8,     # std of omega_i ~ N(0, OMEGA_STD^2)
    "K_MAX":        3.5,     # peak coupling (K_c ~ 1.28 for this sigma)
    "DT":           0.05,    # integration step
    "N_STEPS":      390,     # total sim steps (matches video)
    "R_CYL":        1.0,     # cylinder radius in plot units
    "Z_SCALE":      4.2,     # z-axis stretch for time (visual)
}

THEME = {
    "BG": "#000000", "PANEL_BG": "#0a0a0a", "GRID": "#1a1a1a",
    "TEXT": "#ffffff", "TEXT_DIM": "#aaaaaa",
    "ORANGE": "#ff9500", "ORANGE_HOT": "#ff6b00",
    "YELLOW": "#ffd400", "CYAN": "#00e5ff",
    "MAGENTA": "#ff1493", "FONT": "Arial",
}

CMAP = LinearSegmentedColormap.from_list("kura", [
    "#00e5ff",      # cool: low omega (slowest negative drifters)
    "#3d66ff",
    "#a050ff",      # mid: zero omega
    "#ff3d8f",
    "#ff9500",      # warm: high omega (fastest positive drifters)
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
# MODULE 1: KURAMOTO SIMULATION (mean-field form, linear K ramp)
# ═══════════════════════════════════════════════════════════════════
def simulate_kuramoto():
    log("[SIM] Starting simulation...")
    rng = np.random.default_rng(CONFIG["SEED"])
    N = KURAMOTO["N"]
    dt = KURAMOTO["DT"]
    n_steps = KURAMOTO["N_STEPS"]

    omega = rng.normal(0.0, KURAMOTO["OMEGA_STD"], size=N)
    theta = rng.uniform(-np.pi, np.pi, size=N)

    theta_hist = np.zeros((n_steps + 1, N))
    r_hist = np.zeros(n_steps + 1)
    psi_hist = np.zeros(n_steps + 1)
    K_hist = np.zeros(n_steps + 1)

    theta_hist[0] = theta
    z = np.mean(np.exp(1j * theta))
    r_hist[0] = np.abs(z); psi_hist[0] = np.angle(z); K_hist[0] = 0.0

    K_max = KURAMOTO["K_MAX"]
    for t in range(1, n_steps + 1):
        K = K_max * (t / n_steps)
        z = np.mean(np.exp(1j * theta))
        r, psi = np.abs(z), np.angle(z)
        dtheta = omega + K * r * np.sin(psi - theta)
        theta = theta + dt * dtheta
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

        theta_hist[t] = theta
        r_hist[t] = r; psi_hist[t] = psi; K_hist[t] = K

    # Kuramoto critical coupling for Gaussian g(0) = 1/(sigma*sqrt(2pi))
    K_c = 2.0 / (np.pi * (1.0 / (KURAMOTO["OMEGA_STD"] * np.sqrt(2 * np.pi))))
    log(f"[SIM] Done. N={N}  K_max={K_max:.2f}  K_c~{K_c:.2f}  r_final={r_hist[-1]:.3f}")
    return omega, theta_hist, r_hist, psi_hist, K_hist, K_c


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: VISUALIZATION (full time-cylinder, single static frame)
# ═══════════════════════════════════════════════════════════════════
def visualize(omega, theta_hist, r_hist, psi_hist, K_hist, K_c):
    log("[Visual] Generating static snapshot...")

    N = omega.shape[0]
    n_steps_plus = theta_hist.shape[0]  # n_steps + 1
    R = KURAMOTO["R_CYL"]
    z_axis = np.linspace(0.0, KURAMOTO["Z_SCALE"], n_steps_plus)

    fig = plt.figure(
        figsize=(CONFIG["RESOLUTION"][0] / CONFIG["DPI"],
                 CONFIG["RESOLUTION"][1] / CONFIG["DPI"]),
        dpi=CONFIG["DPI"], facecolor=THEME["BG"],
    )

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80], projection="3d",
                      computed_zorder=False)
    ax.set_facecolor(THEME["BG"])

    pane = (0.02, 0.02, 0.02, 1)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["grid"]["color"] = (0.10, 0.10, 0.10, 0.5)
        a._axinfo["grid"]["linewidth"] = 0.3

    # --- Build thread segments, skipping phase-wrap chords ---
    x_all = R * np.cos(theta_hist)
    y_all = R * np.sin(theta_hist)
    z_all = np.broadcast_to(z_axis[:, None], theta_hist.shape)

    dtheta_abs = np.abs(theta_hist[1:] - theta_hist[:-1])
    wrap_mask = dtheta_abs > np.pi

    omega_min, omega_max = omega.min(), omega.max()
    omega_range = max(omega_max - omega_min, 1e-6)

    segs = []
    seg_colors = []
    for i in range(N):
        c = CMAP((omega[i] - omega_min) / omega_range)
        xi, yi, zi = x_all[:, i], y_all[:, i], z_all[:, i]
        for t0 in range(n_steps_plus - 1):
            if wrap_mask[t0, i]:
                continue
            segs.append([(xi[t0], yi[t0], zi[t0]),
                         (xi[t0 + 1], yi[t0 + 1], zi[t0 + 1])])
            seg_colors.append(c)

    lc = Line3DCollection(segs, colors=seg_colors, linewidths=0.9, alpha=0.70)
    ax.add_collection3d(lc)

    # --- Guide rings at bottom (incoherent) and top (synced) ---
    ring_theta = np.linspace(-np.pi, np.pi, 180)
    ax.plot(R * np.cos(ring_theta), R * np.sin(ring_theta),
            np.full_like(ring_theta, z_axis[0]),
            color=THEME["GRID"], lw=0.7, alpha=0.7, zorder=4)
    ax.plot(R * np.cos(ring_theta), R * np.sin(ring_theta),
            np.full_like(ring_theta, z_axis[-1]),
            color=THEME["GRID"], lw=0.7, alpha=0.7, zorder=4)

    # --- Phase dots at top (final synchronized snapshot) ---
    theta_top = theta_hist[-1]
    x_top = R * np.cos(theta_top); y_top = R * np.sin(theta_top)
    dot_colors = CMAP((omega - omega_min) / omega_range)
    ax.scatter(x_top, y_top, np.full(N, z_axis[-1]),
               c=dot_colors, s=22, depthshade=False,
               edgecolors="none", zorder=20)

    # --- Also phase dots at bottom (initial incoherent snapshot) ---
    theta_bot = theta_hist[0]
    x_bot = R * np.cos(theta_bot); y_bot = R * np.sin(theta_bot)
    ax.scatter(x_bot, y_bot, np.full(N, z_axis[0]),
               c=dot_colors, s=14, depthshade=False,
               edgecolors="none", alpha=0.65, zorder=18)

    # --- Order-parameter arrow at top (gold) ---
    r_f, psi_f = r_hist[-1], psi_hist[-1]
    ax.plot([0, R * r_f * np.cos(psi_f)], [0, R * r_f * np.sin(psi_f)],
            [z_axis[-1], z_axis[-1]],
            color=THEME["YELLOW"], lw=3.0, zorder=25)
    ax.scatter([R * r_f * np.cos(psi_f)], [R * r_f * np.sin(psi_f)],
               [z_axis[-1]],
               color=THEME["YELLOW"], s=60, zorder=26, edgecolors="none")

    # --- Order-parameter arrow at bottom (faint, for contrast) ---
    r_0, psi_0 = r_hist[0], psi_hist[0]
    ax.plot([0, R * r_0 * np.cos(psi_0)], [0, R * r_0 * np.sin(psi_0)],
            [z_axis[0], z_axis[0]],
            color=THEME["YELLOW"], lw=1.6, alpha=0.4, zorder=15)

    # --- K_c threshold marker on cylinder (horizontal ring where K = K_c) ---
    K_arr = K_hist
    if K_arr[-1] > K_c:
        t_cross = int(np.argmax(K_arr >= K_c))
        z_cross = z_axis[t_cross]
        ax.plot(R * np.cos(ring_theta) * 1.04,
                R * np.sin(ring_theta) * 1.04,
                np.full_like(ring_theta, z_cross),
                color=THEME["ORANGE_HOT"], lw=1.2, alpha=0.7,
                linestyle="--", zorder=6)

    # --- Axes / view ---
    ax.set_xlim(-R * 1.15, R * 1.15)
    ax.set_ylim(-R * 1.15, R * 1.15)
    ax.set_zlim(z_axis[0], z_axis[-1])
    ax.set_xlabel(r"$\cos\theta$", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=10, fontfamily=THEME["FONT"])
    ax.set_ylabel(r"$\sin\theta$", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=10, fontfamily=THEME["FONT"])
    ax.set_zlabel("TIME →  K(t) ↑", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=12, fontfamily=THEME["FONT"])
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=8)
    ax.view_init(elev=12, azim=-62)

    # --- Title ---
    fig.text(0.50, 0.955, "KURAMOTO MODEL",
             ha="center", fontsize=30, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=THEME["FONT"])
    fig.text(0.50, 0.918,
             r"$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_j \sin(\theta_j - \theta_i)$",
             ha="center", fontsize=15, color=THEME["TEXT"],
             fontfamily=THEME["FONT"])
    fig.text(0.50, 0.886,
             f"PHASE SYNCHRONIZATION OF {KURAMOTO['N']} OSCILLATORS    "
             r"$\omega \sim \mathcal{N}(0, " f"{KURAMOTO['OMEGA_STD']})" r"$    "
             f"K_c ≈ {K_c:.2f}",
             ha="center", fontsize=11, fontweight="bold",
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    # --- Legend block (left) ---
    fig.text(0.03, 0.895,
             "GOLD ARROW = ORDER PARAMETER  r·e^(iψ)\n"
             "ORANGE DASH = K_c THRESHOLD\n"
             "HUE = NATURAL FREQUENCY  ω_i",
             ha="left", fontsize=9, fontweight="bold",
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    # --- Final-state readout (right) ---
    r_final = r_hist[-1]; K_final = K_hist[-1]
    fig.text(0.97, 0.895,
             f"K_final = {K_final:.2f}    K/K_c = {K_final / K_c:.2f}\n"
             f"r_final = {r_final:.3f}    STATE: SYNCHRONIZED",
             ha="right", fontsize=10, fontweight="bold",
             color=THEME["YELLOW"], fontfamily=THEME["FONT"])

    # --- Watermark ---
    fig.text(0.98, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

    out = os.path.join(BASE_DIR, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"[Visual] Saved to {out}")


def main():
    t0 = time.time()
    log("=== KURAMOTO PIPELINE ===")
    omega, theta_hist, r_hist, psi_hist, K_hist, K_c = simulate_kuramoto()
    visualize(omega, theta_hist, r_hist, psi_hist, K_hist, K_c)
    log(f"=== PIPELINE FINISHED ({time.time() - t0:.1f}s) ===")


if __name__ == "__main__":
    main()
