"""
StochResonance_Reel_Pipeline.py
================================
Cinematic landscape reel: Stochastic Resonance in a bistable system.
A weak periodic signal (sub-threshold — alone it can't flip wells) becomes
COHERENTLY detectable when the right amount of noise is added.
Too little noise: ball stuck in one well. Too much: random chaos.
Just right: the ball flips between wells locked to the signal's beat.

System: dx/dt = x − x³ + A·sin(2π f t) + σ·η(t)

The reel sweeps σ from low → optimal → high. Trajectory is rendered as
a 3D ribbon (time, σ, x). Color = which well, accent = signal phase coherence.

Pipeline: SIMULATE -> RENDER STATIC IMAGE
Resolution: 1920x1080
"""

import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "N_STEPS": 12000, "DT": 0.02,
    "AMPLITUDE": 0.30,         # subthreshold drive
    "FREQ": 0.10,              # signal frequency
    "SIGMA_MIN": 0.05, "SIGMA_MAX": 0.85,
    "DPI": 100,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "StochResonance_Final.png"),
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}

WELL_CMAP = LinearSegmentedColormap.from_list(
    "wells",
    ["#ff3050", "#ff9500", "#888888", "#00f2ff", "#0066ff"],
    N=256,
)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def simulate():
    n = CONFIG["N_STEPS"]
    dt = CONFIG["DT"]
    A = CONFIG["AMPLITUDE"]; f = CONFIG["FREQ"]
    s_lo, s_hi = CONFIG["SIGMA_MIN"], CONFIG["SIGMA_MAX"]
    sigma_t = np.linspace(s_lo, s_hi, n)

    np.random.seed(7)
    xs = np.zeros(n); xs[0] = 1.0
    sig = np.zeros(n)
    for i in range(1, n):
        x = xs[i - 1]
        s = sigma_t[i]
        sig[i] = A * np.sin(2 * np.pi * f * i * dt)
        eta = np.random.randn() * np.sqrt(dt)
        dx = (x - x ** 3 + sig[i]) * dt + s * eta
        xs[i] = x + dx
    sig[0] = A * np.sin(0.0)
    return xs, sig, sigma_t


def coherence(xs, sig):
    # Sliding-window correlation between sign(xs) and sign(sig)
    w = 240
    n = len(xs)
    s_x = np.sign(xs); s_s = np.sign(sig)
    coh = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w); hi = min(n, i + 1)
        coh[i] = np.mean(s_x[lo:hi] * s_s[lo:hi])
    return coh


def render_static_image(xs, sig, sigma_t, coh):
    try:
        n = len(xs)
        t = 1.0

        n_vis = n

        azim = -68 + 40 * t
        elev = 18 + 6 * np.sin(t * np.pi)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=CONFIG["DPI"], facecolor=THEME["BG"])

        # Title
        fig.text(0.5, 0.945, "NOISE CAN MAKE YOU HEAR", ha="center",
                 fontsize=28, fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
        fig.text(0.5, 0.905,
                 "Stochastic resonance  ·  bistable well  ·  σ swept low → high",
                 ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

        # Side hook
        fig.text(0.04, 0.78, "Too little noise:",
                 fontsize=14, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.755, "ball is stuck.", fontsize=12, color=THEME["RED"])
        fig.text(0.04, 0.71, "Just right:",
                 fontsize=14, color=THEME["TEXT"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.685, "ball locks to the signal.",
                 fontsize=12, color=THEME["GREEN"])
        fig.text(0.04, 0.64, "Too much:",
                 fontsize=14, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.615, "noise drowns the signal.", fontsize=12, color=THEME["RED"])

        # Main 3D ribbon
        ax = fig.add_axes([0.24, 0.05, 0.55, 0.92], projection="3d", facecolor=THEME["BG"])

        time_axis = np.arange(n_vis)
        x_traj = xs[:n_vis]
        s_traj = sigma_t[:n_vis]

        if n_vis > 1:
            pts = np.array([time_axis, s_traj, x_traj]).T.reshape(-1, 1, 3)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            # Color by which well (sign of x), bright for high coherence
            cnorm = np.clip((x_traj[:-1] + 2) / 4, 0, 1)
            colors = WELL_CMAP(cnorm)
            colors[:, -1] = 0.55 + 0.40 * np.clip(np.abs(coh[:n_vis - 1]), 0, 1)
            lc = Line3DCollection(segments, colors=colors, linewidths=0.6)
            ax.add_collection3d(lc)

            # Glowing leading point
            cur_color = WELL_CMAP(np.clip((x_traj[-1] + 2) / 4, 0, 1))
            ax.scatter([time_axis[-1]], [s_traj[-1]], [x_traj[-1]],
                       s=55, color=cur_color, edgecolors="white",
                       linewidths=0.6, zorder=10)

            # Well guide planes at x = ±1
            for well_x, well_color in [(1.0, THEME["CYAN"]), (-1.0, THEME["RED"])]:
                px, py = np.meshgrid(np.linspace(0, n, 3), np.linspace(sigma_t.min(), sigma_t.max(), 3))
                pz = np.full_like(px, well_x, dtype=float)
                ax.plot_surface(px, py, pz, color=well_color,
                                alpha=0.05, linewidth=0, antialiased=True, shade=False)

        ax.set_xlim(0, n)
        ax.set_ylim(sigma_t.min(), sigma_t.max())
        ax.set_zlim(-2.2, 2.2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.6, 0.9, 0.8))

        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([-1, 0, 1])
        ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=8, pad=0)
        ax.set_xlabel("time", color=THEME["TEXT_DIM"], fontsize=9, labelpad=-8)
        ax.set_ylabel("σ (noise)", color=THEME["TEXT_DIM"], fontsize=9, labelpad=-8)
        ax.set_zlabel("state", color=THEME["TEXT_DIM"], fontsize=9, labelpad=-4)

        # Right panel: live signal + coherence
        # Signal panel
        ax_s = fig.add_axes([0.82, 0.55, 0.16, 0.30], facecolor="#0a0a0a")
        ax_s.plot(sig[:n_vis], color=THEME["YELLOW"], linewidth=0.6, alpha=0.85)
        ax_s.set_xlim(0, n); ax_s.set_ylim(-CONFIG["AMPLITUDE"] * 1.5, CONFIG["AMPLITUDE"] * 1.5)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        for s in ax_s.spines.values(): s.set_color("#222"); s.set_linewidth(0.5)
        ax_s.set_title("forcing signal A·sin(2πft)", color=THEME["TEXT_DIM"], fontsize=9, loc="left", pad=2)

        # State panel
        ax_x = fig.add_axes([0.82, 0.20, 0.16, 0.30], facecolor="#0a0a0a")
        ax_x.plot(xs[:n_vis], color=THEME["CYAN"], linewidth=0.5, alpha=0.85)
        ax_x.axhline(1, color=THEME["CYAN"], linewidth=0.3, alpha=0.4)
        ax_x.axhline(-1, color=THEME["RED"], linewidth=0.3, alpha=0.4)
        ax_x.set_xlim(0, n); ax_x.set_ylim(-2.2, 2.2)
        ax_x.set_xticks([]); ax_x.set_yticks([])
        for s in ax_x.spines.values(): s.set_color("#222"); s.set_linewidth(0.5)
        ax_x.set_title("state x(t)", color=THEME["TEXT_DIM"], fontsize=9, loc="left", pad=2)

        # HUD
        cur_sigma = sigma_t[max(0, n_vis - 1)]
        cur_coh = coh[max(0, n_vis - 1)]
        regime = "stuck" if cur_sigma < 0.18 else ("RESONANT" if cur_sigma < 0.45 else "noisy")
        regime_color = (THEME["RED"] if regime == "stuck"
                        else THEME["GREEN"] if regime == "RESONANT"
                        else THEME["RED"])

        fig.text(0.04, 0.50, "noise level σ", fontsize=10, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.465, f"{cur_sigma:.3f}", fontsize=20, color=THEME["YELLOW"],
                 fontweight="bold")
        fig.text(0.04, 0.435, regime.upper(), fontsize=12, color=regime_color, fontweight="bold")

        fig.text(0.04, 0.380, "phase coherence", fontsize=10, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.350,
                 f"⟨sign(x)·sign(s)⟩ = {cur_coh:+.2f}",
                 fontsize=11, color=THEME["GREEN"] if abs(cur_coh) > 0.4 else THEME["TEXT"])
        fig.text(0.04, 0.325, "(±1 = locked   0 = random)",
                 fontsize=8, color=THEME["TEXT_DIM"], style="italic")

        fig.text(0.04, 0.255,
                 r"$\dot x = x - x^3 + A\sin(2\pi f t) + \sigma\,\eta(t)$",
                 fontsize=11, color=THEME["TEXT"])
        fig.text(0.04, 0.225,
                 f"A = {CONFIG['AMPLITUDE']}    f = {CONFIG['FREQ']}",
                 fontsize=10, color=THEME["TEXT_DIM"])

        fig.text(0.04, 0.150, "Counter-intuition:",
                 fontsize=10, color=THEME["TEXT_DIM"], style="italic")
        fig.text(0.04, 0.127, "noise can amplify",
                 fontsize=11, color=THEME["TEXT"], style="italic")
        fig.text(0.04, 0.105, "what's beneath the threshold.",
                 fontsize=11, color=THEME["TEXT"], style="italic")

        fig.text(0.985, 0.018, "@quant.traderr", ha="right", va="bottom",
                 fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

        fp = CONFIG["OUTPUT_FILE"]
        fig.savefig(fp, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
        plt.close(fig)
        log(f"Image saved: {fp}")
        return True
    except Exception:
        import traceback; traceback.print_exc(); return False


def main():
    t0 = time.time(); log("=== STOCHASTIC RESONANCE ===")
    xs, sig, sigma_t = simulate()
    coh = coherence(xs, sig)
    log(f"sim: n={len(xs)}  sigma range [{sigma_t.min():.2f},{sigma_t.max():.2f}]")

    log("Rendering static image...")
    render_static_image(xs, sig, sigma_t, coh)
    log(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
