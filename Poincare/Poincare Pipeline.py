"""
Poincare_Reel_Pipeline.py
==========================
Cinematic landscape reel: Poincaré section of a chaotic attractor.
Lorenz-63 system trajectory flows in 3D; a transparent plane at z = z*
slices through it. Each upward crossing is recorded as a point on the plane,
revealing the fractal structure of chaos.

Phase 1: trajectory builds, plane visible, no points yet.
Phase 2: plane crossings start firing; points accumulate on the slice.
Phase 3: trajectory fades, plane rotates to face camera, full section reveal.

Pipeline: SIMULATE -> RENDER (parallel) -> COMPILE
Resolution: 1920x1080 @ 30 FPS, ~12s + 2s hold
"""

import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "SIGMA": 10.0, "RHO": 28.0, "BETA": 8 / 3,
    "N_STEPS": 18000, "DT": 0.005, "Z_SLICE": 27.0,
    "DPI": 100,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "Poincare_Section.png"),
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "MAGENTA": "#ff1493", "FONT": "Arial",
}

CHAOS_CMAP = LinearSegmentedColormap.from_list(
    "chaos",
    ["#04001f", "#0066ff", "#00f2ff", "#ffd400", "#ff9500", "#ff3050"],
    N=256,
)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def simulate_lorenz():
    s, r, b, dt = CONFIG["SIGMA"], CONFIG["RHO"], CONFIG["BETA"], CONFIG["DT"]
    n = CONFIG["N_STEPS"]
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n)
    xs[0], ys[0], zs[0] = 1.0, 1.0, 1.0
    for i in range(1, n):
        x, y, z = xs[i - 1], ys[i - 1], zs[i - 1]
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        xs[i] = x + dx * dt
        ys[i] = y + dy * dt
        zs[i] = z + dz * dt
    return xs, ys, zs


def find_crossings(xs, ys, zs, z_slice):
    # Detect upward crossings of z = z_slice (z[i-1] < z_slice <= z[i])
    above = zs >= z_slice
    cross_idx = np.where((~above[:-1]) & above[1:])[0]
    # Linear interpolation on the crossing
    pts = []
    for i in cross_idx:
        z0, z1 = zs[i], zs[i + 1]
        if z1 == z0: continue
        a = (z_slice - z0) / (z1 - z0)
        x = xs[i] + a * (xs[i + 1] - xs[i])
        y = ys[i] + a * (ys[i + 1] - ys[i])
        pts.append((i, x, y))
    return np.array(pts) if pts else np.zeros((0, 3))


def render_image(xs, ys, zs, crossings, z_slice, x_lim, y_lim, z_lim, out_path):
    try:
        n = len(xs)
        n_traj = n
        n_cross = len(crossings)
        
        # Good angle to see both the 3D attractor and the section plane
        traj_alpha = 0.45
        elev = 45
        azim = -45

        fig = plt.figure(figsize=(19.2, 10.8), dpi=CONFIG["DPI"], facecolor=THEME["BG"])

        # Title
        fig.text(0.5, 0.945, "SLICE A CHAOS, FIND A FRACTAL", ha="center",
                 fontsize=28, fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
        fig.text(0.5, 0.905, f"Poincaré section  ·  Lorenz-63  ·  z = {z_slice:.0f}",
                 ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

        # Side hook
        fig.text(0.04, 0.78, "A 3D chaotic flow",
                 fontsize=15, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.74, "looks like noise.",
                 fontsize=15, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.66, "Slice it,",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.62, "and structure appears.",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])

        # 3D plot
        ax = fig.add_axes([0.26, 0.05, 0.70, 0.92], projection="3d", facecolor=THEME["BG"])

        # Trajectory
        x_t, y_t, z_t = xs[:n_traj], ys[:n_traj], zs[:n_traj]
        if len(x_t) > 1:
            pts = np.array([x_t, y_t, z_t]).T.reshape(-1, 1, 3)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            # Color by speed (magnitude of derivative ~ |z|)
            speed = np.abs(z_t[:-1] - z_t[1:])
            sn = (speed - speed.min()) / (np.ptp(speed) + 1e-9)
            colors = CHAOS_CMAP(sn)
            colors[:, -1] = traj_alpha
            lc = Line3DCollection(segments, colors=colors, linewidths=0.55)
            ax.add_collection3d(lc)

        # Slicing plane (semi-transparent)
        plane_x = np.array([[x_lim[0], x_lim[1]], [x_lim[0], x_lim[1]]])
        plane_y = np.array([[y_lim[0], y_lim[0]], [y_lim[1], y_lim[1]]])
        plane_z = np.full_like(plane_x, z_slice, dtype=float)
        ax.plot_surface(plane_x, plane_y, plane_z, color=THEME["CYAN"],
                        alpha=0.15, linewidth=0, antialiased=True, shade=False)

        # Crossings on the plane
        if n_cross > 0:
            cx = crossings[:n_cross, 1]
            cy = crossings[:n_cross, 2]
            cz = np.full_like(cx, z_slice + 0.05)
            pt_size = 20
            ax.scatter(cx, cy, cz, s=pt_size, c=THEME["YELLOW"],
                       edgecolors=THEME["RED"], linewidths=0.5,
                       alpha=0.95, depthshade=False, zorder=10)

        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.85))

        # Strip 3D chrome
        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

        # HUD
        fig.text(0.04, 0.49, "PHASE: REVEALING", fontsize=11, color=THEME["CYAN"],
                 fontweight="bold", family=THEME["FONT"])

        fig.text(0.04, 0.42, "Lorenz-63 system:", fontsize=10, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.395, r"$\dot x = \sigma(y - x)$", fontsize=12, color=THEME["TEXT"])
        fig.text(0.04, 0.370, r"$\dot y = x(\rho - z) - y$", fontsize=12, color=THEME["TEXT"])
        fig.text(0.04, 0.345, r"$\dot z = xy - \beta z$", fontsize=12, color=THEME["TEXT"])

        fig.text(0.04, 0.28, f"σ = {CONFIG['SIGMA']:.0f}   ρ = {CONFIG['RHO']:.0f}   β = {CONFIG['BETA']:.2f}",
                 fontsize=10, color=THEME["YELLOW"], family=THEME["FONT"])
        fig.text(0.04, 0.255, f"crossings: {n_cross} / {len(crossings)}",
                 fontsize=11, color=THEME["RED"], family=THEME["FONT"], fontweight="bold")
        fig.text(0.04, 0.225, "(yellow dots on cyan plane)",
                 fontsize=9, color=THEME["TEXT_DIM"], style="italic")

        fig.text(0.04, 0.13, "The flow looks formless.",
                 fontsize=11, color=THEME["TEXT_DIM"], style="italic")
        fig.text(0.04, 0.105, "The section is a fractal map",
                 fontsize=11, color=THEME["TEXT"], style="italic")
        fig.text(0.04, 0.085, "of chaos itself.",
                 fontsize=11, color=THEME["TEXT"], style="italic")

        fig.text(0.985, 0.018, "@quant.traderr", ha="right", va="bottom",
                 fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

        log(f"Saving static image to {out_path}...")
        fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
        plt.close(fig)
        return True
    except Exception:
        import traceback; traceback.print_exc(); return False


def main():
    t0 = time.time(); log("=== POINCARE SECTION REEL ===")
    xs, ys, zs = simulate_lorenz()
    crossings = find_crossings(xs, ys, zs, CONFIG["Z_SLICE"])
    log(f"Trajectory: {len(xs)} pts   crossings: {len(crossings)}")

    # Discard transient
    burn = 1500
    xs, ys, zs = xs[burn:], ys[burn:], zs[burn:]
    crossings = crossings[crossings[:, 0] > burn]
    crossings[:, 0] -= burn

    pad = 0.05
    x_lim = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs))
    y_lim = (ys.min() - pad * np.ptp(ys), ys.max() + pad * np.ptp(ys))
    z_lim = (zs.min() - pad * np.ptp(zs), zs.max() + pad * np.ptp(zs))

    render_image(xs, ys, zs, crossings, CONFIG["Z_SLICE"], x_lim, y_lim, z_lim, CONFIG["OUTPUT_FILE"])

    log(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
