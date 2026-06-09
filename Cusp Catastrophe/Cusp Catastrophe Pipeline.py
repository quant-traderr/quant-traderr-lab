"""
CuspCatastrophe_Static.py
=========================
Single 1080x1920 PNG of the cusp catastrophe reel at its final state.
Renders the complete trajectory (drift → fold crossing → collapse) in one shot.
"""

import os, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "FPS": 30,
    "DURATION_SEC": 12,
    "WIDTH": 1080, "HEIGHT": 1920,
    "DPI": 150,
    "A_MIN": -1.9, "A_MAX":  0.5,
    "B_MIN": -1.10, "B_MAX": 1.10,
    "Z_MIN": -1.6, "Z_MAX":  1.6,
    "MESH_A": 70, "MESH_Z": 70,
    "SIGMA": 0.045,
    "RATE":  5.0,
    "SUBSTEPS": 40,
    "SEED": 7,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "CuspCatastrophe_Static.png"),
}

THEME = {
    "BG":       "#000000",
    "PANEL":    "#0a0a0a",
    "TEXT":     "#ffffff",
    "TEXT_DIM": "#888888",
    "ORANGE":   "#ff9500",
    "CYAN":     "#00f2ff",
    "YELLOW":   "#ffd400",
    "RED":      "#ff3050",
    "GREEN":    "#00ff8c",
    "MAGENTA":  "#ff1493",
    "FONT":     "Arial",
}

SHEET_CMAP = LinearSegmentedColormap.from_list(
    "sheet",
    ["#ff1f3f", "#ff8f3f", "#3a3a3a", "#3fc6ff", "#00ff8c"],
    N=256,
)

# Color trajectory by time: cyan (early) -> orange (mid) -> red (crash) -> dim red (settled)
TRAJ_CMAP = LinearSegmentedColormap.from_list(
    "traj",
    ["#00f2ff", "#00f2ff", "#ffd400", "#ff9500", "#ff3050", "#661020"],
    N=256,
)


def smoothstep(t, lo, hi):
    x = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_surface():
    a = np.linspace(CONFIG["A_MIN"], CONFIG["A_MAX"], CONFIG["MESH_A"])
    z = np.linspace(CONFIG["Z_MIN"], CONFIG["Z_MAX"], CONFIG["MESH_Z"])
    A, Z = np.meshgrid(a, z)
    B = -Z ** 3 - A * Z
    STABLE = (3.0 * Z ** 2 + A) > 0.0
    return a, z, A, B, Z, STABLE


def fold_curves_3d(n=240):
    a = np.linspace(0.0, CONFIG["A_MIN"], n)
    a = a[a <= 0.0]
    z_pos =  np.sqrt(-a / 3.0)
    z_neg = -np.sqrt(-a / 3.0)
    b_pos = -z_pos ** 3 - a * z_pos
    b_neg = -z_neg ** 3 - a * z_neg
    return a, b_pos, z_pos, b_neg, z_neg


def cusp_boundary_2d(n=300):
    a = np.linspace(0.0, CONFIG["A_MIN"], n)
    a = a[a <= 0.0]
    b = np.sqrt(np.clip(-4.0 * a ** 3 / 27.0, 0.0, None))
    return a, b


def design_trajectory(n_frames):
    t = np.linspace(0.0, 1.0, n_frames)
    a = 0.2 + (-1.6 - 0.2) * smoothstep(t, 0.05, 0.50)
    b = -0.55 + 1.55 * smoothstep(t, 0.62, 0.80)
    rng = np.random.default_rng(CONFIG["SEED"])
    a = a + 0.025 * np.cumsum(rng.standard_normal(n_frames)) / np.sqrt(n_frames)
    return a.astype(float), b.astype(float)


def simulate_ball(a_t, b_t):
    n = len(a_t)
    rng = np.random.default_rng(CONFIG["SEED"] + 1)
    z = np.zeros(n)
    roots = np.roots([1.0, 0.0, a_t[0], b_t[0]])
    real = sorted([r.real for r in roots if abs(r.imag) < 1e-8])
    z[0] = real[-1] if real else 1.0
    dt = 1.0 / CONFIG["FPS"]
    sub_dt = dt / CONFIG["SUBSTEPS"]
    sqrt_sub = np.sqrt(sub_dt)
    for i in range(1, n):
        zi = z[i - 1]
        for _ in range(CONFIG["SUBSTEPS"]):
            grad = zi ** 3 + a_t[i] * zi + b_t[i]
            zi = zi - CONFIG["RATE"] * grad * sub_dt + CONFIG["SIGMA"] * rng.standard_normal() * sqrt_sub
        z[i] = zi
    return z


def render_static(a_traj, b_traj, z_traj, surface_data):
    a_grid, z_grid, A, B, Z, STABLE = surface_data
    n = len(a_traj)
    t_vals = np.linspace(0.0, 1.0, n)

    figsize = (CONFIG["WIDTH"] / CONFIG["DPI"], CONFIG["HEIGHT"] / CONFIG["DPI"])
    fig = plt.figure(figsize=figsize, dpi=CONFIG["DPI"], facecolor=THEME["BG"])

    # ---------- HERO 3D MANIFOLD ----------
    ax3d = fig.add_axes([-0.06, 0.30, 1.12, 0.64], projection="3d", facecolor=THEME["BG"])

    z_norm = (Z - CONFIG["Z_MIN"]) / (CONFIG["Z_MAX"] - CONFIG["Z_MIN"])
    rgba = SHEET_CMAP(z_norm)
    rgba[..., 3] = np.where(STABLE, 0.42, 0.10)
    ax3d.plot_surface(A, B, Z, facecolors=rgba,
                      rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)

    # Yellow fold edges
    a_f, b_fp, z_fp, b_fn, z_fn = fold_curves_3d()
    ax3d.plot(a_f, b_fp, z_fp, color=THEME["YELLOW"], linewidth=1.8, alpha=0.95)
    ax3d.plot(a_f, b_fn, z_fn, color=THEME["YELLOW"], linewidth=1.8, alpha=0.95)

    # Full trajectory colored by time
    pts = np.array([a_traj, b_traj, z_traj]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
    colors = TRAJ_CMAP(t_mid)
    colors[:, 3] = np.clip(0.35 + 0.65 * t_mid, 0, 1)
    lc = Line3DCollection(segs, colors=colors, linewidths=2.2)
    ax3d.add_collection3d(lc)

    # Final ball (settled on lower sheet) with glow
    zc = z_traj[-1]
    for s, a in [(560, 0.06), (280, 0.16), (130, 0.38), (55, 1.00)]:
        ax3d.scatter([a_traj[-1]], [b_traj[-1]], [zc],
                     s=s, color=THEME["RED"], alpha=a, edgecolors="none", depthshade=False)

    # Camera at final angle (matching reel t=1)
    ax3d.view_init(elev=24.0, azim=-78.0)
    ax3d.set_xlim(CONFIG["A_MIN"], CONFIG["A_MAX"])
    ax3d.set_ylim(CONFIG["B_MIN"], CONFIG["B_MAX"])
    ax3d.set_zlim(CONFIG["Z_MIN"], CONFIG["Z_MAX"])
    ax3d.set_box_aspect((1.6, 1.1, 1.0))

    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_alpha(0); pane.set_edgecolor((0, 0, 0, 0))
    ax3d.grid(False)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_xlabel("a  speculation",  color=THEME["TEXT_DIM"], fontsize=9, labelpad=-8)
    ax3d.set_ylabel("b  bias",         color=THEME["TEXT_DIM"], fontsize=9, labelpad=-8)
    ax3d.set_zlabel("z  state",        color=THEME["TEXT_DIM"], fontsize=9, labelpad=-4)

    # ---------- CONTROL PLANE ----------
    ax2d = fig.add_axes([0.12, 0.13, 0.76, 0.16], facecolor=THEME["PANEL"])
    a_c, b_c = cusp_boundary_2d()
    ax2d.fill_between(a_c, -b_c, b_c, color=THEME["MAGENTA"], alpha=0.16)
    ax2d.plot(a_c,  b_c, color=THEME["YELLOW"], linewidth=1.1)
    ax2d.plot(a_c, -b_c, color=THEME["YELLOW"], linewidth=1.1)

    # Full trajectory in control plane, colored by time
    for i in range(n - 1):
        ax2d.plot(a_traj[i:i+2], b_traj[i:i+2],
                  color=TRAJ_CMAP(t_vals[i]), linewidth=1.5,
                  alpha=0.5 + 0.5 * t_vals[i], solid_capstyle="round")

    ax2d.scatter([a_traj[-1]], [b_traj[-1]],
                 s=46, color=THEME["RED"], zorder=10, edgecolors="white", linewidths=0.6)

    ax2d.set_xlim(CONFIG["A_MIN"], CONFIG["A_MAX"])
    ax2d.set_ylim(CONFIG["B_MIN"], CONFIG["B_MAX"])
    ax2d.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    ax2d.set_xlabel("a", color=THEME["TEXT_DIM"], fontsize=9, labelpad=2)
    ax2d.set_ylabel("b", color=THEME["TEXT_DIM"], fontsize=9, labelpad=2)
    ax2d.set_title("control plane (a, b)   shaded = bistable region",
                   color=THEME["TEXT_DIM"], fontsize=9, loc="left", pad=4)
    for s in ax2d.spines.values(): s.set_color("#222"); s.set_linewidth(0.5)

    # ---------- HEADER ----------
    fig.text(0.5, 0.965, "CRASHES HAVE GEOMETRY.",
             ha="center", fontsize=24, fontweight="bold",
             color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.940,
             "Zeeman's cusp catastrophe   ·   drift becomes collapse",
             ha="center", fontsize=11, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, 0.915, "Equilibrium destroyed.",
             ha="center", fontsize=12, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"], style="italic")

    # ---------- HUD ----------
    fig.text(0.05, 0.085, "potential", fontsize=9, color=THEME["TEXT_DIM"])
    fig.text(0.05, 0.060,
             r"$V(z)=\frac{1}{4}z^{4}+\frac{a}{2}z^{2}+bz$",
             fontsize=12, color=THEME["TEXT"])
    fig.text(0.05, 0.032, "bifurcation curve", fontsize=9, color=THEME["TEXT_DIM"])
    fig.text(0.05, 0.010, r"$4a^{3}+27b^{2}=0$",
             fontsize=11, color=THEME["YELLOW"])

    fig.text(0.56, 0.085, "state z", fontsize=9, color=THEME["TEXT_DIM"])
    fig.text(0.56, 0.052, f"{zc:+.3f}",
             fontsize=19, color=THEME["RED"], fontweight="bold")

    fig.text(0.76, 0.085, "(a, b)", fontsize=9, color=THEME["TEXT_DIM"])
    fig.text(0.76, 0.058,
             f"({a_traj[-1]:+.2f}, {b_traj[-1]:+.2f})",
             fontsize=11, color=THEME["TEXT"])

    fig.text(0.95, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.65, family=THEME["FONT"])

    fig.savefig(CONFIG["OUTPUT_FILE"], dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    print(f"Saved: {CONFIG['OUTPUT_FILE']}")


def main():
    import time
    t0 = time.time()
    print("Building surface...")
    surface_data = build_surface()
    print("Designing trajectory...")
    n_frames = CONFIG["FPS"] * CONFIG["DURATION_SEC"]
    a_traj, b_traj = design_trajectory(n_frames)
    print("Simulating ball...")
    z_traj = simulate_ball(a_traj, b_traj)
    print("Rendering static image...")
    render_static(a_traj, b_traj, z_traj, surface_data)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
