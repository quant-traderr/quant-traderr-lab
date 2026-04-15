"""
Rough_Volatility_Pipeline.py
============================
Production-ready pipeline for Rough Volatility / Fractional Brownian Motion.

3D surface of fBM sample paths parameterized by the Hurst exponent H:
    X = time  (0 -> 1)
    Y = Hurst parameter  (0.05 -> 0.95)
    Z = B_H(t)  path value  (unit-variance normalised)

Each path is normalised to unit variance so the visual contrast is
purely in TEXTURE (roughness), not amplitude.  The surface is jagged
at low H (rough-vol regime, H ~ 0.1) and perfectly smooth at high H.

Key result: real realized volatility has H ~ 0.07-0.14
            (Gatheral, Jaisson & Rosenbaum 2018).

Pipeline:  GENERATE  ->  VISUALIZE (static PNG)
Output:    1920x1080 PNG
Deps:      pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION":     (1920, 1080),
    "DPI":            100,
    "OUTPUT_IMAGE":   "Rough_Volatility_Output.png",
    "LOG_FILE":       os.path.join(BASE_DIR, "roughvol_pipeline.log"),
}

# fBM surface parameters
FBM = {
    "N_TIME":    250,     # time steps per fBM path (Cholesky)
    "N_H":       80,      # number of Hurst values
    "H_MIN":     0.05,
    "H_MAX":     0.95,
    "SEED":      42,
    "N_DISP_T":  140,     # display grid: time axis
    "N_DISP_H":  60,      # display grid: H axis
}

# ─── THEME (Bloomberg Dark) ──────────────────────────────────────
THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "ORANGE":     "#ff9500",
    "ORANGE_HOT": "#ff6b00",
    "YELLOW":     "#ffd400",
    "RED":        "#ff3050",
    "CYAN":       "#00f2ff",
    "GREEN":      "#00ff41",
    "MAGENTA":    "#ff1493",
    "FONT":       "Arial",
}

# Surface colored by H: rough(red/orange) -> BM(yellow) -> smooth(cyan/violet)
SURFACE_CMAP = LinearSegmentedColormap.from_list("rough_smooth", [
    "#ff3050",       # H=0.05 : hot red    (ROUGH)
    "#ff6b00",       # ~0.28  : orange
    "#ffd400",       # H=0.50 : yellow     (BM boundary)
    "#00d4ff",       # ~0.72  : cyan
    "#8844ff",       # H=0.95 : violet     (SMOOTH)
])


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════
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
# MODULE 1 — fBM SURFACE GENERATOR
# ═══════════════════════════════════════════════════════════════════
def generate_fbm_surface():
    """Build (n_H x n_time+1) surface of fBM paths via Cholesky.

    All paths share the same Gaussian vector Z so the surface
    transitions smoothly across H.  Each path is normalised to
    unit variance so visual contrast is purely in texture.
    """
    n_t    = FBM["N_TIME"]
    n_h    = FBM["N_H"]
    h_vals = np.linspace(FBM["H_MIN"], FBM["H_MAX"], n_h)
    t_arr  = np.linspace(1.0 / n_t, 1.0, n_t)

    rng = np.random.RandomState(FBM["SEED"])
    Z   = rng.randn(n_t)

    surface = np.zeros((n_h, n_t + 1))          # col-0 = t=0 anchor

    T1, T2   = np.meshgrid(t_arr, t_arr)
    abs_diff = np.abs(T1 - T2)

    for i, H in enumerate(h_vals):
        C = 0.5 * (np.abs(T1) ** (2 * H)
                  + np.abs(T2) ** (2 * H)
                  - abs_diff ** (2 * H))
        C[np.diag_indices_from(C)] += 1e-10
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C[np.diag_indices_from(C)] += 1e-6
            L = np.linalg.cholesky(C)

        path = L @ Z
        # ── normalise to unit variance ────────────────────────
        std = np.std(path)
        if std > 1e-8:
            path = path / std

        surface[i, 0]  = 0.0
        surface[i, 1:] = path

        if (i + 1) % 10 == 0:
            log(f"  fBM path {i + 1}/{n_h}  (H = {H:.3f})")

    t_full = np.concatenate([[0.0], t_arr])
    return t_full, h_vals, surface


def downsample_for_display(t_full, h_vals, surface):
    nt = FBM["N_DISP_T"]
    nh = FBM["N_DISP_H"]
    ti = np.linspace(0, len(t_full) - 1, nt).astype(int)
    hi = np.linspace(0, len(h_vals) - 1, nh).astype(int)
    return t_full[ti], h_vals[hi], surface[np.ix_(hi, ti)]


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — VISUALIZATION (static)
# ═══════════════════════════════════════════════════════════════════
def visualize(t_arr, h_vals, surf):
    """Render the full fBM surface as a single static snapshot."""
    log("[Visual] Generating static snapshot ...")

    elev, azim, dist = 28, 240, 1.10

    T_g, H_g = np.meshgrid(t_arr, h_vals)

    z_lo    = surf.min() - 0.3
    z_hi    = surf.max() + 0.3
    z_floor = z_lo - 0.8

    # ── figure ────────────────────────────────────────────────
    fig = plt.figure(
        figsize=(CONFIG["RESOLUTION"][0] / CONFIG["DPI"],
                 CONFIG["RESOLUTION"][1] / CONFIG["DPI"]),
        dpi=CONFIG["DPI"],
        facecolor=THEME["BG"],
    )
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.83],
                      projection="3d", computed_zorder=False)
    ax.set_facecolor(THEME["BG"])

    pane = (0.02, 0.02, 0.02, 1)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.6)
        a._axinfo["grid"]["linewidth"] = 0.4

    # ── facecolors: colour each row by its H value ────────────
    norm_h = Normalize(FBM["H_MIN"], FBM["H_MAX"])
    fc = np.zeros((*H_g.shape, 4))
    for i in range(len(h_vals)):
        fc[i] = SURFACE_CMAP(norm_h(h_vals[i]))
    fc[:, :, 3] = 0.95

    # ── main surface ──────────────────────────────────────────
    ax.plot_surface(
        T_g, H_g, surf,
        facecolors=fc,
        rstride=1, cstride=1,
        edgecolor=(0.5, 0.5, 0.5, 0.05),
        linewidth=0.15,
        antialiased=True, zorder=1,
    )

    # ── reference paths at H = 0.1 / 0.5 / 0.9 ──────────────
    ref_lines = [
        (0.10, THEME["ORANGE_HOT"], 3.0),     # ROUGH VOL
        (0.50, THEME["YELLOW"],     2.5),      # STANDARD BM
        (0.90, THEME["CYAN"],       2.0),      # SMOOTH
    ]
    for target_h, color, lw in ref_lines:
        ri = np.argmin(np.abs(h_vals - target_h))
        h_val = h_vals[ri]
        path  = surf[ri]
        ax.plot(t_arr, np.full_like(t_arr, h_val), path,
                color=color, lw=lw, alpha=1.0, zorder=15)
        # floor shadow
        ax.plot(t_arr, np.full_like(t_arr, h_val),
                np.full_like(t_arr, z_floor),
                color=color, lw=1.0, alpha=0.35, zorder=3)

    # ── H = 0.5 floor divider (dashed yellow) ────────────────
    ax.plot([0, 1], [0.5, 0.5], [z_floor, z_floor],
            color=THEME["YELLOW"], lw=0.8, alpha=0.4,
            ls="--", zorder=2)

    # ── axes styling ──────────────────────────────────────────
    ax.set_xlabel("TIME  t", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=14,
                  fontfamily=THEME["FONT"])
    ax.set_ylabel("HURST  H", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=14,
                  fontfamily=THEME["FONT"])
    ax.set_zlabel(r"$B_H(t)$", fontsize=12, fontweight="bold",
                  color=THEME["TEXT_DIM"], labelpad=12,
                  fontfamily=THEME["FONT"])
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(FBM["H_MIN"], FBM["H_MAX"])
    ax.set_zlim(z_floor, z_hi)

    ax.set_box_aspect([1.5 * dist, 1.0 * dist, 1.1 * dist])
    ax.view_init(elev=elev, azim=azim)

    # ── Title bar ─────────────────────────────────────────────
    fig.text(0.50, 0.955, "ROUGH VOLATILITY",
             ha="center", va="center", fontsize=30, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=THEME["FONT"])

    fig.text(0.50, 0.918,
             r"$\mathrm{Cov}(B_H(s),\, B_H(t))"
             r" \;=\; \frac{1}{2}\left(|s|^{2H} + |t|^{2H}"
             r" - |t-s|^{2H}\right)$",
             ha="center", va="center", fontsize=14,
             color=THEME["TEXT"], fontfamily=THEME["FONT"])

    fig.text(0.50, 0.886,
             r"FRACTIONAL BROWNIAN MOTION    "
             r"$H \in [0.05,\; 0.95]$    "
             r"REAL VOL:  $H \approx 0.1$",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    # ── Legend (3 reference paths) ────────────────────────────
    fig.text(0.03, 0.895,
             "ORANGE = ROUGH VOL  (H \u2248 0.1)\n"
             "YELLOW = BM  (H = 0.5)\n"
             "CYAN = SMOOTH  (H = 0.9)",
             ha="left", va="center", fontsize=9, fontweight="bold",
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    # ── Footer ────────────────────────────────────────────────
    fig.text(0.98, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"],
             alpha=0.6)

    # ── Save ──────────────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, CONFIG["OUTPUT_IMAGE"])
    fig.savefig(out_path, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log("=" * 60)
    log("Rough Volatility Pipeline START")
    log("=" * 60)

    # 1. Generate fBM surface
    log("Generating fBM surface (Cholesky + unit-variance normalisation) ...")
    t_full, h_vals, surface = generate_fbm_surface()
    log(f"  Raw surface shape: {surface.shape}")
    log(f"  Z range: [{surface.min():.3f}, {surface.max():.3f}]")

    # 2. Downsample for display
    t_disp, h_disp, s_disp = downsample_for_display(t_full, h_vals, surface)
    log(f"  Display grid: H={s_disp.shape[0]} x T={s_disp.shape[1]}")

    # 3. Visualize
    visualize(t_disp, h_disp, s_disp)

    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
