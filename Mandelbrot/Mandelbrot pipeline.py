"""
mandelbrot_static_pipeline.py
=============================
Branded static hero generator for the Mandelbrot set, in the dark "Bloomberg"
house style (same signature spectral palette as the Lorenz-BTC reel). Full-bleed
fractal + left-scrim HUD: title, fractal-markets framing, the z^2+c equation, and
live coord/zoom/iteration readouts + @quant.dhawan.

Parametrized with named deep-zoom presets. Pure-numpy escape-time with smooth
(continuous) iteration coloring and an alive-mask for speed.

    python mandelbrot_static_pipeline.py                      # full set, branded
    python mandelbrot_static_pipeline.py --preset seahorse    # deep zoom preset
    python mandelbrot_static_pipeline.py --preset mini --dpi 300
    python mandelbrot_static_pipeline.py --no-hud --cmap ember # clean frame, gold
    python mandelbrot_static_pipeline.py --list               # list presets

Output: <preset>_hero.png (16:9), default 2560x1440.
"""
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

BASE = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = os.path.normpath(os.path.join(BASE, "..", "Demiurge", "fonts"))

THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "DIM": "#8a93a0",
    "CYAN": "#00f2ff", "GOLD": "#ffd24a", "ORANGE": "#ff9500",
}

# Signature reel palette (matches Lorenz-BTC) + an ember alt
CMAPS = {
    "spectral": LinearSegmentedColormap.from_list(
        "spectral", ["#05010a", "#0a1a4a", "#0066ff", "#00f2ff",
                     "#ffd400", "#ff7a18", "#ff3050", "#ffffff"], N=512),
    "ember": LinearSegmentedColormap.from_list(
        "ember", ["#05040a", "#1a0e08", "#5c2e0a", "#b5611a",
                  "#e89a2c", "#f7c64e", "#fff0c0", "#ffffff"], N=512),
}

# center (re, im), span (real-axis width), iters, color stretch lo/hi pct, gamma
PRESETS = {
    "full":     dict(center=(-0.60, 0.0),       span=3.2,   iters=500,
                     lo=0, hi=99.0, gamma=0.70, label="THE FULL SET"),
    "seahorse": dict(center=(-0.745, 0.1135),   span=0.085, iters=900,
                     lo=20, hi=99.5, gamma=0.85, label="SEAHORSE VALLEY"),
    "elephant": dict(center=(0.2855, 0.0115),   span=0.045, iters=900,
                     lo=20, hi=99.5, gamma=0.85, label="ELEPHANT VALLEY"),
    "spiral":   dict(center=(-0.74364, 0.13182), span=0.0055, iters=1400,
                     lo=25, hi=99.6, gamma=0.90, label="SEAHORSE SPIRAL"),
    "mini":     dict(center=(-1.76877, 0.00174), span=0.0045, iters=1600,
                     lo=20, hi=99.6, gamma=0.88, label="MINI-MANDELBROT"),
}


def register_fonts():
    fam = {}
    for fn, key in [("SpaceGrotesk-Variable.ttf", "display"),
                    ("JetBrainsMono-Bold.ttf", "mono_b"),
                    ("JetBrainsMono-Regular.ttf", "mono")]:
        p = os.path.join(FONTS_DIR, fn)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            fam[key] = font_manager.FontProperties(fname=p).get_name()
    return fam

_F = register_fonts()
DISPLAY = _F.get("display", "DejaVu Sans")
MONO = _F.get("mono", "DejaVu Sans Mono")
MONO_B = _F.get("mono_b", MONO)


def compute(center, span, iters, w, h):
    """Smooth escape-time. Returns (norm_div, escaped_mask), both h x w."""
    cx, cy = center
    aspect = h / w
    re = np.linspace(cx - span / 2, cx + span / 2, w)
    im = np.linspace(cy - span * aspect / 2, cy + span * aspect / 2, h)
    C = re[np.newaxis, :] + 1j * im[:, np.newaxis]
    Z = np.zeros_like(C)
    div = np.full(C.shape, float(iters))
    alive = np.ones(C.shape, dtype=bool)
    for n in range(iters):
        Z[alive] = Z[alive] ** 2 + C[alive]
        mag2 = Z.real * Z.real + Z.imag * Z.imag
        esc = alive & (mag2 > (1 << 16))
        if esc.any():
            div[esc] = n + 1 - np.log(np.log(np.sqrt(mag2[esc]))) / np.log(2)
        alive &= ~esc
        if not alive.any():
            break
    return div, ~alive  # escaped where no longer alive... see note below


def colorize(div, escaped, cmap, lo, hi, gamma):
    e = div[escaped]
    if e.size == 0:
        norm = np.zeros_like(div)
    else:
        vlo, vhi = np.percentile(e, lo), np.percentile(e, hi)
        norm = np.clip((div - vlo) / (vhi - vlo + 1e-9), 0, 1) ** gamma
    norm[~escaped] = 0.0          # interior -> black
    rgb = cmap(norm)[:, :, :3]
    rgb[~escaped] = 0.0
    return rgb


def left_scrim(w, h):
    """RGBA overlay: dark on the left, fading to transparent, for HUD legibility."""
    x = np.linspace(0, 1, w)
    a = np.clip(1.0 - x / 0.46, 0, 1) ** 1.4 * 0.82
    yb = np.linspace(0, 1, h)[:, None]
    a = np.maximum(a[None, :], (np.clip((yb - 0.86) / 0.14, 0, 1) ** 1.5) * 0.7)
    rgba = np.zeros((h, w, 4))
    rgba[..., 3] = a
    return rgba


def compose(rgb, preset, name, dpi, hud=True):
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=THEME["BG"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(rgb, origin="lower", aspect="auto", interpolation="bilinear")

    if not hud:
        return fig

    ax.imshow(left_scrim(w, h), origin="lower", aspect="auto", zorder=5)

    cx, cy = preset["center"]
    T = THEME
    # title block
    fig.text(0.035, 0.90, "FRACTAL MARKETS", fontsize=34, color=T["TEXT"],
             family=DISPLAY, fontweight="bold", zorder=10)
    fig.text(0.035, 0.862, "The Mandelbrot Set", fontsize=15, color=T["GOLD"],
             family=DISPLAY, zorder=10)

    # reframe hook
    fig.text(0.035, 0.78, "Most see a smooth curve.", fontsize=14, color=T["DIM"],
             family=DISPLAY, style="italic", zorder=10)
    fig.text(0.035, 0.752, "Mandelbrot saw infinite", fontsize=14, color=T["TEXT"],
             family=DISPLAY, style="italic", zorder=10)
    fig.text(0.035, 0.726, "roughness at every scale.", fontsize=14,
             color=T["CYAN"], family=DISPLAY, style="italic", zorder=10)

    # equation
    fig.text(0.035, 0.64, r"$z_{n+1} = z_n^{2} + c$", fontsize=20,
             color=T["TEXT"], zorder=10)
    fig.text(0.035, 0.60, "iterate. stay bounded = in the set.", fontsize=11,
             color=T["DIM"], family=MONO, zorder=10)

    # readouts
    fig.text(0.035, 0.50, "REGION", fontsize=10, color=T["DIM"], family=MONO,
             zorder=10)
    fig.text(0.035, 0.468, preset["label"], fontsize=16, color=T["GOLD"],
             family=MONO_B, zorder=10)
    fig.text(0.035, 0.40, "CENTER", fontsize=10, color=T["DIM"], family=MONO,
             zorder=10)
    fig.text(0.035, 0.372, f"{cx:+.5f}, {cy:+.5f} i", fontsize=12, color=T["TEXT"],
             family=MONO, zorder=10)
    fig.text(0.035, 0.335, "SPAN", fontsize=10, color=T["DIM"], family=MONO,
             zorder=10)
    fig.text(0.035, 0.307, f"{preset['span']:.2e}", fontsize=12, color=T["CYAN"],
             family=MONO, zorder=10)
    fig.text(0.035, 0.270, "MAX ITER", fontsize=10, color=T["DIM"], family=MONO,
             zorder=10)
    fig.text(0.035, 0.242, f"{preset['iters']}", fontsize=12, color=T["ORANGE"],
             family=MONO, zorder=10)

    fig.text(0.035, 0.155, "Pure math meets the", fontsize=11,
             color=T["DIM"], family=DISPLAY, style="italic", zorder=10)
    fig.text(0.035, 0.130, "geometry of how markets", fontsize=11,
             color=T["DIM"], family=DISPLAY, style="italic", zorder=10)
    fig.text(0.035, 0.105, "actually move.", fontsize=11, color=T["TEXT"],
             family=DISPLAY, style="italic", zorder=10)

    fig.text(0.985, 0.035, "@quant.dhawan", ha="right", fontsize=13,
             color=T["DIM"], family=DISPLAY, alpha=0.85, zorder=10)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200, help="200 -> 2560x1440")
    ap.add_argument("--cmap", default="spectral", choices=list(CMAPS))
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in PRESETS.items():
            print(f"  {k:10s} center={v['center']} span={v['span']:.2e} "
                  f"iters={v['iters']}  ({v['label']})")
        return

    t0 = time.time()
    p = PRESETS[args.preset]
    w, h = int(12.8 * args.dpi), int(7.2 * args.dpi)
    print(f"[render] {args.preset}  {w}x{h}  iters={p['iters']} ...")

    div, escaped = compute(p["center"], p["span"], p["iters"], w, h)
    rgb = colorize(div, escaped, CMAPS[args.cmap], p["lo"], p["hi"], p["gamma"])
    fig = compose(rgb, p, args.preset, args.dpi, hud=not args.no_hud)

    out = args.out or os.path.join(BASE, f"{args.preset}_hero.png")
    fig.savefig(out, dpi=args.dpi, facecolor=THEME["BG"], pad_inches=0)
    plt.close(fig)
    print(f"[done] {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
