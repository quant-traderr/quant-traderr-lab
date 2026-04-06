"""
Avellaneda_Stoikov_Pipeline.py
==============================
Cinematic timelapse of the Avellaneda-Stoikov optimal market-making model.

Left panel  : 3D surface — optimal bid spread  s − s_b(t, q)
Right panel : 4-stack 2D simulation — Price+Quotes · Inventory · Cash · PnL

Pipeline: SURFACE → SIMULATE → RENDER (parallel) → COMPILE
Resolution: 1920×1080 @ 30 FPS, Bloomberg Dark aesthetic

Dependencies: pip install numpy matplotlib imageio imageio-ffmpeg
"""

import os, sys, time, warnings, shutil
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from multiprocessing import Pool
import imageio

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "FPS": 30,
    "DURATION_SEC": 13,
    "HOLD_LAST_SEC": 2,
    "RESOLUTION": (1920, 1080),
    "TEMP_DIR": "temp_avellaneda_frames",
    "OUTPUT_FILE": "Avellaneda_Stoikov_Output.mp4",
    "LOG_FILE": "avellaneda_pipeline.log",
    "N_WORKERS": 6,
    "PHASE1_END": 3.0,   # materialize
    "PHASE2_END": 11.0,  # orbit + sweep
}

# 3D surface params (tuned to match Avellaneda-Stoikov Fig. 1 shape)
SURFACE = {
    "gamma": 0.001,
    "sigma": 0.3,
    "k": 0.3,
    "T": 600.0,
    "n_grid": 50,
    "q_lo": -30,
    "q_hi": 30,
}

# Simulation params (slightly different γ for tighter inventory control)
SIM = {
    "S0": 100.0,
    "sigma": 0.3,
    "gamma": 0.005,
    "k": 0.3,
    "A": 0.9,
    "T": 600.0,
    "dt": 1.0,
    "seed": 42,
}

# ─── THEME ────────────────────────────────────────────────────────
THEME = {
    "BG":        "#0b0b0b",
    "PANEL_BG":  "#0e0e0e",
    "GRID":      "#1a1a1a",
    "GRID_ALT":  "#1f1f1f",
    "SPINE":     "#333333",
    "TEXT":      "#ffffff",
    "TEXT_DIM":  "#888888",
    "TEXT_SEC":  "#c0c0c0",
    "CYAN":      "#00f2ff",
    "GREEN":     "#00ff41",
    "RED":       "#ff0055",
    "ORANGE":    "#ff9800",
    "BLUE":      "#00bfff",
    "YELLOW":    "#ffcc00",
    "MAGENTA":   "#ff1493",
    "FONT":      "DejaVu Sans",
}

CMAP_CYAN = LinearSegmentedColormap.from_list("as_cyan", [
    "#0b0b0b", "#002233", "#004466", "#006699",
    "#0099cc", "#00ccff", "#00f2ff",
])

# ═══════════════════════════════════════════════════════════════════
# LOGGING
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
# MODULE 1 — DATA
# ═══════════════════════════════════════════════════════════════════

def compute_surface():
    """Optimal bid spread surface: s − s_b(t, q)."""
    p = SURFACE
    t_arr = np.linspace(0, p["T"], p["n_grid"])
    q_arr = np.linspace(p["q_lo"], p["q_hi"], p["n_grid"])
    T_g, Q_g = np.meshgrid(t_arr, q_arr)

    tau = p["T"] - T_g
    # s - s_b = γσ²(T-t)(q + 1/2) + (1/γ)ln(1 + γ/k)
    Z = (p["gamma"] * p["sigma"]**2 * tau * (Q_g + 0.5)
         + (1 / p["gamma"]) * np.log(1 + p["gamma"] / p["k"]))
    return t_arr, q_arr, T_g, Q_g, Z


def run_simulation():
    """Simulate the A-S market maker for one session."""
    p = SIM
    np.random.seed(p["seed"])
    n = int(p["T"] / p["dt"])

    t   = np.linspace(0, p["T"], n)
    mid = np.zeros(n)
    bid = np.zeros(n)
    ask = np.zeros(n)
    inv = np.zeros(n)
    cash = np.zeros(n)
    pnl  = np.zeros(n)

    mid[0] = p["S0"]
    q, c = 0.0, 0.0

    for i in range(1, n):
        tau = max(p["T"] - t[i], 1e-6)

        # price evolution (arithmetic BM)
        mid[i] = mid[i - 1] + p["sigma"] * np.sqrt(p["dt"]) * np.random.randn()
        s = mid[i]

        # reservation price & optimal spread
        r = s - q * p["gamma"] * p["sigma"]**2 * tau
        delta = (p["gamma"] * p["sigma"]**2 * tau
                 + (2 / p["gamma"]) * np.log(1 + p["gamma"] / p["k"]))
        dh = delta / 2

        bid[i] = r - dh
        ask[i] = r + dh

        # order arrivals
        lam_b = p["A"] * np.exp(-p["k"] * max(s - bid[i], 0))
        lam_a = p["A"] * np.exp(-p["k"] * max(ask[i] - s, 0))

        if np.random.rand() < 1 - np.exp(-lam_b * p["dt"]):
            q += 1;  c -= bid[i]
        if np.random.rand() < 1 - np.exp(-lam_a * p["dt"]):
            q -= 1;  c += ask[i]

        inv[i]  = q
        cash[i] = c
        pnl[i]  = c + q * s

    bid[0], ask[0] = bid[1], ask[1]
    return dict(t=t, mid=mid, bid=bid, ask=ask, inv=inv, cash=cash, pnl=pnl)


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — RENDERING
# ═══════════════════════════════════════════════════════════════════

_SH = {}  # shared data for workers

def _init_worker(data):
    global _SH
    _SH = data


def _ss(x):
    """Smoothstep 0→1."""
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _style_2d(ax, xlabel=False):
    """Apply Bloomberg Dark to a 2-D axes."""
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"]); sp.set_linewidth(0.6)
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9,
                   direction="in", length=4)
    ax.yaxis.grid(True, lw=0.3, alpha=0.4, color=THEME["GRID_ALT"])
    ax.xaxis.grid(False)
    if not xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def render_frame(idx):
    """Render a single frame to PNG."""
    data = _SH
    fp = os.path.join(data["tmp"], f"frame_{idx:04d}.png")
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        return True
    try:
        # ── progress & camera ──────────────────────────────────────
        total  = data["total"]
        fps    = CONFIG["FPS"]
        p1     = int(CONFIG["PHASE1_END"] * fps)
        p2     = int(CONFIG["PHASE2_END"] * fps)
        active = total - fps * CONFIG["HOLD_LAST_SEC"]

        prog   = min(idx / max(active - 1, 1), 1.0)
        sim    = data["sim"]
        si     = max(int(prog * (len(sim["t"]) - 1)), 1)
        ct     = sim["t"][si]

        if idx <= p1:
            t = _ss(idx / max(p1, 1))
            elev, azim, salpha = 20 + 15*t, -65 + 20*t, 0.08 + 0.72*t
        elif idx <= p2:
            t = (idx - p1) / max(p2 - p1, 1)
            elev = 35 - 5*np.sin(t * np.pi * 0.5)
            azim = -45 + 50*t
            salpha = 0.82
        else:
            t = _ss((idx - p2) / max(total - p2, 1))
            elev, azim, salpha = 30 + 5*t, 5 + 10*t, 0.88

        # ── figure ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])

        gs = gridspec.GridSpec(
            4, 2, width_ratios=[1, 1.15],
            height_ratios=[1.2, 1, 1, 1.2],
            hspace=0.30, wspace=0.28,
            left=0.06, right=0.97, top=0.86, bottom=0.06,
        )

        # ── LEFT: 3-D surface ─────────────────────────────────────
        ax = fig.add_subplot(gs[:, 0], projection="3d",
                             computed_zorder=False)
        ax.set_facecolor(THEME["BG"])
        pane = (0.043, 0.043, 0.043, 1)
        ax.xaxis.set_pane_color(pane)
        ax.yaxis.set_pane_color(pane)
        ax.zaxis.set_pane_color(pane)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a._axinfo["grid"]["color"] = (0.1, 0.1, 0.1, 0.5)
            a._axinfo["grid"]["linewidth"] = 0.4

        t_arr, q_arr, Tg, Qg, Z = data["surf"]
        ax.plot_surface(Tg, Qg, Z, cmap=CMAP_CYAN, alpha=salpha,
                        rstride=2, cstride=2,
                        edgecolor=(0, 0.95, 1, 0.10), linewidth=0.3,
                        antialiased=True, zorder=1)

        # time-sweep line on surface
        ti = np.argmin(np.abs(t_arr - ct))
        ax.plot([ct]*len(q_arr), q_arr, Z[:, ti],
                color=THEME["CYAN"], lw=2.8, alpha=0.95, zorder=5)
        # ground shadow
        ax.plot([ct]*len(q_arr), q_arr,
                np.full_like(q_arr, Z.min() - 0.15),
                color=THEME["CYAN"], lw=1.0, alpha=0.25, zorder=2)

        # current-state dot
        qi = np.clip(sim["inv"][si], SURFACE["q_lo"], SURFACE["q_hi"])
        zi = np.interp(qi, q_arr, Z[:, ti])
        ax.scatter([ct], [qi], [zi], color=THEME["CYAN"],
                   s=60, edgecolors="white", linewidths=0.8,
                   zorder=10, depthshade=False)

        ax.set_xlabel("Time [s]", fontsize=11, color=THEME["TEXT_DIM"],
                       fontfamily=THEME["FONT"], labelpad=10)
        ax.set_ylabel("Inventory  q", fontsize=11, color=THEME["TEXT_DIM"],
                       fontfamily=THEME["FONT"], labelpad=10)
        ax.set_zlabel("s − sᵇ  [Tick]", fontsize=11, color=THEME["TEXT_DIM"],
                       fontfamily=THEME["FONT"], labelpad=10)
        ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("Optimal Bid Spread", fontsize=14, fontweight="bold",
                      color=THEME["CYAN"], fontfamily=THEME["FONT"], pad=8)

        # ── RIGHT: 2-D panels ─────────────────────────────────────
        sl = slice(0, si + 1)
        ts = sim["t"][sl]
        yr = data["yr"]   # precomputed y-ranges

        # Panel 1 — Price & Quotes
        a1 = fig.add_subplot(gs[0, 1])
        _style_2d(a1)
        a1.plot(ts, sim["mid"][sl],  color=THEME["TEXT"],  lw=1.0, label="Fair Value")
        a1.plot(ts, sim["bid"][sl],  color=THEME["GREEN"], lw=0.8,
                ls="--", alpha=0.85, label="Bid")
        a1.plot(ts, sim["ask"][sl],  color=THEME["RED"],   lw=0.8,
                ls="--", alpha=0.85, label="Ask")
        a1.fill_between(ts, sim["bid"][sl], sim["ask"][sl],
                         color=THEME["CYAN"], alpha=0.04)
        a1.legend(loc="upper left", fontsize=8, facecolor=THEME["BG"],
                  edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
                  framealpha=0.85)
        a1.set_xlim(0, SIM["T"]); a1.set_ylim(*yr["price"])
        a1.set_title("Price & Quotes", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_SEC"], fontfamily=THEME["FONT"],
                      loc="left", pad=4)

        # Panel 2 — Inventory
        a2 = fig.add_subplot(gs[1, 1])
        _style_2d(a2)
        a2.plot(ts, sim["inv"][sl], color=THEME["BLUE"], lw=0.9)
        a2.fill_between(ts, 0, sim["inv"][sl], color=THEME["BLUE"], alpha=0.12)
        a2.axhline(0, color=THEME["SPINE"], lw=0.5, alpha=0.5)
        a2.set_xlim(0, SIM["T"]); a2.set_ylim(*yr["inv"])
        a2.set_title("Inventory", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_SEC"], fontfamily=THEME["FONT"],
                      loc="left", pad=4)

        # Panel 3 — Cash Flow
        a3 = fig.add_subplot(gs[2, 1])
        _style_2d(a3)
        a3.plot(ts, sim["cash"][sl], color=THEME["ORANGE"], lw=0.9)
        a3.fill_between(ts, 0, sim["cash"][sl], color=THEME["ORANGE"], alpha=0.10)
        a3.axhline(0, color=THEME["SPINE"], lw=0.5, alpha=0.5)
        a3.set_xlim(0, SIM["T"]); a3.set_ylim(*yr["cash"])
        a3.set_title("Cash Flow", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_SEC"], fontfamily=THEME["FONT"],
                      loc="left", pad=4)

        # Panel 4 — PnL
        a4 = fig.add_subplot(gs[3, 1])
        _style_2d(a4, xlabel=True)
        a4.plot(ts, sim["pnl"][sl], color=THEME["CYAN"], lw=1.1)
        a4.fill_between(ts, 0, sim["pnl"][sl],
                         where=np.array(sim["pnl"][sl]) >= 0,
                         color=THEME["CYAN"], alpha=0.12)
        a4.fill_between(ts, 0, sim["pnl"][sl],
                         where=np.array(sim["pnl"][sl]) < 0,
                         color=THEME["RED"], alpha=0.12)
        a4.axhline(0, color=THEME["SPINE"], lw=0.5, alpha=0.5)
        a4.set_xlim(0, SIM["T"]); a4.set_ylim(*yr["pnl"])
        a4.set_xlabel("Time [s]", fontsize=10, color=THEME["TEXT_DIM"],
                       fontfamily=THEME["FONT"])
        a4.set_title("PnL  (mark-to-market)", fontsize=11, fontweight="bold",
                      color=THEME["TEXT_SEC"], fontfamily=THEME["FONT"],
                      loc="left", pad=4)

        # ── Title bar ─────────────────────────────────────────────
        fig.text(0.50, 0.955,
                 "The Avellaneda\u2013Stoikov Model",
                 ha="center", va="center", fontsize=24, fontweight="bold",
                 color=THEME["TEXT"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.916,
                 r"$r\;=\;s \;-\; q \gamma \sigma^2 (T - t)$"
                 "          |          "
                 r"$\delta^* = \gamma \sigma^2 (T - t)"
                 r" + \frac{2}{\gamma} \ln(1 + \frac{\gamma}{k})$",
                 ha="center", va="center", fontsize=13,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.891,
                 r"$\sigma = 0.3$    $\gamma = 0.005$    $k = 0.3$"
                 r"    $A = 0.9$    $T = 600$ s",
                 ha="center", va="center", fontsize=10,
                 color="#555555", fontfamily=THEME["FONT"])

        # ── HUD ───────────────────────────────────────────────────
        fig.text(0.97, 0.875,
                 f"t = {ct:5.0f}s   q = {sim['inv'][si]:+4.0f}"
                 f"   PnL = {sim['pnl'][si]:+8.1f}",
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=THEME["CYAN"], fontfamily=THEME["FONT"], alpha=0.85)

        # ── Footer ────────────────────────────────────────────────
        fig.text(0.98, 0.012, "@quant.traderr",
                 ha="right", va="bottom", fontsize=10,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.55)

        fig.savefig(fp, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)

        if idx % 30 == 0:
            print(f"  [worker] frame {idx}/{total}")
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — COMPILATION
# ═══════════════════════════════════════════════════════════════════

def compile_video(temp_dir, out_path):
    frames = sorted(
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
        if f.endswith(".png")
    )
    if not frames:
        log("ERROR: no frames found"); return

    log(f"Compiling {len(frames)} frames -> {out_path}")
    w = imageio.get_writer(out_path, fps=CONFIG["FPS"], codec="libx264",
                           quality=None, macro_block_size=1,
                           output_params=["-b:v", "15000k",
                                          "-pix_fmt", "yuv420p"])
    for fp in frames:
        w.append_data(imageio.imread(fp))

    # hold last frame
    last = imageio.imread(frames[-1])
    for _ in range(CONFIG["FPS"] * CONFIG["HOLD_LAST_SEC"]):
        w.append_data(last)
    w.close()
    log(f"Video saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("Avellaneda-Stoikov Pipeline START")
    log("=" * 60)

    # 1 — Data
    log("Computing optimal bid-spread surface ...")
    surf = compute_surface()
    log(f"  Surface grid: {surf[2].shape}")

    log("Running market-making simulation ...")
    sim = run_simulation()
    log(f"  Steps: {len(sim['t'])}  |  Final PnL: {sim['pnl'][-1]:.2f}")

    # precompute y-ranges for stable axes
    pad_price = max((sim["mid"].max() - sim["mid"].min()) * 0.15, 1.0)
    all_p = np.concatenate([sim["mid"], sim["bid"][1:], sim["ask"][1:]])

    def _yr(arr, pad_frac=0.15, min_pad=1.0):
        lo, hi = arr.min(), arr.max()
        pad = max((hi - lo) * pad_frac, min_pad)
        return (lo - pad, hi + pad)

    yr = {
        "price": (all_p.min() - pad_price, all_p.max() + pad_price),
        "inv":   _yr(sim["inv"]),
        "cash":  _yr(sim["cash"]),
        "pnl":   _yr(sim["pnl"]),
    }

    # 2 — Render
    total = CONFIG["FPS"] * CONFIG["DURATION_SEC"]
    tmp   = CONFIG["TEMP_DIR"]
    os.makedirs(tmp, exist_ok=True)

    todo = [i for i in range(total)
            if not os.path.exists(os.path.join(tmp, f"frame_{i:04d}.png"))
            or os.path.getsize(os.path.join(tmp, f"frame_{i:04d}.png")) == 0]

    log(f"Frames: {total} total, {len(todo)} to render  "
        f"({CONFIG['N_WORKERS']} workers)")

    if todo:
        shared = dict(sim=sim, surf=surf, yr=yr, tmp=tmp, total=total)
        with Pool(CONFIG["N_WORKERS"], _init_worker, (shared,)) as pool:
            res = list(pool.imap_unordered(render_frame, todo, chunksize=2))
        ok = sum(bool(r) for r in res)
        log(f"  Rendered: {ok}/{len(todo)}")

    # 3 — Compile
    compile_video(tmp, CONFIG["OUTPUT_FILE"])

    elapsed = time.time() - t0
    log(f"Pipeline complete in {elapsed:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
