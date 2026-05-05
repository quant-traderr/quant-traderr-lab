"""
Lorenz_BTC_Reel_Pipeline.py
============================
Cinematic landscape reel: Lorenz-63 attractor driven by BTC realized volatility.
ρ(t) = f(BTC realized vol) — when BTC is calm, the attractor collapses to a
fixed point. When BTC is volatile, the iconic butterfly blooms.

The trajectory is integrated ONCE with time-varying ρ(t), so the path smoothly
warps through bifurcations as the market regime changes. Color = current ρ.
A small inset plots BTC price + vol synced to the trajectory time.

Pipeline: DATA -> SIM (time-varying rho) -> RENDER (parallel) -> COMPILE
Resolution: 1920x1080 @ 30 FPS, ~12s + 2s hold
"""

import os, time, shutil, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKER": "BTC-USD", "LOOKBACK_DAYS": 360,
    "STEPS_PER_DAY": 60, "DT": 0.005,
    "RHO_MIN": 14.0, "RHO_MAX": 32.0,
    "SIGMA": 10.0, "BETA": 8 / 3,
    "FPS": 30, "DURATION_SEC": 12, "HOLD_LAST_SEC": 2,
    "DPI": 300,
    "OUTPUT_FILE": os.path.join(BASE_DIR, "Lorenz_BTC_Static.png"),
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#888888",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": "Arial",
}

RHO_CMAP = LinearSegmentedColormap.from_list(
    "rho",
    ["#0066ff", "#00f2ff", "#ffd400", "#ff9500", "#ff3050", "#ffffff"],
    N=256,
)


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def fetch_data():
    log(f"[Data] Fetching {CONFIG['TICKER']}...")
    try:
        data = yf.download(CONFIG["TICKER"], period="2y",
                           interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG["TICKER"], axis=1, level=1)
        prices = data["Close"].values.flatten()
        dates = data.index
        n = CONFIG["LOOKBACK_DAYS"]
        return prices[-n - 1:], dates[-n - 1:]
    except Exception:
        n = CONFIG["LOOKBACK_DAYS"]
        prices = 30000 * np.exp(np.cumsum(np.random.normal(0, 0.04, n + 1)))
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n + 1, freq="D")
        return prices, dates


def build_rho_series(prices):
    log_returns = np.diff(np.log(prices)) * 100
    n = len(log_returns)
    win = 14
    vol = np.array([np.std(log_returns[max(0, i - win):i + 1]) for i in range(n)])
    # Smooth so rho doesn't jitter
    smooth = np.zeros_like(vol); smooth[0] = vol[0]
    a = 0.10
    for k in range(1, n):
        smooth[k] = a * vol[k] + (1 - a) * smooth[k - 1]
    v_lo, v_hi = np.percentile(smooth, 5), np.percentile(smooth, 95)
    norm = np.clip((smooth - v_lo) / (v_hi - v_lo + 1e-9), 0, 1)
    rho = CONFIG["RHO_MIN"] + norm * (CONFIG["RHO_MAX"] - CONFIG["RHO_MIN"])
    return rho, smooth


def integrate_lorenz(rho_per_day):
    sigma, beta, dt = CONFIG["SIGMA"], CONFIG["BETA"], CONFIG["DT"]
    n_days = len(rho_per_day)
    spd = CONFIG["STEPS_PER_DAY"]
    n = n_days * spd
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n); rho_path = np.zeros(n)
    xs[0], ys[0], zs[0] = 0.5, 0.5, 0.5
    for d in range(n_days):
        rho_d = rho_per_day[d]
        for s in range(spd):
            i = d * spd + s
            if i == 0: continue
            x, y, z = xs[i - 1], ys[i - 1], zs[i - 1]
            dx = sigma * (y - x)
            dy = x * (rho_d - z) - y
            dz = x * y - beta * z
            xs[i] = x + dx * dt
            ys[i] = y + dy * dt
            zs[i] = z + dz * dt
            rho_path[i] = rho_d
    rho_path[0] = rho_per_day[0]
    return xs, ys, zs, rho_path


_SH = {}
def _init(data):
    global _SH; _SH = data


def render_frame(idx):
    try:
        d = _SH
        xs, ys, zs, rho_path = d["xs"], d["ys"], d["zs"], d["rho_path"]
        prices, dates, vol = d["prices"], d["dates"], d["vol"]
        total = d["total"]
        x_lim, y_lim, z_lim = d["x_lim"], d["y_lim"], d["z_lim"]
        rho_lo, rho_hi = d["rho_lo"], d["rho_hi"]
        spd = CONFIG["STEPS_PER_DAY"]

        n_traj = len(xs)
        n_days = len(prices) - 1
        t = idx / max(total - 1, 1)

        reveal_t = min(t / 0.92, 1.0)
        n_vis = max(2, int(reveal_t * n_traj))
        cur_day = min(n_days - 1, int(reveal_t * n_days))

        # Slow camera rotation
        azim = -65 + 90 * t
        elev = 22 + 6 * np.sin(t * np.pi)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=CONFIG["DPI"], facecolor=THEME["BG"])

        # Title
        fig.text(0.5, 0.945, "BTC AS CHAOS", ha="center",
                 fontsize=28, fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
        fig.text(0.5, 0.905,
                 f"Lorenz-63 driven by BTC realized vol  ·  ρ(t) = f(σ_BTC)",
                 ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

        # Side hook
        fig.text(0.04, 0.78, "Most see prices.",
                 fontsize=15, color=THEME["TEXT_DIM"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.75, "Lorenz sees regimes.",
                 fontsize=15, color=THEME["TEXT"], style="italic", family=THEME["FONT"])
        fig.text(0.04, 0.69, "Calm BTC →  fixed point.",
                 fontsize=11, color=THEME["CYAN"], family=THEME["FONT"])
        fig.text(0.04, 0.665, "Volatile BTC →  butterfly.",
                 fontsize=11, color=THEME["RED"], family=THEME["FONT"])

        # 3D Lorenz attractor
        ax = fig.add_axes([0.24, 0.05, 0.55, 0.92], projection="3d", facecolor=THEME["BG"])

        x = xs[:n_vis]; y = ys[:n_vis]; z = zs[:n_vis]; rp = rho_path[:n_vis]
        if len(x) > 1:
            pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            norms = np.clip((rp[:-1] - rho_lo) / (rho_hi - rho_lo + 1e-9), 0, 1)
            colors = RHO_CMAP(norms)
            colors[:, -1] = 0.55
            lc = Line3DCollection(segments, colors=colors, linewidths=0.45)
            ax.add_collection3d(lc)

            # Glowing leading point
            cur_color = RHO_CMAP(np.clip((rp[-1] - rho_lo) / (rho_hi - rho_lo + 1e-9), 0, 1))
            ax.scatter([x[-1]], [y[-1]], [z[-1]], s=55,
                       color=cur_color, edgecolors="white", linewidths=0.6, zorder=10)

        ax.set_xlim(x_lim); ax.set_ylim(y_lim); ax.set_zlim(z_lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.95))

        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0); ax.zaxis.pane.set_alpha(0)
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

        # Right side: BTC mini panel
        ax_p = fig.add_axes([0.82, 0.55, 0.16, 0.30], facecolor="#0a0a0a")
        ax_p.plot(prices[1:cur_day + 2], color=THEME["CYAN"], linewidth=0.8)
        ax_p.set_xlim(0, n_days); ax_p.set_ylim(prices.min() * 0.95, prices.max() * 1.05)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        for s in ax_p.spines.values(): s.set_color("#222"); s.set_linewidth(0.5)
        ax_p.set_title("BTC price", color=THEME["TEXT_DIM"], fontsize=9, loc="left", pad=2)

        ax_v = fig.add_axes([0.82, 0.20, 0.16, 0.30], facecolor="#0a0a0a")
        ax_v.fill_between(range(cur_day + 1), 0, vol[:cur_day + 1],
                          color=THEME["ORANGE"], alpha=0.45, linewidth=0)
        ax_v.plot(vol[:cur_day + 1], color=THEME["ORANGE"], linewidth=0.8)
        ax_v.set_xlim(0, n_days); ax_v.set_ylim(0, vol.max() * 1.1)
        ax_v.set_xticks([]); ax_v.set_yticks([])
        for s in ax_v.spines.values(): s.set_color("#222"); s.set_linewidth(0.5)
        ax_v.set_title("realized vol", color=THEME["TEXT_DIM"], fontsize=9, loc="left", pad=2)

        # HUD
        cur_rho = rho_path[max(0, n_vis - 1)]
        cur_price = prices[cur_day + 1]
        cur_vol = vol[cur_day]
        cur_date = pd.to_datetime(dates[cur_day + 1]).strftime("%Y-%m-%d")

        fig.text(0.04, 0.50, "ρ (chaos parameter)", fontsize=10, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.465, f"{cur_rho:.2f}", fontsize=20, color=THEME["YELLOW"],
                 fontweight="bold")
        regime = "STABLE" if cur_rho < 24.74 else "CHAOTIC"
        regime_color = THEME["CYAN"] if cur_rho < 24.74 else THEME["RED"]
        fig.text(0.04, 0.435, regime, fontsize=11, color=regime_color, fontweight="bold")
        fig.text(0.04, 0.41, "Lorenz bifurcation at ρ ≈ 24.74",
                 fontsize=8, color=THEME["TEXT_DIM"], style="italic")

        fig.text(0.04, 0.345, "BTC", fontsize=10, color=THEME["TEXT_DIM"])
        fig.text(0.04, 0.32, f"{cur_date}", fontsize=10, color=THEME["TEXT"])
        fig.text(0.04, 0.295, f"${cur_price:,.0f}", fontsize=12, color=THEME["CYAN"],
                 fontweight="bold")
        fig.text(0.04, 0.275, f"σ = {cur_vol:.2f}%/day", fontsize=10,
                 color=THEME["ORANGE"])

        fig.text(0.04, 0.215, r"$\dot x = \sigma(y-x)$", fontsize=11, color=THEME["TEXT"])
        fig.text(0.04, 0.190, r"$\dot y = x(\rho-z) - y$", fontsize=11, color=THEME["TEXT"])
        fig.text(0.04, 0.165, r"$\dot z = xy - \beta z$", fontsize=11, color=THEME["TEXT"])

        fig.text(0.04, 0.115, "Same equations.",
                 fontsize=10, color=THEME["TEXT_DIM"], style="italic")
        fig.text(0.04, 0.092, "Different ρ.",
                 fontsize=10, color=THEME["TEXT_DIM"], style="italic")
        fig.text(0.04, 0.069, "Different markets.",
                 fontsize=10, color=THEME["TEXT"], style="italic")

        fig.text(0.985, 0.018, "@quant.traderr", ha="right", va="bottom",
                 fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

        fp = CONFIG["OUTPUT_FILE"]
        fig.savefig(fp, dpi=CONFIG["DPI"], facecolor=THEME["BG"])
        plt.close(fig)
        log(f"Static image saved to {fp}")
        return True
    except Exception:
        import traceback; traceback.print_exc(); return False

def main():
    t0 = time.time(); log("=== LORENZ-BTC REEL ===")
    prices, dates = fetch_data()
    rho, vol = build_rho_series(prices)
    log(f"BTC days: {len(rho)}  rho range: [{rho.min():.2f}, {rho.max():.2f}]")

    xs, ys, zs, rho_path = integrate_lorenz(rho)
    log(f"Lorenz steps: {len(xs)}")

    pad = 0.06
    x_lim = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs))
    y_lim = (ys.min() - pad * np.ptp(ys), ys.max() + pad * np.ptp(ys))
    z_lim = (0, zs.max() + pad * np.ptp(zs))

    total = CONFIG["FPS"] * CONFIG["DURATION_SEC"]

    shared = {
        "xs": xs, "ys": ys, "zs": zs, "rho_path": rho_path,
        "prices": prices, "dates": dates, "vol": vol,
        "total": total,
        "x_lim": x_lim, "y_lim": y_lim, "z_lim": z_lim,
        "rho_lo": float(rho.min()), "rho_hi": float(rho.max()),
    }

    _init(shared)
    log("Rendering final static image...")
    render_frame(total - 1)

    log(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
