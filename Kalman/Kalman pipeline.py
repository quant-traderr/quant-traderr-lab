"""
Kalman Filter - static title frame for the Demiurge carousel.

Loads REAL price data for a user-specified pair from yfinance and runs a
1-D Kalman filter to track the time-varying hedge ratio beta_t between
them. The frame contrasts the frozen full-sample OLS beta (the static
hedge ratio everyone else uses) against the dynamic Kalman estimate that
updates every bar, in the dark neon house style.

Usage:
    python kalman_static.py                       # defaults: KO / PEP
    python kalman_static.py KO PEP --start 2021-01-01
    python kalman_static.py GLD GDX --delta 1e-4 --out gld_gdx.png

Output: kalman_static.png  (or --out)
"""
import os
import argparse
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import gridspec, font_manager as fm

# ── User input ────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Kalman dynamic hedge-ratio frame")
p.add_argument("y_ticker", nargs="?", default="KO", help="dependent leg Y")
p.add_argument("x_ticker", nargs="?", default="PEP", help="independent leg X")
p.add_argument("--start", default="2021-01-01")
p.add_argument("--end", default=None)
p.add_argument("--delta", type=float, default=1e-4,
               help="state-drift / measurement-noise ratio (Q/R)")
p.add_argument("--out", default="kalman_static.png")
args = p.parse_args()
Y, X = args.y_ticker.upper(), args.x_ticker.upper()

# ── Load real prices ──────────────────────────────────────────────────
raw = yf.download([Y, X], start=args.start, end=args.end,
                  auto_adjust=True, progress=False)["Close"]
data = raw[[Y, X]].dropna()
if len(data) < 60:
    raise SystemExit(f"Not enough overlapping data for {Y}/{X} "
                     f"({len(data)} rows). Check tickers / date range.")
y = data[Y].to_numpy()
x = data[X].to_numpy()
N = len(y)
d0, d1 = data.index[0].date(), data.index[-1].date()

# ── 1-D Kalman filter: hidden state is the hedge ratio beta ───────────
static_beta = (x @ y) / (x @ x)             # frozen full-sample OLS slope
R = np.var(y - static_beta * x)             # measurement-noise scale
Q = R * args.delta                          # slow state drift

w = 30                                       # warmup window to seed beta
beta = (x[:w] @ y[:w]) / (x[:w] @ x[:w])
P = 1.0
est = np.zeros(N)
for t in range(N):
    P += Q                                   # predict
    K = P * x[t] / (x[t] ** 2 * P + R)        # Kalman gain
    beta += K * (y[t] - beta * x[t])          # update
    P *= 1 - K * x[t]
    est[t] = beta

# ── House fonts (Space Grotesk + JetBrains Mono) and palette ─────────
FONTS = os.path.join(os.path.dirname(__file__), "..", "Demiurge", "fonts")
for _f in ("SpaceGrotesk-Variable.ttf", "JetBrainsMono-Regular.ttf",
           "JetBrainsMono-Bold.ttf"):
    fm.fontManager.addfont(os.path.join(FONTS, _f))
SG = fm.FontProperties(fname=os.path.join(FONTS, "SpaceGrotesk-Variable.ttf"))
JB = fm.FontProperties(fname=os.path.join(FONTS, "JetBrainsMono-Regular.ttf"))
JB_B = fm.FontProperties(fname=os.path.join(FONTS, "JetBrainsMono-Bold.ttf"))
plt.rcParams["font.family"] = "Space Grotesk"
plt.rcParams["mathtext.fontset"] = "cm"

BG, CYAN, MAGENTA = "#0a0a0f", "#00f0ff", "#ff0055"
GOLD, DIM = "#f0c8a0", "#8a8a99"

fig = plt.figure(figsize=(16, 9), dpi=110)
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.5], hspace=0.16,
                       left=0.07, right=0.95, top=0.80, bottom=0.12)
t = np.arange(N)

# Top panel: the two real asset prices
ax0 = fig.add_subplot(gs[0]); ax0.set_facecolor(BG)
ax0.plot(t, y, color=CYAN, lw=1.4, label=Y)
ax0.plot(t, x, color=MAGENTA, lw=1.4, alpha=0.9, label=X)
ax0.legend(loc="upper left", frameon=False, labelcolor="white",
           fontsize=14, prop=JB)
ax0.set_ylabel("PRICE  (USD)", color=DIM, fontsize=12, labelpad=8,
               fontproperties=SG)

# Bottom panel: frozen OLS beta vs dynamic Kalman beta
ax1 = fig.add_subplot(gs[1]); ax1.set_facecolor(BG)
ax1.scatter(t, y / x, s=6, color=CYAN, alpha=0.14, label=r"$y/x$ (noisy)")
ax1.axhline(static_beta, color="white", lw=1.5, ls="--", alpha=0.75,
            label=r"static OLS $\beta$")
ax1.plot(t, est, color=GOLD, lw=2.2, label=r"Kalman $\hat{\beta}_t$")
ax1.legend(loc="upper right", frameon=False, labelcolor="white",
           fontsize=11, prop=JB, ncol=3)
ax1.set_ylabel(r"HEDGE RATIO  $\beta$", color=DIM, fontsize=12,
               labelpad=8, fontproperties=SG)

for ax in (ax0, ax1):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, N)
    ax.tick_params(colors=DIM, labelsize=9, length=0)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(JB)
    ax.grid(True, color="white", lw=0.4, alpha=0.06)
    ax.set_xticklabels([])

# ── HUD + equation watermark ─────────────────────────────────────────
fig.text(0.07, 0.92, f"PAIR: {Y} / {X}", color="white",
         fontsize=15, fontproperties=JB_B)
fig.text(0.07, 0.875,
         f"KALMAN FILTER   delta = {args.delta:.0e}   n = {N}   "
         f"{d0} .. {d1}",
         color=CYAN, fontsize=12, fontproperties=JB)
fig.text(0.95, 0.92,
         f"static b = {static_beta:.3f}   kalman b = {est[-1]:.3f}",
         color=DIM, fontsize=12, alpha=0.9, ha="right", fontproperties=JB)
fig.text(0.5, 0.045,
         r"$\hat{x}_k = \hat{x}_k^- + K_k\,(z_k - H\hat{x}_k^-)$",
         color="white", fontsize=19, ha="center", alpha=0.9)

fig.savefig(args.out, facecolor=BG)
print(f"{Y}/{X}  n={N}  {d0}..{d1}")
print(f"static beta={static_beta:.3f}  kalman beta (latest)={est[-1]:.3f}")
print(f"saved {args.out}")
