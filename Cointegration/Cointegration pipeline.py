"""
Cointegration - static title frame for the Demiurge carousel.

Builds two cointegrated price series (each an I(1) random walk that
share a common stochastic trend) and their stationary, mean-reverting
spread, drawn in the dark neon house style. The ADF unit-root t-stat is
computed live so the on-frame verdict is honest, not hardcoded.

Output: cointegration_static.png  (1760 x 990, dark)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, font_manager as fm

# ── Synthetic cointegrated pair ───────────────────────────────────────
rng = np.random.default_rng(0)
N = 500
BETA_TRUE = 1.3

x = np.cumsum(rng.standard_normal(N)) + 50.0      # I(1) random walk

# Stationary AR(1) spread (phi < 1) -> visible mean-reverting excursions
phi, e, s = 0.92, rng.standard_normal(N), np.zeros(N)
for i in range(1, N):
    s[i] = phi * s[i - 1] + e[i]
s *= 1.5

y = BETA_TRUE * x + s                              # y and x co-integrate

# ── Recover the hedge ratio and test the spread for stationarity ──────
beta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
spread = y - beta * x
spread -= spread.mean()
sigma = spread.std()

dz, z1 = np.diff(spread), spread[:-1]
g = (z1 @ dz) / (z1 @ z1)
resid = dz - g * z1
se = np.sqrt((resid @ resid) / (len(dz) - 1) / (z1 @ z1))
tstat = g / se
verdict = "COINTEGRATED" if tstat < -2.86 else "NOT COINTEGRATED"

# ── House fonts (Space Grotesk + JetBrains Mono) and palette ──────────
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
gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 1.0], hspace=0.16,
                       left=0.07, right=0.95, top=0.80, bottom=0.12)
t = np.arange(N)

# Top panel: the two cointegrated legs tracking each other
ax0 = fig.add_subplot(gs[0]); ax0.set_facecolor(BG)
ax0.plot(t, y, color=CYAN, lw=1.4, label="Y")
ax0.plot(t, beta * x, color=MAGENTA, lw=1.4, alpha=0.9, label=r"$\beta\,X$")
leg = ax0.legend(loc="upper left", frameon=False, labelcolor="white",
                 fontsize=14, prop=JB)
ax0.set_ylabel("PRICE LEVEL", color=DIM, fontsize=12, labelpad=8,
               fontproperties=SG)

# Bottom panel: the stationary spread with +/- 2 sigma bands
ax1 = fig.add_subplot(gs[1]); ax1.set_facecolor(BG)
ax1.plot(t, spread, color=GOLD, lw=1.1)
ax1.axhline(0, color="white", lw=0.6, alpha=0.4)
for k in (2, -2):
    ax1.axhline(k * sigma, color=DIM, lw=0.8, ls="--", alpha=0.6)
ax1.fill_between(t, spread, 0, where=np.abs(spread) > 2 * sigma,
                 color=MAGENTA, alpha=0.55)
ax1.set_ylabel("SPREAD  z", color=DIM, fontsize=12, labelpad=8,
               fontproperties=SG)

for ax in (ax0, ax1):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(0, N)
    ax.tick_params(colors=DIM, labelsize=9, length=0)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(JB)
    ax.grid(True, color="white", lw=0.4, alpha=0.06)
    ax.set_xticklabels([])

# ── HUD + equation watermark ──────────────────────────────────────────
fig.text(0.07, 0.92, "PAIR: SYNTHETIC  Y / X", color="white",
         fontsize=15, fontproperties=JB_B)
fig.text(0.07, 0.875,
         f"ENGLE-GRANGER    ADF t = {tstat:.1f}    {verdict}",
         color=CYAN, fontsize=12.5, fontproperties=JB)
fig.text(0.95, 0.92, "STAT-ARB", color=DIM, fontsize=12.5, alpha=0.9,
         ha="right", fontproperties=JB)
fig.text(0.5, 0.045, r"$z_t = y_t - \beta\,x_t \sim I(0)$",
         color="white", fontsize=19, ha="center", alpha=0.9)

fig.savefig("cointegration_static.png", facecolor=BG)
print(f"beta={beta:.3f}  ADF t-stat={tstat:.2f}  verdict={verdict}")
print("saved cointegration_static.png")
