"""
Multifractal_Spectrum_Pipeline.py
=================================
Reel pipeline — Multifractal Detrended Fluctuation Analysis (MF-DFA) and the
singularity spectrum f(α) of a return series.

Hook: "Volatility isn't one number.  It's a spectrum."

Pipeline (Kantelhardt et al. 2002):
  1. Generate two contrasting series:
       (a) BTC-style:  fat-tailed multiplicative cascade (binomial cascade
           on returns) → broad multifractal spectrum.
       (b) SPY-style:  near-monofractal fractional Gaussian noise (H≈0.55).
  2. Build the profile  Y(i) = cumsum( r - mean(r) ).
  3. Partition into windows of size s ∈ [s_min, s_max], detrend with a local
     polynomial fit, compute the q-th order fluctuation:
        F_q(s) = ( <|Y - Y_fit|^q>_window )^{1/q}
     for a range of q ∈ [-5, 5].
  4. Generalized Hurst:  F_q(s) ~ s^{h(q)}  →  fit slope on log-log.
  5. Legendre transform:
        τ(q) = q · h(q) - 1
        α(q) = dτ/dq
        f(α) = q · α - τ(q)
  6. Render:
       Left   : 3D wireframe of  log F_q(s)  surface over (q, log s)
       Right  : h(q),  τ(q),  and the iconic f(α) parabola — BTC vs SPY overlay

Dependencies: numpy, matplotlib
"""

import os, sys, time, warnings
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION":   (1920, 1080),
    "OUTPUT_IMAGE": "Multifractal_Spectrum_Output.png",
    "OUTPUT_VIDEO": "Multifractal_Spectrum_Output.mp4",
    "FRAME_DIR":    "temp_mfdfa_frames",
    "LOG_FILE":     "mfdfa_pipeline.log",
    "FPS":          30,
    "N_JOBS":       6,
}

MF = {
    "n":          2**14,                               # series length
    "q_values":   np.linspace(-5.0, 5.0, 41),
    "s_min":      16,
    "s_max":      1024,
    "n_scales":   24,
    "poly_order": 2,
    "cascade_levels": 14,                              # 2^14 = n
    "cascade_p":  0.65,                                # asymmetry → multifractal
    "fGn_H":      0.55,                                # near-monofractal SPY proxy
    "seed":       42,
}

THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "SPINE":      "#333333",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "ORANGE":     "#ff9500",
    "YELLOW":     "#ffd400",
    "CYAN":       "#00f2ff",
    "GREEN":      "#00ff7f",
    "RED":        "#ff3050",
    "PINK":       "#ff2a9e",
    "BLUE":       "#00bfff",
    "PALE":       "#88aaff",
    "FONT":       "Arial",
}


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
# MODULE 1 — SYNTHETIC SERIES
# ═══════════════════════════════════════════════════════════════════

def binomial_cascade(levels, p, rng):
    """Multiplicative binomial cascade — canonical multifractal."""
    arr = np.array([1.0])
    for _ in range(levels):
        mask = rng.random(arr.size) < 0.5
        left  = np.where(mask, p, 1 - p)
        right = 1 - left
        arr = np.empty(arr.size * 2)
        arr[0::2] = arr.size // 2 * 0  # placeholder, overwritten below
        # do it cleanly
        new = np.empty(arr.size)
        new[0::2] = (left  * 2.0)
        new[1::2] = (right * 2.0)
        arr = (np.repeat(arr.reshape(-1, 1) if False else np.array([1.0]),
                         1) if False else None)
        # simple version:
        break
    # clean re-implementation
    measure = np.array([1.0])
    for _ in range(levels):
        mask = rng.random(measure.size) < 0.5
        m_left  = measure * np.where(mask, 2*p, 2*(1-p))
        m_right = measure * np.where(mask, 2*(1-p), 2*p)
        out = np.empty(measure.size * 2)
        out[0::2] = m_left
        out[1::2] = m_right
        measure = out
    # return signed returns: sqrt(measure) * Gaussian
    z = rng.standard_normal(measure.size)
    return np.sqrt(measure) * z * 0.01


def fractional_gaussian_noise(n, H, rng):
    """Simple Davies-Harte-ish fGn via FFT spectral synthesis."""
    k = np.arange(1, n + 1)
    # power-law spectrum  S(f) ~ f^{-(2H-1)}
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid div by zero
    spec = freqs ** -(2*H - 1)
    phases = np.exp(1j * 2*np.pi * rng.random(spec.size))
    X = np.fft.irfft(np.sqrt(spec) * phases, n=n)
    X = (X - X.mean()) / X.std()
    return X * 0.01


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — MF-DFA
# ═══════════════════════════════════════════════════════════════════

def mfdfa(returns, q_values, scales, poly_order):
    """
    Returns:
      F[q_i, s_j] : fluctuation values
      h(q)        : generalized Hurst
      tau(q), alpha(q), f(alpha)
    """
    Y = np.cumsum(returns - returns.mean())
    N = Y.size
    F = np.zeros((len(q_values), len(scales)))

    for j, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 4:
            F[:, j] = np.nan
            continue
        segs = Y[:n_seg * s].reshape(n_seg, s)

        # detrend each segment
        x = np.arange(s)
        coeffs = np.polyfit(x, segs.T, poly_order)         # (p+1, n_seg)
        trends = np.polyval(coeffs, x[:, None]).T           # (n_seg, s)
        var = np.mean((segs - trends) ** 2, axis=1)        # (n_seg,)

        for i, q in enumerate(q_values):
            if abs(q) < 1e-6:
                # q -> 0 limit:  log F_0(s) = 0.5 * <log Var>
                F[i, j] = np.exp(0.5 * np.mean(np.log(var + 1e-30)))
            else:
                F[i, j] = (np.mean(var ** (q/2))) ** (1.0/q)

    log_s = np.log(scales)
    h = np.zeros(len(q_values))
    for i in range(len(q_values)):
        valid = np.isfinite(np.log(F[i] + 1e-30))
        h[i] = np.polyfit(log_s[valid], np.log(F[i][valid]), 1)[0]

    tau = q_values * h - 1.0
    alpha = np.gradient(tau, q_values)
    f_alpha = q_values * alpha - tau
    return F, h, tau, alpha, f_alpha


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — RENDER
# ═══════════════════════════════════════════════════════════════════

def render_static(scales, q_values, F_btc, h_btc, tau_btc, alpha_btc, f_btc,
                  F_spy, h_spy, tau_spy, alpha_spy, f_spy, out_path):
    res = CONFIG["RESOLUTION"]
    fig = plt.figure(figsize=(res[0]/100, res[1]/100), dpi=100,
                     facecolor=THEME["BG"])
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.4, 1.0],
                           hspace=0.45, wspace=0.18,
                           left=0.04, right=0.97, top=0.92, bottom=0.07)

    # ─── LEFT: 3D wireframe of log F_q(s) for BTC cascade ──
    ax3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor=THEME["PANEL_BG"])
    Q, S = np.meshgrid(q_values, np.log(scales), indexing="ij")
    Z = np.log(F_btc + 1e-30)
    ax3d.plot_wireframe(Q, S, Z, rstride=2, cstride=2,
                        color=THEME["ORANGE"], lw=0.7, alpha=0.85)
    ax3d.set_xlabel("q", color=THEME["TEXT_DIM"])
    ax3d.set_ylabel("log s", color=THEME["TEXT_DIM"])
    ax3d.set_zlabel("log F_q(s)", color=THEME["TEXT_DIM"])
    ax3d.set_title("MF-DFA fluctuation surface   F_q(s) ~ s^{h(q)}   (BTC cascade)",
                   color=THEME["TEXT"], fontsize=12, pad=14)
    ax3d.tick_params(colors=THEME["TEXT_DIM"])
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor(THEME["PANEL_BG"])
        pane.set_edgecolor(THEME["GRID"])

    # ─── RIGHT row 0: h(q) ──
    ax_h = fig.add_subplot(gs[0, 1], facecolor=THEME["PANEL_BG"])
    ax_h.plot(q_values, h_btc, color=THEME["ORANGE"], lw=1.6, label="BTC cascade")
    ax_h.plot(q_values, h_spy, color=THEME["CYAN"],   lw=1.6, label="SPY ≈ fGn")
    ax_h.axhline(0.5, color=THEME["TEXT_DIM"], lw=0.6, ls="--")
    ax_h.set_title("Generalized Hurst  h(q)", color=THEME["TEXT"], fontsize=11, loc="left")
    ax_h.set_xlabel("q", color=THEME["TEXT_DIM"])
    ax_h.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    ax_h.grid(True, color=THEME["GRID"], lw=0.4, alpha=0.5)
    for s in ax_h.spines.values(): s.set_color(THEME["SPINE"])
    ax_h.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                labelcolor=THEME["TEXT_DIM"], fontsize=8)

    # ─── RIGHT row 1: tau(q) ──
    ax_t = fig.add_subplot(gs[1, 1], facecolor=THEME["PANEL_BG"])
    ax_t.plot(q_values, tau_btc, color=THEME["ORANGE"], lw=1.6)
    ax_t.plot(q_values, tau_spy, color=THEME["CYAN"],   lw=1.6)
    ax_t.set_title("Mass exponent  τ(q) = q·h(q) − 1", color=THEME["TEXT"],
                   fontsize=11, loc="left")
    ax_t.set_xlabel("q", color=THEME["TEXT_DIM"])
    ax_t.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    ax_t.grid(True, color=THEME["GRID"], lw=0.4, alpha=0.5)
    for s in ax_t.spines.values(): s.set_color(THEME["SPINE"])

    # ─── RIGHT row 2: f(α) — the iconic spectrum ──
    ax_f = fig.add_subplot(gs[2, 1], facecolor=THEME["PANEL_BG"])
    ax_f.plot(alpha_btc, f_btc, color=THEME["ORANGE"], lw=2.0, label="BTC: wide")
    ax_f.plot(alpha_spy, f_spy, color=THEME["CYAN"],   lw=2.0, label="SPY: narrow")
    width_btc = alpha_btc.max() - alpha_btc.min()
    width_spy = alpha_spy.max() - alpha_spy.min()
    ax_f.set_title(f"Singularity spectrum  f(α)    "
                   f"Δα_BTC={width_btc:.2f}   Δα_SPY={width_spy:.2f}",
                   color=THEME["TEXT"], fontsize=11, loc="left")
    ax_f.set_xlabel("α", color=THEME["TEXT_DIM"])
    ax_f.set_ylabel("f(α)", color=THEME["TEXT_DIM"])
    ax_f.tick_params(colors=THEME["TEXT_DIM"], labelsize=8)
    ax_f.grid(True, color=THEME["GRID"], lw=0.4, alpha=0.5)
    for s in ax_f.spines.values(): s.set_color(THEME["SPINE"])
    ax_f.legend(facecolor=THEME["PANEL_BG"], edgecolor=THEME["SPINE"],
                labelcolor=THEME["TEXT_DIM"], fontsize=8)

    fig.suptitle("Multifractal Spectrum   ·   MF-DFA  →  f(α)",
                 color=THEME["TEXT"], fontsize=18, y=0.975)
    fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log("Multifractal Spectrum pipeline — start")
    rng = np.random.default_rng(MF["seed"])

    btc = binomial_cascade(MF["cascade_levels"], MF["cascade_p"], rng)
    spy = fractional_gaussian_noise(MF["n"], MF["fGn_H"], rng)

    scales = np.unique(np.logspace(np.log10(MF["s_min"]),
                                   np.log10(MF["s_max"]),
                                   MF["n_scales"]).astype(int))
    log(f"BTC n={btc.size}  SPY n={spy.size}  scales={len(scales)}  "
        f"q∈[{MF['q_values'].min():.1f},{MF['q_values'].max():.1f}]")

    F_b, h_b, tau_b, a_b, f_b = mfdfa(btc, MF["q_values"], scales, MF["poly_order"])
    F_s, h_s, tau_s, a_s, f_s = mfdfa(spy, MF["q_values"], scales, MF["poly_order"])
    log(f"BTC: Δα = {a_b.max()-a_b.min():.3f}   SPY: Δα = {a_s.max()-a_s.min():.3f}")

    render_static(scales, MF["q_values"],
                  F_b, h_b, tau_b, a_b, f_b,
                  F_s, h_s, tau_s, a_s, f_s,
                  CONFIG["OUTPUT_IMAGE"])
    log(f"Wrote static frame  →  {CONFIG['OUTPUT_IMAGE']}")
    log("Done.  (Video render: TODO — animate q sweep across the surface, "
        "save to FRAME_DIR, stitch via ffmpeg.)")


if __name__ == "__main__":
    main()
