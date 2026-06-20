"""
Presentation charts for the Quantum Momentum project.
Re-runs the backtest from cached prices (fast) and saves PNGs to ./outputs.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import quantum_momentum as qm

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
OUT = qm.OUT_DIR

# Colour map for the five strategy lines
COL = {
    "WML (long-short, gross)": "#d62728",
    "WML + Vol-target": "#1f77b4",
    "Long-only (top quintile)": "#ff7f0e",
    "Long-only + Vol-target": "#2ca02c",
    "Benchmark (QTUM)": "#7f7f7f",
}
CRASHES = [
    ("2020-02-19", "2020-04-01", "COVID crash"),
    ("2022-01-01", "2022-12-31", "2022 tech bear"),
    ("2024-12-09", "2025-03-31", "Quantum crash"),
]


def shade_crashes(ax):
    for a, b, label in CRASHES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.10)
        ax.text(pd.Timestamp(a), ax.get_ylim()[1], " " + label,
                fontsize=8, color="dimgrey", va="top", ha="left")


def chart_equity(results):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, r in results.items():
        eq = qm.equity_curve(r)
        ax.plot(eq.index, eq.values, label=name, color=COL.get(name), lw=1.8,
                alpha=0.95 if "Vol-target" in name or "Benchmark" in name else 0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title("Quantum/AI momentum: crash-protected vs naive  (2018-2026)")
    shade_crashes(ax)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(os.path.join(OUT, "fig1_equity_curves.png"))
    plt.close(fig)


def chart_drawdown(results):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name in ["WML (long-short, gross)", "WML + Vol-target"]:
        eq = qm.equity_curve(results[name])
        dd = eq / eq.cummax() - 1.0
        ax.fill_between(dd.index, dd.values, 0, color=COL[name], alpha=0.25)
        ax.plot(dd.index, dd.values, color=COL[name], lw=1.5,
                label=f"{name}  (min {dd.min():.0%})")
    ax.set_ylabel("Drawdown")
    ax.set_title("Vol-targeting tames the momentum crash  (max drawdown: -81% -> -35%)")
    ax.legend(loc="lower left", fontsize=9)
    shade_crashes(ax)
    fig.savefig(os.path.join(OUT, "fig2_drawdown.png"))
    plt.close(fig)


def chart_vol_leverage(diag):
    sigma = diag["sigma_hat"]
    lev = diag["leverage_wml"]
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    ax1.plot(sigma.index, sigma.values, color="#d62728", lw=1.5,
             label="Realised vol of WML (126d, annualised)")
    ax1.axhline(qm.VOL_TARGET, color="#1f77b4", ls="--", lw=1.2,
                label=f"Vol target = {qm.VOL_TARGET:.0%}")
    ax1.set_ylabel("Annualised volatility")
    ax1.set_ylim(0, min(2.0, sigma.max() * 1.05))
    ax2 = ax1.twinx()
    ax2.plot(lev.index, lev.values, color="#2ca02c", lw=1.2, alpha=0.7,
             label="Vol-target leverage (right)")
    ax2.set_ylabel("Leverage applied")
    ax2.grid(False)
    ax1.set_title("The mechanism: leverage falls exactly when volatility spikes")
    shade_crashes(ax1)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig3_vol_leverage.png"))
    plt.close(fig)


def chart_crash_zoom(results, px):
    a, b = "2024-11-01", "2025-04-30"
    huang = pd.Timestamp("2025-01-08")
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(11, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]})

    # --- Top: the underlying sector boom & bust (context / the "why") ---
    pure = [t for t in qm.UNIVERSE["pure_play_quantum"] if t in px.columns]
    pp = px[pure].pct_change().loc[a:b].mean(axis=1)
    pp_eq = (1 + pp.fillna(0)).cumprod(); pp_eq /= pp_eq.iloc[0]
    axT.plot(pp_eq.index, pp_eq.values, color="purple", lw=1.8, ls=":",
             label="Pure-play quantum basket (equal-weight)")
    axT.fill_between(pp_eq.index, pp_eq.values, 1.0, color="purple", alpha=0.08)
    axT.set_ylabel("Growth of 1\n(rebased Nov 2024)")
    axT.set_title("The quantum momentum crash, up close")
    axT.legend(loc="upper left", fontsize=9)
    axT.annotate(f"Pure-plays ~{pp_eq.max():.0f}x\n(Rigetti +1,654%/yr,\nGoogle 'Willow')",
                 xy=(pp_eq.idxmax(), pp_eq.max()*0.85), fontsize=8, color="purple", ha="center")

    # --- Bottom: what it did to the momentum strategies (consistent rebasing) ---
    eqs = {}
    for name in ["WML (long-short, gross)", "WML + Vol-target", "Benchmark (QTUM)"]:
        r = results[name].loc[a:b]
        eq = (1 + r).cumprod(); eq /= eq.iloc[0]
        eqs[name] = eq
        axB.plot(eq.index, eq.values, label=name, color=COL[name], lw=2.0)
    axB.axhline(1.0, color="black", lw=0.6, alpha=0.4)
    # Headline numbers computed from the data at the WML trough (same date for both)
    wml_eq = eqs["WML (long-short, gross)"]
    trough = wml_eq.idxmin()
    wml_dd = wml_eq.min() - 1.0
    vt_at = eqs["WML + Vol-target"].loc[trough] - 1.0
    axB.set_ylabel("Growth of 1\n(rebased Nov 2024)")
    axB.legend(loc="lower left", fontsize=9)
    axB.set_title(f"At the trough ({trough.date()}): naive WML {wml_dd:.0%} vs crash-protected {vt_at:.0%}")
    # Two-phase annotation
    axB.axvspan(pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31"), color="orange", alpha=0.08)
    axB.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31"), color="red", alpha=0.06)
    yb = axB.get_ylim()[0]
    axB.text(pd.Timestamp("2024-11-15"), yb + 0.05, "Phase 1:\nshort-loser squeeze", fontsize=7.5, color="darkorange")
    axB.text(pd.Timestamp("2025-02-01"), yb + 0.05, "Phase 2:\nlong-winner reversal", fontsize=7.5, color="firebrick")

    for ax in (axT, axB):
        ax.axvline(huang, color="black", ls="--", lw=1.0)
    axT.text(huang, axT.get_ylim()[1]*0.99, "  Jan 8 2025: Huang -> quantum '15-30 yrs' away",
             fontsize=8, va="top", ha="left")
    axB.xaxis.set_major_locator(mdates.MonthLocator())
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.savefig(os.path.join(OUT, "fig4_quantum_crash_zoom.png"))
    plt.close(fig)


def main():
    px = qm.download_prices()
    results, diag = qm.run_backtest(px)
    chart_equity(results)
    chart_drawdown(results)
    chart_vol_leverage(diag)
    chart_crash_zoom(results, px)
    print("[charts] wrote fig1..fig4 to", OUT)


if __name__ == "__main__":
    main()