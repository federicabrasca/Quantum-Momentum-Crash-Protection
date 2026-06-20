"""
What this script does
---------------------
1. Builds a self-built "quantum computing + enabling technology" universe
   (pure-plays + semiconductor / photonics / big-tech enablers), inspired by the
   index methodology of the Defiance Quantum ETF (QTUM, live since Sep 2018),
   which is also used as the buy-and-hold sector benchmark.
2. Implements a plain CROSS-SECTIONAL 12-1
   (skip-month) momentum strategy -> long the top quintile (winners),
   short the bottom quintile (losers): the Winner-Minus-Loser (WML) portfolio.
   A realistic LONG-ONLY top-quintile variant is also produced.
3. Adds the refinement: a VOLATILITY-TARGETED / crash-protected overlay in the
   spirit of Barroso & Santa-Clara (2015): scale gross exposure by
   w_t = sigma_target / sigma_hat_{t-1}, where sigma_hat is the trailing
   realised volatility of the WML portfolio (Daniel & Moskowitz 2016) -
   "dynamic risk management nearly doubles Sharpe".
4. Computes, for every strategy AND the benchmark, standard performance
   measures: annualised return, annualised volatility,
   maximum drawdown and the Sharpe ratio.
5. Produces charts and saves all outputs.

Design notes
---------------------------
* NO LOOK-AHEAD. The signal formed at month-end t uses prices up to t-1
  (skip month), and governs the portfolio HELD over month t+1. The
  vol-target leverage applied during month t+1 uses volatility estimated only
  with data available up to month-end t.
* Survivorship / selection bias: the universe is built from names known to be
  relevant TODAY, so it is forward-looking. Pure-plays only enter when listed
  (unbalanced panel). These limitations are disclosed in 'Quantum_Momentum.ipynb'.
"""

from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Configuration

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "data_cache")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DOWNLOAD_START = "2016-06-01"     # buffer for the 12-month lookback
BACKTEST_START = "2018-10-31"     # align with QTUM inception (Sep 2018)
LOOKBACK = 12                     # formation window (months)
SKIP = 1                          # skip most recent month (12-1 momentum)
N_QUANTILES = 5                   # quintiles
MIN_NAMES = 15                    # min eligible names to form quintiles
VOL_TARGET = 0.12                 # annualised target vol for WML (Barroso-Santa-Clara)
VOL_TARGET_LONGONLY = 0.25        # annualised target vol for the long-only portfolio (more aggressive, since no short leg)
VOL_LOOKBACK_D = 126              # ~6 months of trading days for realised vol
MAX_LEVERAGE = 2.0                # cap on vol-target leverage
RF_ANNUAL = 0.0                   # risk-free assumed 0 (disclosed); Sharpe = excess/vol
TRADING_DAYS = 252

BENCHMARK = "QTUM"                # Defiance Quantum ETF - sector benchmark

# Self-built quantum + enabling-technology universe (~60 names)
# Grouped only for documentation, the strategy treats them as one cross-section
UNIVERSE = {
    "pure_play_quantum": ["IONQ", "RGTI", "QBTS", "QUBT", "ARQQ", "LAES"],
    "semis_and_equipment": [
        "NVDA", "AMD", "INTC", "MU", "AVGO", "QCOM", "MRVL", "TXN", "ADI",
        "NXPI", "ON", "MCHP", "AMAT", "LRCX", "KLAC", "TER", "ASML", "STM",
        "MPWR", "SWKS", "QRVO", "LSCC", "MKSI", "ENTG", "ONTO", "FORM",
        "COHR", "IPGP",
    ],
    "big_tech_quantum_rd": ["IBM", "GOOGL", "MSFT", "AMZN", "HON", "AAPL", "META", "ORCL"],
    "networking_photonics": ["CSCO", "ANET", "JNPR", "CIEN", "LITE", "FN", "INFN", "NTGR"],
    "software_eda_ml": ["SNPS", "CDNS", "PLTR", "NOW", "CRM"],
    "comms_hardware": ["NOK", "ERIC"],
}
TICKERS = sorted({t for grp in UNIVERSE.values() for t in grp})



# 1. Data acquisition (with on-disk cache so re-runs are instant)

def download_prices() -> pd.DataFrame:
    """Daily adjusted-close prices for the universe + benchmark, cached to CSV."""
    cache = os.path.join(CACHE_DIR, "prices_daily.csv")
    if os.path.exists(cache):
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[data] loaded {px.shape[1]} series x {px.shape[0]} days from cache")
        return px

    import yfinance as yf
    syms = TICKERS + [BENCHMARK]
    print(f"[data] downloading {len(syms)} symbols from yfinance ...")
    raw = yf.download(syms, start=DOWNLOAD_START, auto_adjust=True,
                      progress=False, threads=True)["Close"]
    raw = raw.dropna(how="all").sort_index()
    raw.to_csv(cache)
    print(f"[data] downloaded {raw.shape[1]} series x {raw.shape[0]} days -> cached")
    return raw



# 2. Cross-sectional 12-1 momentum backtest (daily accounting, monthly rebalance)

def run_backtest(px: pd.DataFrame):
    bench_px = px[BENCHMARK].dropna()
    stocks_px = px[TICKERS].copy()

    # Monthly month-end prices and the 12-1 momentum signal
    m_px = stocks_px.resample("ME").last()
    signal = m_px.shift(SKIP) / m_px.shift(LOOKBACK) - 1.0   # known at each month-end t

    # Daily simple returns of every stock (for daily P&L accounting)
    d_ret = stocks_px.pct_change()

    form_dates = m_px.index[(m_px.index >= pd.Timestamp(BACKTEST_START))]
    daily_index = stocks_px.index

    wml_daily, long_daily = {}, {}     # date -> return
    n_long_hist, leg_records = {}, []  # diagnostics

    for i in range(len(form_dates) - 1):
        f = form_dates[i]
        nxt = form_dates[i + 1]
        sig = signal.loc[f].dropna()
        # eligible = has a 12-1 signal AND a tradeable price at formation
        price_now = m_px.loc[f]
        sig = sig[price_now.reindex(sig.index).notna()]
        if len(sig) < MIN_NAMES:
            continue

        q = max(1, len(sig) // N_QUANTILES)
        ranked = sig.sort_values()
        losers = ranked.index[:q]               # bottom quintile
        winners = ranked.index[-q:]             # top quintile
        n_long_hist[f] = len(winners)
        leg_records.append({"date": f, "n_eligible": len(sig),
                            "winners": list(winners), "losers": list(losers)})

        hold_days = daily_index[(daily_index > f) & (daily_index <= nxt)]
        for d in hold_days:
            w_ret = d_ret.loc[d, winners].dropna()
            l_ret = d_ret.loc[d, losers].dropna()
            if len(w_ret) == 0:
                continue
            long_leg = w_ret.mean()
            short_leg = l_ret.mean() if len(l_ret) else 0.0
            long_daily[d] = long_leg
            wml_daily[d] = long_leg - short_leg

    wml = pd.Series(wml_daily).sort_index()        # gross (full-exposure) WML
    longonly = pd.Series(long_daily).sort_index()  # long top-quintile only

    # ---- Vol-targeting overlay (Barroso-Santa-Clara)
    # Reusable: scale a gross daily-return series by leverage L_f fixed at each
    # formation date f, where L_f = target / sigma_hat_f, capped. sigma_hat is the
    # trailing ~126-day realised vol (lagged 1 day) -> uses only past information.
    def apply_vol_target(gross: pd.Series, target: float):
        sigma_hat = gross.rolling(VOL_LOOKBACK_D).std().shift(1) * np.sqrt(TRADING_DAYS)
        leverage = (target / sigma_hat).clip(upper=MAX_LEVERAGE)
        lev_for_day = pd.Series(index=gross.index, dtype=float)
        for i in range(len(form_dates) - 1):
            f, nxt = form_dates[i], form_dates[i + 1]
            lev_f = leverage.loc[:f].dropna()
            if lev_f.empty:
                continue
            L = float(lev_f.iloc[-1])
            mask = (gross.index > f) & (gross.index <= nxt)
            lev_for_day.loc[gross.index[mask]] = L
        return (gross * lev_for_day).dropna(), lev_for_day.dropna(), sigma_hat.dropna()

    voltgt, lev_wml, sigma_hat_wml = apply_vol_target(wml, VOL_TARGET)
    lo_voltgt, lev_lo, _ = apply_vol_target(longonly, VOL_TARGET_LONGONLY)

    # ---- Benchmark daily returns over the common window
    bench_ret = bench_px.pct_change().reindex(wml.index).dropna()

    results = {
        "WML (long-short, gross)": wml,
        "WML + Vol-target": voltgt,
        "Long-only (top quintile)": longonly,
        "Long-only + Vol-target": lo_voltgt,
        f"Benchmark ({BENCHMARK})": bench_ret,
    }
    diagnostics = {
        "leverage_wml": lev_wml,
        "leverage_longonly": lev_lo,
        "sigma_hat": sigma_hat_wml,
        "leg_records": pd.DataFrame(leg_records).set_index("date") if leg_records else pd.DataFrame(),
        "monthly_signal": signal,
    }
    return results, diagnostics



# 3. Performance metrics

def equity_curve(daily_ret: pd.Series) -> pd.Series:
    return (1.0 + daily_ret.fillna(0)).cumprod()

def perf_metrics(daily_ret: pd.Series) -> dict:
    r = daily_ret.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["CAGR", "Ann.Vol", "Sharpe", "MaxDD", "n_days"]}
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (TRADING_DAYS / n) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    rf_daily = RF_ANNUAL / TRADING_DAYS
    sharpe = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0).min()
    return {"CAGR": cagr, "Ann.Vol": ann_vol, "Sharpe": sharpe, "MaxDD": dd, "n_days": n}

def metrics_table(results: dict, common_window: bool = False) -> pd.DataFrame:
    res = results
    if common_window:
        start = max(r.dropna().index.min() for r in results.values())
        end = min(r.dropna().index.max() for r in results.values())
        res = {k: v.loc[start:end] for k, v in results.items()}
    rows = {name: perf_metrics(r) for name, r in res.items()}
    df = pd.DataFrame(rows).T
    df = df[["CAGR", "Ann.Vol", "Sharpe", "MaxDD", "n_days"]]
    return df

def max_dd_date(daily_ret: pd.Series):
    eq = (1 + daily_ret.dropna()).cumprod()
    underwater = eq / eq.cummax() - 1.0
    return underwater.idxmin(), underwater.min()

def window_return(daily_ret: pd.Series, start: str, end: str) -> float:
    r = daily_ret.loc[start:end].dropna()
    return (1 + r).prod() - 1 if len(r) else np.nan


if __name__ == "__main__":
    px = download_prices()
    results, diag = run_backtest(px)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    table = metrics_table(results)
    print("\n=== Full-sample performance (since {}) ===".format(BACKTEST_START))
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(table)

    table_cw = metrics_table(results, common_window=True)
    print("\n=== Performance on the COMMON window (fair comparison) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(table_cw)

    print("\n=== Max-drawdown timing ===")
    for name, r in results.items():
        d, v = max_dd_date(r)
        print(f"  {name:<28s} worst {v:7.1%} on {d.date()}")

    # Crash-window stress test (the quantum momentum crash + 2020/2022)
    print("\n=== Drawdown / return over stress windows ===")
    windows = {
        "Quantum crash 2024-11-01..2025-03-31": ("2024-11-01", "2025-03-31"),
        "COVID 2020-02-19..2020-03-23": ("2020-02-19", "2020-03-23"),
        "2022 tech bear 2022-01-01..2022-12-31": ("2022-01-01", "2022-12-31"),
    }
    stress = {}
    for label, (a, b) in windows.items():
        stress[label] = {name: window_return(r, a, b) for name, r in results.items()}
    stress_df = pd.DataFrame(stress).T
    with pd.option_context("display.float_format", lambda x: f"{x:,.2%}", "display.width", 220):
        print(stress_df.to_string())

    # Persist outputs
    table.to_csv(os.path.join(OUT_DIR, "metrics_full_sample.csv"))
    stress_df.to_csv(os.path.join(OUT_DIR, "stress_windows.csv"))
    eqs = pd.DataFrame({name: equity_curve(r) for name, r in results.items()})
    eqs.to_csv(os.path.join(OUT_DIR, "equity_curves.csv"))
    table_cw.to_csv(os.path.join(OUT_DIR, "metrics_common_window.csv"))
    diag["leverage_wml"].to_csv(os.path.join(OUT_DIR, "leverage_wml.csv"))
    diag["sigma_hat"].to_csv(os.path.join(OUT_DIR, "sigma_hat.csv"))
    print(f"\n[done] outputs written to {OUT_DIR}")