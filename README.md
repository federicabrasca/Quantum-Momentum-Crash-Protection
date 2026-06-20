# Crash-Protected Momentum in the Quantum-Computing Sector
**FMA 2026 Final Project — Advanced Momentum**

A cross-sectional 12-1 momentum strategy applied to a self-built **quantum / AI-technology** universe, refined
with a **volatility-targeting / crash-protection** overlay (Barroso–Santa-Clara 2015), and stress-tested on the
**Dec-2024 → Jan-2025 quantum momentum crash**.

## Files
| File | Description |
|---|---|
| `Quantum_Momentum.ipynb` | Presentation notebook with the main work. |
| `quantum_momentum.py` | Backtest engine: universe, data, signal, WML + long-only, vol-targeting, metrics. |
| `make_charts.py` | Generates the four presentation charts into `outputs/`. |
| `outputs/` | Charts + metrics/stress CSVs. |
| `data_cache/` | Cached daily prices (delete to force a fresh yfinance download). |

! First run downloads ~58 tickers from Yahoo Finance (needs internet), afterwards it uses `data_cache/`.


## Method (no look-ahead)
- **Signal** at month-end *t*: `M = P[t-1]/P[t-12] − 1` (12-1, skip-month), uses info only up to *t−1*.
- **Portfolios:** quintile sort → long top 20 % / short bottom 20 % (**WML**), also a realistic **long-only** book.
- **Vol-targeting:** `w_t = σ_target / σ̂_{t-1}`, σ̂ = trailing 126-day realised vol of the WML book, σ_target = 12 %, leverage ≤ 2× — the leverage for month *t+1* uses vol only through month-end *t*.
- **Benchmark:** QTUM (Defiance Quantum ETF) buy-and-hold, same metrics computed for it.

## Headline results (≈, full sample 2018-2026)
- Naive WML: Sharpe ≈ 0, **max drawdown −81 %**, vol 40 %.
- WML + vol-target: vol **13 %**, max drawdown **−35 %**.
- Quantum crash (Nov-2024→Mar-2025): naive **−66 %** vs crash-protected **−25 %**.
- QTUM benchmark: Sharpe ≈ **1.14** — beats every momentum variant on a risk-adjusted basis.