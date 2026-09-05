# Project State

## Current status

- Git repository initialized on the `main` branch.
- GitHub remote configured at `https://github.com/kstephens1/Research`.
- Python virtual environments are excluded from version control.
- `analyze_holding.py` provides a configurable VectorBT buy-and-hold CLI.
- `analyze_dmac.py` provides a configurable long-only dual moving average crossover CLI.
- `analyze_dmac_multi.py` compares up to 10 paired DMAC strategies in one vectorized run.
- `compare_strategies_by_symbol.py` performs train/test selection for any Yahoo Finance ticker, defaulting to SPY.
- Runtime dependencies are pinned, including Plotly 6.9.0 for VectorBT compatibility.
- All 47 offline tests pass and cover every analyzer plus the train/test comparison workflow.
- Live dated `BTC-USD` Yahoo Finance smoke tests pass for all analyzers.
- Local `notes.txt` research notes are excluded from version control.
- README guidance covers interactive DMAC experiments in Jupyter notebooks.
- All Jupyter notebooks and generated checkpoint directories are excluded from version control.

## Latest change

- Added the initial Yahoo Finance buy-and-hold analyzer, documentation, and tests.
- Normalized requested date boundaries to UTC to prevent off-by-one-day results.
- Verified calculations, dependencies, CLI behavior, and live market-data access.
- Added a 10-day/20-day DMAC strategy with configurable windows and trade reporting.
- Verified the official 2019 BTC-USD scenario with a 63.52% simulated return.
- Added paired multi-strategy DMAC analysis with console-table results and a 10-strategy cap.
- Verified the 2019 defaults at 84.72% for `(10,30)` and 54.34% for `(20,30)`.
- Added `notes.txt` to `.gitignore` so private research notes cannot be committed accidentally.
- Documented optional JupyterLab usage, DataFrame analysis, and notebook hygiene.
- Added `*.ipynb` to `.gitignore` so all local notebooks remain private.
- Added adjusted-price SPY comparison across DMAC, price-trend, momentum, and buy-and-hold.
- Documented the comparison methodology, CLI options, metrics, and notebook DataFrames.
- Live SPY validation selected DMAC 30/40, price/SMA 150, and 189-day momentum on 2017-2021 training data.
- On the untouched 2022-2026-09-05 period, price/SMA 150 returned 55.74% versus 71.18% for buy and hold.
- Generalized the comparison CLI and documentation for any single Yahoo Finance ticker.
- Annualized metrics now infer each ticker's observation frequency instead of assuming 252 sessions.
- Verified the generalized workflow with a live AAPL train/test comparison.
- Renamed the generalized comparison command to `compare_strategies_by_symbol.py`.
- Expanded README usage and notebook guidance for the renamed symbol comparison command.
