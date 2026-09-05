# Project State

## Current status

- Git repository initialized on the `main` branch.
- GitHub remote configured at `https://github.com/kstephens1/Research`.
- Python virtual environments are excluded from version control.
- `analyze_holding.py` provides a configurable VectorBT buy-and-hold CLI.
- `analyze_dmac.py` provides a configurable long-only dual moving average crossover CLI.
- `analyze_dmac_multi.py` compares up to 10 paired DMAC strategies in one vectorized run.
- Runtime dependencies are pinned, including Plotly 6.9.0 for VectorBT compatibility.
- All 35 offline tests pass across the three strategy analyzers.
- Live dated `BTC-USD` Yahoo Finance smoke tests pass for all analyzers.
- Local `notes.txt` research notes are excluded from version control.

## Latest change

- Added the initial Yahoo Finance buy-and-hold analyzer, documentation, and tests.
- Normalized requested date boundaries to UTC to prevent off-by-one-day results.
- Verified calculations, dependencies, CLI behavior, and live market-data access.
- Added a 10-day/20-day DMAC strategy with configurable windows and trade reporting.
- Verified the official 2019 BTC-USD scenario with a 63.52% simulated return.
- Added paired multi-strategy DMAC analysis with console-table results and a 10-strategy cap.
- Verified the 2019 defaults at 84.72% for `(10,30)` and 54.34% for `(20,30)`.
- Added `notes.txt` to `.gitignore` so private research notes cannot be committed accidentally.
