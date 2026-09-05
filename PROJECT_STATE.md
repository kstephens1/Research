# Project State

## Current status

- Git repository initialized on the `main` branch.
- GitHub remote configured at `https://github.com/kstephens1/Research`.
- Python virtual environments are excluded from version control.
- `analyze_holding.py` provides a configurable VectorBT buy-and-hold CLI.
- Runtime dependencies are pinned, including Plotly 6.9.0 for VectorBT compatibility.
- All 11 offline unit tests pass locally.
- A live dated `BTC-USD` Yahoo Finance smoke test passes.

## Latest change

- Added the initial Yahoo Finance buy-and-hold analyzer, documentation, and tests.
- Normalized requested date boundaries to UTC to prevent off-by-one-day results.
- Verified calculations, dependencies, CLI behavior, and live market-data access.
