# VectorBT Trading Research

Command-line tools for evaluating Yahoo Finance assets with
[VectorBT](https://vectorbt.dev/).

## Setup

Yahoo Finance does not require an account or API key for this example.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Plotly version is pinned below 7 because VectorBT 1.1.0 still references a
trace type removed in Plotly 7.

## Buy and hold

Run the default analysis using all available daily `BTC-USD` history and an
initial balance of 100:

```bash
python analyze_holding.py
```

Analyze another ticker over a fixed period:

```bash
python analyze_holding.py AAPL --init-cash 1000 \
  --start 2020-01-01 --end 2025-01-01
```

Include a transaction fee expressed in percentage points. For example, `0.1`
means `0.1%`:

```bash
python analyze_holding.py BTC-USD --fee-pct 0.1
```

Use `python analyze_holding.py --help` to see every option. Yahoo treats the
end date as exclusive.

## Dual moving average crossover

The DMAC strategy enters a long position when the fast simple moving average
crosses above the slow average. It exits to cash when the fast average crosses
below the slow average. The defaults reproduce VectorBT's 10-day/20-day
example:

```bash
python analyze_dmac.py
```

Choose a ticker, date range, moving-average windows, starting balance, and
transaction fee from the command line:

```bash
python analyze_dmac.py BTC-USD \
  --fast-window 10 --slow-window 20 \
  --init-cash 100 --fee-pct 0.1 \
  --start 2019-01-01 --end 2020-01-01
```

Fees are charged on every executed buy and sell. If a position remains open at
the end of the data, its value is marked using the final close rather than
forcing a sale. Use `python analyze_dmac.py --help` to see every option.

## Multiple DMAC strategies

Run up to ten moving-average pairs in one vectorized simulation. Fast and slow
windows are paired by their positions in the two lists. The default pairs are
`(10, 30)` and `(20, 30)`, matching VectorBT's multi-strategy example:

```bash
python analyze_dmac_multi.py BTC-USD \
  --fast-windows 10 20 \
  --slow-windows 30 30 \
  --start 2019-01-01 --end 2020-01-01
```

Each strategy receives its own copy of the initial cash and is evaluated
independently. The command prints one console-table row per strategy with its
signals, trades, final value, profit, and return. Duplicate pairs, mismatched
list lengths, and more than ten strategies are rejected.

## Jupyter notebooks

Jupyter is useful for running several parameter combinations, comparing the
results as a DataFrame, and keeping Markdown notes beside each experiment. It
is optional and is not included in the runtime requirements. Install and start
JupyterLab inside the project environment with:

```bash
source .venv/bin/activate
python -m pip install jupyterlab
python -m jupyter lab
```

From a notebook, run the command-line script directly and display its console
table:

```python
%run analyze_dmac_multi.py BTC-USD \
    --fast-windows 5 10 20 \
    --slow-windows 20 30 50 \
    --init-cash 1000 \
    --fee-pct 0.1 \
    --start 2020-01-01 \
    --end 2025-01-01
```

For further filtering and comparison, import the analysis functions instead:

```python
from datetime import date

import pandas as pd

from analyze_dmac_multi import analyze_dmac_multi
from analyze_holding import download_close

prices = download_close(
    "BTC-USD",
    start=date(2020, 1, 1),
    end=date(2025, 1, 1),
)
results = analyze_dmac_multi(
    prices,
    fast_windows=[5, 10, 20],
    slow_windows=[20, 30, 50],
    init_cash=1000,
    fee_pct=0.1,
)

results_df = pd.DataFrame([vars(result) for result in results])
results_df.sort_values("total_return", ascending=False)
```

Use Markdown cells for conclusions and follow-up ideas. All `.ipynb` files and
generated `.ipynb_checkpoints/` directories are ignored so private research
notes and cell outputs are not committed accidentally.

## Compare strategies by symbol

Compare 19 curated long-only timing strategies for any single Yahoo Finance
ticker. `SPY` remains the default:

```bash
python compare_strategies_by_symbol.py
python compare_strategies_by_symbol.py TICKER [options]
```

Pass another ticker as the first argument, for example:

```bash
python compare_strategies_by_symbol.py AAPL
python compare_strategies_by_symbol.py BTC-USD
python compare_strategies_by_symbol.py '^FTSE'
```

Symbols beginning with `^` should be quoted so the shell passes them through
unchanged. Yahoo Finance does not require an API key for these downloads.

The default run uses 2017-01-01 through 2021-12-31 as training data, selects
the best candidate from each strategy family, and evaluates those frozen
winners from 2022-01-01 through the latest available trading day. It compares:

- nine dual-moving-average pairs;
- five price-above-SMA windows;
- five positive trailing-return windows; and
- buy and hold as the out-of-sample benchmark.

Selection uses training return, then lower drawdown and fewer completed trades
as tie-breakers. Signals are delayed by one bar and execute at the next daily
close, with independent starting capital and a `0.1%` fee per order by default.
The output includes return, CAGR, drawdown, Sharpe ratio, volatility, exposure,
and trade counts. Annualized metrics infer the observation frequency, so they
support both weekday-traded assets and seven-day markets. Override the dates
and assumptions when needed:

```bash
python compare_strategies_by_symbol.py SPY \
  --start 2017-01-01 --split 2022-01-01 --end 2026-01-01 \
  --init-cash 1000 --fee-pct 0.1
```

For notebook analysis, import the renamed module directly. Include warm-up
history before the requested training period so the longest indicators are
ready at the boundary:

```python
from datetime import date, timedelta

from analyze_holding import download_close
from compare_strategies_by_symbol import (
    WARMUP_CALENDAR_DAYS,
    compare_strategies,
)

symbol = "AAPL"
start = date(2017, 1, 1)
split = date(2022, 1, 1)
end = date(2026, 1, 1)

prices = download_close(
    symbol,
    start=start - timedelta(days=WARMUP_CALENDAR_DAYS),
    end=end,
)
comparison = compare_strategies(
    prices,
    start=start,
    split=split,
    end=end,
    init_cash=1000,
    fee_pct=0.1,
)

comparison.training_results
comparison.selected_training
comparison.test_results
```

The comparison is a historical simulation, not evidence that a strategy will
continue to perform out of sample.

## Tests

The unit tests use synthetic or mocked data and do not require internet access:

```bash
python -m unittest discover -s tests -v
```

Yahoo Finance data can occasionally be missing, adjusted, throttled, or
unavailable. This project is intended for research and education, not financial
advice.
