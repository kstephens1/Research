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

## Tests

The unit tests use synthetic or mocked data and do not require internet access:

```bash
python -m unittest discover -s tests -v
```

Yahoo Finance data can occasionally be missing, adjusted, throttled, or
unavailable. This project is intended for research and education, not financial
advice.
