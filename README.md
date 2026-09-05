# VectorBT Trading Research

A small command-line tool for evaluating a Yahoo Finance asset as a simple
buy-and-hold portfolio with [VectorBT](https://vectorbt.dev/).

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

## Usage

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

## Tests

The unit tests use synthetic or mocked data and do not require internet access:

```bash
python -m unittest discover -s tests -v
```

Yahoo Finance data can occasionally be missing, adjusted, throttled, or
unavailable. This project is intended for research and education, not financial
advice.
