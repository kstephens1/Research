#!/usr/bin/env python3
"""Simulate up to ten paired DMAC strategies with VectorBT."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import vectorbt as vbt

from analyze_dmac import positive_int
from analyze_holding import (
    MarketDataError,
    download_close,
    format_index_value,
    iso_date,
    non_negative_float,
    positive_float,
)

MAX_STRATEGIES = 10
DEFAULT_FAST_WINDOWS = (10, 20)
DEFAULT_SLOW_WINDOWS = (30, 30)


@dataclass(frozen=True)
class DmacStrategySummary:
    fast_window: int
    slow_window: int
    final_value: float
    total_profit: float
    total_return: float
    entry_signals: int
    exit_signals: int
    completed_trades: int
    open_trades: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare up to ten paired dual moving average strategies."
    )
    parser.add_argument("ticker", nargs="?", default="BTC-USD")
    parser.add_argument(
        "--fast-windows",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_FAST_WINDOWS),
        help="fast MA windows paired by position (default: 10 20)",
    )
    parser.add_argument(
        "--slow-windows",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_SLOW_WINDOWS),
        help="slow MA windows paired by position (default: 30 30)",
    )
    parser.add_argument("--init-cash", type=positive_float, default=100.0)
    parser.add_argument("--start", type=iso_date)
    parser.add_argument("--end", type=iso_date, help="exclusive end date")
    parser.add_argument(
        "--fee-pct",
        type=non_negative_float,
        default=0.0,
        help="transaction fee percentage; 0.1 means 0.1%%",
    )
    return parser


def validate_window_pairs(
    fast_windows: Sequence[int], slow_windows: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    fast_windows = tuple(fast_windows)
    slow_windows = tuple(slow_windows)
    if not fast_windows:
        raise ValueError("at least one strategy is required")
    if len(fast_windows) != len(slow_windows):
        raise ValueError("fast and slow window lists must have the same length")
    if len(fast_windows) > MAX_STRATEGIES:
        raise ValueError(f"a maximum of {MAX_STRATEGIES} strategies is allowed")

    pairs = tuple(zip(fast_windows, slow_windows))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate moving average pairs are not allowed")
    for fast_window, slow_window in pairs:
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("moving average windows must be greater than zero")
        if fast_window >= slow_window:
            raise ValueError(
                f"fast window {fast_window} must be smaller than "
                f"slow window {slow_window}"
            )
    return pairs


def to_1d_values(value: object, count: int, label: str) -> np.ndarray:
    values = np.asarray(value).reshape(-1)
    if len(values) != count:
        raise RuntimeError(
            f"VectorBT returned {len(values)} {label} values for {count} strategies"
        )
    return values


def analyze_dmac_multi(
    close: pd.Series,
    fast_windows: Sequence[int] = DEFAULT_FAST_WINDOWS,
    slow_windows: Sequence[int] = DEFAULT_SLOW_WINDOWS,
    init_cash: float = 100.0,
    fee_pct: float = 0.0,
) -> tuple[DmacStrategySummary, ...]:
    pairs = validate_window_pairs(fast_windows, slow_windows)
    largest_slow = max(slow for _, slow in pairs)
    if len(close) <= largest_slow:
        raise MarketDataError(
            f"need at least {largest_slow + 1} observations for "
            f"a {largest_slow}-day slow moving average crossover"
        )

    fast_values = [fast for fast, _ in pairs]
    slow_values = [slow for _, slow in pairs]
    fast_ma = vbt.MA.run(close, fast_values, short_name="fast")
    slow_ma = vbt.MA.run(close, slow_values, short_name="slow")
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=init_cash,
        fees=fee_pct / 100.0,
        direction="longonly",
        freq="1D",
    )

    count = len(pairs)
    metrics = {
        "final": to_1d_values(portfolio.final_value(), count, "final value"),
        "profit": to_1d_values(portfolio.total_profit(), count, "profit"),
        "return": to_1d_values(portfolio.total_return(), count, "return"),
        "entries": to_1d_values(entries.sum(), count, "entry signal"),
        "exits": to_1d_values(exits.sum(), count, "exit signal"),
        "closed": to_1d_values(portfolio.trades.closed.count(), count, "closed trade"),
        "open": to_1d_values(portfolio.trades.open.count(), count, "open trade"),
    }
    return tuple(
        DmacStrategySummary(
            fast_window=fast,
            slow_window=slow,
            final_value=float(metrics["final"][index]),
            total_profit=float(metrics["profit"][index]),
            total_return=float(metrics["return"][index]),
            entry_signals=int(metrics["entries"][index]),
            exit_signals=int(metrics["exits"][index]),
            completed_trades=int(metrics["closed"][index]),
            open_trades=int(metrics["open"][index]),
        )
        for index, (fast, slow) in enumerate(pairs)
    )


def print_summary(
    ticker: str,
    close: pd.Series,
    init_cash: float,
    fee_pct: float,
    summaries: Sequence[DmacStrategySummary],
) -> None:
    print(f"Ticker: {ticker}")
    print(
        f"Period: {format_index_value(close.index[0])} to "
        f"{format_index_value(close.index[-1])}"
    )
    print(f"Observations: {len(close)}")
    print(f"Initial cash per strategy: {init_cash:.2f}")
    print(f"Fee rate: {fee_pct:.4f}%")
    print()
    header = (
        f"{'#':>3} {'Fast':>6} {'Slow':>6} {'Entries':>8} {'Exits':>7} "
        f"{'Closed':>8} {'Open':>6} {'Final value':>12} {'Profit':>12} {'Return':>10}"
    )
    print(header)
    print("-" * len(header))
    for index, result in enumerate(summaries, start=1):
        print(
            f"{index:>3} {result.fast_window:>6} {result.slow_window:>6} "
            f"{result.entry_signals:>8} {result.exit_signals:>7} "
            f"{result.completed_trades:>8} {result.open_trades:>6} "
            f"{result.final_value:>12.2f} {result.total_profit:>12.2f} "
            f"{result.total_return * 100:>9.2f}%"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("ticker must not be empty")
    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("--start must be earlier than --end")
    try:
        validate_window_pairs(args.fast_windows, args.slow_windows)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        close = download_close(ticker, start=args.start, end=args.end)
        summaries = analyze_dmac_multi(
            close,
            fast_windows=args.fast_windows,
            slow_windows=args.slow_windows,
            init_cash=args.init_cash,
            fee_pct=args.fee_pct,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(ticker, close, args.init_cash, args.fee_pct, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
