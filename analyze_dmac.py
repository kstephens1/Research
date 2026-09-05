#!/usr/bin/env python3
"""Simulate a dual moving average crossover strategy with VectorBT."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import vectorbt as vbt

from analyze_holding import (
    MarketDataError,
    download_close,
    format_index_value,
    iso_date,
    non_negative_float,
    positive_float,
)


@dataclass(frozen=True)
class DmacSummary:
    """Headline results from a dual moving average crossover simulation."""

    final_value: float
    total_profit: float
    total_return: float
    entry_signals: int
    exit_signals: int
    completed_trades: int
    open_trades: int


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a dual moving average crossover strategy."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="BTC-USD",
        help="Yahoo Finance ticker (default: BTC-USD)",
    )
    parser.add_argument(
        "--fast-window",
        type=positive_int,
        default=10,
        help="fast simple moving average window in days (default: 10)",
    )
    parser.add_argument(
        "--slow-window",
        type=positive_int,
        default=20,
        help="slow simple moving average window in days (default: 20)",
    )
    parser.add_argument(
        "--init-cash",
        type=positive_float,
        default=100.0,
        help="initial portfolio cash (default: 100)",
    )
    parser.add_argument(
        "--start",
        type=iso_date,
        help="first date to request, in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        type=iso_date,
        help="exclusive end date to request, in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--fee-pct",
        type=non_negative_float,
        default=0.0,
        help="transaction fee in percentage points; 0.1 means 0.1%% (default: 0)",
    )
    return parser


def analyze_dmac(
    close: pd.Series,
    fast_window: int = 10,
    slow_window: int = 20,
    init_cash: float = 100.0,
    fee_pct: float = 0.0,
) -> DmacSummary:
    """Run a long-only dual moving average crossover simulation."""
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("moving average windows must be greater than zero")
    if fast_window >= slow_window:
        raise ValueError("fast window must be smaller than slow window")
    if len(close) <= slow_window:
        raise MarketDataError(
            f"need at least {slow_window + 1} observations for "
            f"a {slow_window}-day slow moving average crossover"
        )

    fast_ma = vbt.MA.run(close, fast_window, short_name="fast")
    slow_ma = vbt.MA.run(close, slow_window, short_name="slow")
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
    return DmacSummary(
        final_value=float(portfolio.final_value()),
        total_profit=float(portfolio.total_profit()),
        total_return=float(portfolio.total_return()),
        entry_signals=int(entries.sum()),
        exit_signals=int(exits.sum()),
        completed_trades=int(portfolio.trades.closed.count()),
        open_trades=int(portfolio.trades.open.count()),
    )


def print_summary(
    ticker: str,
    close: pd.Series,
    fast_window: int,
    slow_window: int,
    init_cash: float,
    fee_pct: float,
    summary: DmacSummary,
) -> None:
    """Print a stable, human-readable strategy report."""
    print(f"Ticker: {ticker}")
    print(
        "Period: "
        f"{format_index_value(close.index[0])} to "
        f"{format_index_value(close.index[-1])}"
    )
    print(f"Observations: {len(close)}")
    print(f"Fast MA window: {fast_window} days")
    print(f"Slow MA window: {slow_window} days")
    print(f"Entry signals: {summary.entry_signals}")
    print(f"Exit signals: {summary.exit_signals}")
    print(f"Completed trades: {summary.completed_trades}")
    print(f"Open trades: {summary.open_trades}")
    print(f"Initial cash: {init_cash:.2f}")
    print(f"Fee rate: {fee_pct:.4f}%")
    print(f"Final value: {summary.final_value:.2f}")
    print(f"Total profit: {summary.total_profit:.2f}")
    print(f"Total return: {summary.total_return * 100:.2f}%")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker.strip().upper()

    if not ticker:
        parser.error("ticker must not be empty")
    if args.fast_window >= args.slow_window:
        parser.error("--fast-window must be smaller than --slow-window")
    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("--start must be earlier than --end")

    try:
        close = download_close(ticker, start=args.start, end=args.end)
        summary = analyze_dmac(
            close,
            fast_window=args.fast_window,
            slow_window=args.slow_window,
            init_cash=args.init_cash,
            fee_pct=args.fee_pct,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(
        ticker,
        close,
        args.fast_window,
        args.slow_window,
        args.init_cash,
        args.fee_pct,
        summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
