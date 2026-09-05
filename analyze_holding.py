#!/usr/bin/env python3
"""Analyze a simple buy-and-hold portfolio with VectorBT."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import pandas as pd
import vectorbt as vbt


class MarketDataError(RuntimeError):
    """Raised when usable market data cannot be obtained."""


@dataclass(frozen=True)
class HoldingSummary:
    """Headline results from a buy-and-hold simulation."""

    final_value: float
    total_profit: float
    total_return: float


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD format") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a Yahoo Finance asset as a buy-and-hold portfolio."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="BTC-USD",
        help="Yahoo Finance ticker (default: BTC-USD)",
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


def extract_close(data: object) -> pd.Series:
    """Extract and validate a close-price series from VectorBT data."""
    try:
        close = data.get("Close")  # type: ignore[attr-defined]
    except (KeyError, TypeError, AttributeError) as exc:
        raise MarketDataError("downloaded data does not contain a Close column") from exc

    if not isinstance(close, pd.Series):
        raise MarketDataError("downloaded Close data is not a single price series")

    close = close.dropna()
    if close.empty:
        raise MarketDataError("Yahoo Finance returned no usable close prices")
    return close


def download_close(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    auto_adjust: bool = True,
) -> pd.Series:
    """Download daily close prices from Yahoo Finance."""
    kwargs: dict[str, str | bool] = {"auto_adjust": auto_adjust}
    if start is not None:
        kwargs["start"] = f"{start.isoformat()} 00:00:00 UTC"
    if end is not None:
        kwargs["end"] = f"{end.isoformat()} 00:00:00 UTC"
    return extract_close(vbt.YFData.download(ticker, **kwargs))


def analyze_holding(
    close: pd.Series, init_cash: float = 100.0, fee_pct: float = 0.0
) -> HoldingSummary:
    """Run a long-only buy-and-hold simulation."""
    portfolio = vbt.Portfolio.from_holding(
        close,
        init_cash=init_cash,
        fees=fee_pct / 100.0,
    )
    return HoldingSummary(
        final_value=float(portfolio.final_value()),
        total_profit=float(portfolio.total_profit()),
        total_return=float(portfolio.total_return()),
    )


def format_index_value(value: object) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()  # type: ignore[union-attr]
    return str(value)


def print_summary(
    ticker: str,
    close: pd.Series,
    init_cash: float,
    fee_pct: float,
    summary: HoldingSummary,
) -> None:
    """Print a stable, human-readable report."""
    print(f"Ticker: {ticker}")
    print(
        "Period: "
        f"{format_index_value(close.index[0])} to "
        f"{format_index_value(close.index[-1])}"
    )
    print(f"Observations: {len(close)}")
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
    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("--start must be earlier than --end")

    try:
        close = download_close(ticker, start=args.start, end=args.end)
        summary = analyze_holding(close, args.init_cash, args.fee_pct)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(ticker, close, args.init_cash, args.fee_pct, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
