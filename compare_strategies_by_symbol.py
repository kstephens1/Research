#!/usr/bin/env python3
"""Compare curated long-only timing strategies for a Yahoo Finance symbol."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd
import vectorbt as vbt

from analyze_holding import (
    MarketDataError,
    download_close,
    iso_date,
    non_negative_float,
    positive_float,
)

WARMUP_CALENDAR_DAYS = 400
DEFAULT_TICKER = "SPY"
DEFAULT_START = date(2017, 1, 1)
DEFAULT_SPLIT = date(2022, 1, 1)

DMAC_PAIRS = (
    (10, 30),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
)
PRICE_SMA_WINDOWS = (50, 100, 150, 200, 250)
MOMENTUM_WINDOWS = (21, 63, 126, 189, 252)


@dataclass(frozen=True)
class StrategySpec:
    key: str
    family: str
    strategy: str
    parameters: str


@dataclass(frozen=True)
class ComparisonResult:
    training_results: pd.DataFrame
    selected_training: pd.DataFrame
    test_results: pd.DataFrame


def strategy_specs() -> tuple[StrategySpec, ...]:
    specs = [
        StrategySpec(
            key=f"dmac_{fast}_{slow}",
            family="DMAC",
            strategy="Moving-average crossover",
            parameters=f"{fast}/{slow}",
        )
        for fast, slow in DMAC_PAIRS
    ]
    specs.extend(
        StrategySpec(
            key=f"price_sma_{window}",
            family="Price trend",
            strategy="Price above SMA",
            parameters=str(window),
        )
        for window in PRICE_SMA_WINDOWS
    )
    specs.extend(
        StrategySpec(
            key=f"momentum_{window}",
            family="Momentum",
            strategy="Positive trailing return",
            parameters=str(window),
        )
        for window in MOMENTUM_WINDOWS
    )
    return tuple(specs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select timing strategies for any Yahoo Finance ticker on training "
            "data and test them out of sample."
        )
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=DEFAULT_TICKER,
        help=f"Yahoo Finance ticker symbol (default: {DEFAULT_TICKER})",
    )
    parser.add_argument("--start", type=iso_date, default=DEFAULT_START)
    parser.add_argument("--split", type=iso_date, default=DEFAULT_SPLIT)
    parser.add_argument("--end", type=iso_date, help="exclusive end date")
    parser.add_argument("--init-cash", type=positive_float, default=1000.0)
    parser.add_argument("--fee-pct", type=non_negative_float, default=0.1)
    return parser


def boundary_timestamp(value: date, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is not None:
        timestamp = timestamp.tz_localize(index.tz)
    return timestamp


def slice_period(
    values: pd.Series | pd.DataFrame,
    start: date,
    end: date,
) -> pd.Series | pd.DataFrame:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("market data must use a DatetimeIndex")
    start_timestamp = boundary_timestamp(start, values.index)
    end_timestamp = boundary_timestamp(end, values.index)
    return values.loc[(values.index >= start_timestamp) & (values.index < end_timestamp)]


def build_raw_states(
    close: pd.Series, specs: Sequence[StrategySpec] | None = None
) -> pd.DataFrame:
    """Build unshifted risk-on states for each candidate strategy."""
    specs = strategy_specs() if specs is None else tuple(specs)
    requested_keys = {spec.key for spec in specs}
    states: dict[str, pd.Series] = {}

    dmac_specs = [spec for spec in specs if spec.family == "DMAC"]
    if dmac_specs:
        pairs = [tuple(map(int, spec.parameters.split("/"))) for spec in dmac_specs]
        fast_ma = vbt.MA.run(close, [pair[0] for pair in pairs], short_name="fast")
        slow_ma = vbt.MA.run(close, [pair[1] for pair in pairs], short_name="slow")
        fast_values = pd.DataFrame(fast_ma.ma, index=close.index)
        slow_values = pd.DataFrame(slow_ma.ma, index=close.index)
        for index, spec in enumerate(dmac_specs):
            states[spec.key] = fast_values.iloc[:, index] > slow_values.iloc[:, index]

    trend_specs = [spec for spec in specs if spec.family == "Price trend"]
    if trend_specs:
        windows = [int(spec.parameters) for spec in trend_specs]
        averages = pd.DataFrame(
            vbt.MA.run(close, windows, short_name="trend").ma,
            index=close.index,
        )
        for index, spec in enumerate(trend_specs):
            states[spec.key] = close > averages.iloc[:, index]

    momentum_specs = [spec for spec in specs if spec.family == "Momentum"]
    for spec in momentum_specs:
        states[spec.key] = close.pct_change(int(spec.parameters)) > 0

    missing_keys = requested_keys.difference(states)
    if missing_keys:
        raise ValueError(f"unsupported strategy specifications: {sorted(missing_keys)}")
    return pd.DataFrame(states, index=close.index).fillna(False).astype(bool)


def shift_for_next_close(states: pd.DataFrame) -> pd.DataFrame:
    """Delay risk states by one bar so trades occur after signals are known."""
    return states.shift(1, fill_value=False).astype(bool)


def as_values(value: object, count: int, label: str) -> np.ndarray:
    values = np.asarray(value).reshape(-1)
    if len(values) != count:
        raise RuntimeError(
            f"VectorBT returned {len(values)} {label} values for {count} strategies"
        )
    return values


def observations_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualization frequency for weekday-only and seven-day markets."""
    if len(index) < 2:
        raise MarketDataError("evaluation period needs at least two prices")
    elapsed_years = (
        (index[-1] - index[0]).total_seconds() / 86400.0 / 365.2425
    )
    if elapsed_years <= 0:
        raise MarketDataError("evaluation period must span more than one day")
    return (len(index) - 1) / elapsed_years


def simulate_segment(
    close: pd.Series,
    states: pd.DataFrame,
    specs: Sequence[StrategySpec],
    init_cash: float,
    fee_pct: float,
) -> pd.DataFrame:
    """Simulate independent portfolios from desired long/cash states."""
    specs = tuple(specs)
    if close.empty or len(close) < 2:
        raise MarketDataError("each evaluation period needs at least two prices")
    if list(states.columns) != [spec.key for spec in specs]:
        raise ValueError("strategy state columns do not match strategy specifications")

    previous_states = states.shift(1, fill_value=False)
    entries = states & ~previous_states
    exits = ~states & previous_states
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=init_cash,
        fees=fee_pct / 100.0,
        direction="longonly",
        freq="1D",
    )

    count = len(specs)
    final_values = as_values(portfolio.final_value(), count, "final value")
    total_profits = as_values(portfolio.total_profit(), count, "profit")
    total_returns = as_values(portfolio.total_return(), count, "return")
    max_drawdowns = as_values(portfolio.max_drawdown(), count, "drawdown")
    exposures = as_values(portfolio.position_coverage(), count, "exposure")
    completed = as_values(portfolio.trades.closed.count(), count, "closed trade")
    open_trades = as_values(portfolio.trades.open.count(), count, "open trade")

    daily_returns = np.asarray(portfolio.returns())
    if daily_returns.ndim == 1:
        daily_returns = daily_returns.reshape(-1, 1)
    annualization_periods = observations_per_year(close.index)
    annual_volatility = np.std(daily_returns, axis=0, ddof=1) * np.sqrt(
        annualization_periods
    )
    return_std = np.std(daily_returns, axis=0, ddof=1)
    sharpe = np.divide(
        np.mean(daily_returns, axis=0) * np.sqrt(annualization_periods),
        return_std,
        out=np.full(count, np.nan),
        where=return_std != 0,
    )
    elapsed_years = (
        (close.index[-1] - close.index[0]).total_seconds() / 86400.0 / 365.2425
    )
    cagr = np.power(final_values / init_cash, 1.0 / elapsed_years) - 1.0

    return pd.DataFrame(
        [
            {
                "spec_key": spec.key,
                "family": spec.family,
                "strategy": spec.strategy,
                "parameters": spec.parameters,
                "final_value": float(final_values[index]),
                "total_profit": float(total_profits[index]),
                "total_return": float(total_returns[index]),
                "cagr": float(cagr[index]),
                "max_drawdown": float(max_drawdowns[index]),
                "sharpe": float(sharpe[index]),
                "annual_volatility": float(annual_volatility[index]),
                "exposure": float(exposures[index]),
                "completed_trades": int(completed[index]),
                "open_trades": int(open_trades[index]),
            }
            for index, spec in enumerate(specs)
        ]
    )


def select_family_winners(training_results: pd.DataFrame) -> pd.DataFrame:
    """Select one winner per family using training metrics only."""
    winners = []
    for family in ("DMAC", "Price trend", "Momentum"):
        candidates = training_results.loc[training_results["family"] == family]
        if candidates.empty:
            raise ValueError(f"training results do not include {family}")
        ranked = candidates.sort_values(
            ["total_return", "max_drawdown", "completed_trades", "spec_key"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        winners.append(ranked.iloc[0])
    return pd.DataFrame(winners).reset_index(drop=True)


def compare_strategies(
    close: pd.Series,
    start: date = DEFAULT_START,
    split: date = DEFAULT_SPLIT,
    end: date | None = None,
    init_cash: float = 1000.0,
    fee_pct: float = 0.1,
) -> ComparisonResult:
    """Select candidates on the training period and evaluate them out of sample."""
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("market data must use a DatetimeIndex")
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("market data index must be sorted and unique")
    if end is None:
        last_timestamp = close.index[-1]
        end = (last_timestamp + pd.Timedelta(days=1)).date()
    if not start < split < end:
        raise ValueError("dates must satisfy start < split < end")

    specs = strategy_specs()
    raw_states = build_raw_states(close, specs)
    executable_states = shift_for_next_close(raw_states)
    training_close = slice_period(close, start, split)
    test_close = slice_period(close, split, end)
    if not isinstance(training_close, pd.Series) or not isinstance(test_close, pd.Series):
        raise TypeError("expected close-price series")
    if training_close.empty:
        raise MarketDataError("training period contains no prices")
    if test_close.empty:
        raise MarketDataError("test period contains no prices")

    training_states = executable_states.loc[training_close.index, [s.key for s in specs]]
    training_results = simulate_segment(
        training_close,
        training_states,
        specs,
        init_cash,
        fee_pct,
    )
    selected_training = select_family_winners(training_results)
    specs_by_key = {spec.key: spec for spec in specs}
    selected_specs = tuple(
        specs_by_key[key] for key in selected_training["spec_key"].tolist()
    )

    benchmark = StrategySpec(
        key="buy_and_hold",
        family="Benchmark",
        strategy="Buy and hold",
        parameters="Always invested",
    )
    test_specs = (benchmark, *selected_specs)
    test_states = pd.DataFrame(index=test_close.index)
    test_states[benchmark.key] = True
    for spec in selected_specs:
        test_states[spec.key] = executable_states.loc[test_close.index, spec.key]
    test_results = simulate_segment(
        test_close,
        test_states,
        test_specs,
        init_cash,
        fee_pct,
    )
    return ComparisonResult(
        training_results=training_results,
        selected_training=selected_training,
        test_results=test_results,
    )


def print_results_table(title: str, results: pd.DataFrame) -> None:
    columns = [
        "family",
        "parameters",
        "final_value",
        "total_profit",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "annual_volatility",
        "exposure",
        "completed_trades",
        "open_trades",
    ]
    formatters = {
        "final_value": lambda value: f"{value:,.2f}",
        "total_profit": lambda value: f"{value:,.2f}",
        "total_return": lambda value: f"{value:.2%}",
        "cagr": lambda value: f"{value:.2%}",
        "max_drawdown": lambda value: f"{value:.2%}",
        "sharpe": lambda value: f"{value:.2f}",
        "annual_volatility": lambda value: f"{value:.2%}",
        "exposure": lambda value: f"{value:.2%}",
    }
    print(f"\n{title}")
    print(results[columns].to_string(index=False, formatters=formatters))


def print_comparison(
    ticker: str,
    start: date,
    split: date,
    end: date,
    init_cash: float,
    fee_pct: float,
    result: ComparisonResult,
) -> None:
    print(f"Ticker: {ticker} (adjusted close)")
    print(f"Training period: {start.isoformat()} to {split.isoformat()} (exclusive)")
    print(f"Test period: {split.isoformat()} to {end.isoformat()} (exclusive)")
    print(f"Initial cash per portfolio: {init_cash:.2f}")
    print(f"Fee rate per order: {fee_pct:.4f}%")
    print("Execution: next available daily close after each signal")
    print_results_table("All training candidates", result.training_results)
    print_results_table("Selected training winners", result.selected_training)
    print_results_table("Untouched test results", result.test_results)

    benchmark_return = float(
        result.test_results.loc[
            result.test_results["family"] == "Benchmark", "total_return"
        ].iloc[0]
    )
    strategies = result.test_results.loc[result.test_results["family"] != "Benchmark"]
    best = strategies.sort_values("total_return", ascending=False).iloc[0]
    comparison = "beat" if best["total_return"] > benchmark_return else "did not beat"
    print(
        f"\nOutcome: best frozen strategy was {best['family']} "
        f"({best['parameters']}) at {best['total_return']:.2%}; it {comparison} "
        f"buy-and-hold at {benchmark_return:.2%}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker.strip().upper()
    end = args.end or (date.today() + timedelta(days=1))
    if not ticker:
        parser.error("ticker must not be empty")
    if not args.start < args.split < end:
        parser.error("dates must satisfy --start < --split < --end")

    download_start = args.start - timedelta(days=WARMUP_CALENDAR_DAYS)
    try:
        close = download_close(ticker, start=download_start, end=end)
        result = compare_strategies(
            close,
            start=args.start,
            split=args.split,
            end=end,
            init_cash=args.init_cash,
            fee_pct=args.fee_pct,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_comparison(
        ticker,
        args.start,
        args.split,
        end,
        args.init_cash,
        args.fee_pct,
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
