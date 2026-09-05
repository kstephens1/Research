import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

import compare_strategies_by_symbol as app


class CompareTickerStrategiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.bdate_range("2015-12-01", "2024-12-31")
        offset = np.arange(len(index))
        cls.close = pd.Series(
            100.0 * np.exp(0.0003 * offset + 0.08 * np.sin(offset / 35.0)),
            index=index,
            name="Close",
        )

    def test_curated_candidate_count(self):
        specs = app.strategy_specs()

        self.assertEqual(len(specs), 19)
        self.assertEqual(sum(spec.family == "DMAC" for spec in specs), 9)
        self.assertEqual(sum(spec.family == "Price trend" for spec in specs), 5)
        self.assertEqual(sum(spec.family == "Momentum" for spec in specs), 5)

    def test_execution_state_is_shifted_one_bar(self):
        states = pd.DataFrame({"test": [False, True, False, True]})

        shifted = app.shift_for_next_close(states)

        self.assertEqual(shifted["test"].tolist(), [False, False, True, False])

    def test_annualization_adapts_to_ticker_observation_frequency(self):
        daily = pd.date_range("2024-01-01", "2025-01-01", freq="D")
        weekdays = daily[daily.weekday < 5]

        self.assertGreater(app.observations_per_year(daily), 360)
        self.assertLess(app.observations_per_year(weekdays), 265)

    def test_comparison_returns_training_winners_and_test_benchmark(self):
        result = app.compare_strategies(
            self.close,
            start=date(2017, 1, 1),
            split=date(2022, 1, 1),
            end=date(2025, 1, 1),
        )

        self.assertEqual(len(result.training_results), 19)
        self.assertEqual(
            result.selected_training["family"].tolist(),
            ["DMAC", "Price trend", "Momentum"],
        )
        self.assertEqual(
            result.test_results["family"].tolist(),
            ["Benchmark", "DMAC", "Price trend", "Momentum"],
        )

    def test_test_prices_cannot_change_training_selection(self):
        baseline = app.compare_strategies(
            self.close,
            start=date(2017, 1, 1),
            split=date(2022, 1, 1),
            end=date(2025, 1, 1),
        )
        changed = self.close.copy()
        test_mask = changed.index >= pd.Timestamp("2022-01-01")
        changed.loc[test_mask] *= np.linspace(1.0, 4.0, test_mask.sum())
        altered = app.compare_strategies(
            changed,
            start=date(2017, 1, 1),
            split=date(2022, 1, 1),
            end=date(2025, 1, 1),
        )

        self.assertEqual(
            baseline.selected_training["spec_key"].tolist(),
            altered.selected_training["spec_key"].tolist(),
        )

    def test_selection_ties_use_drawdown_then_trade_count(self):
        rows = []
        for family in ("DMAC", "Price trend", "Momentum"):
            rows.extend(
                [
                    {
                        "spec_key": f"{family}_worse_drawdown",
                        "family": family,
                        "total_return": 0.5,
                        "max_drawdown": -0.30,
                        "completed_trades": 2,
                    },
                    {
                        "spec_key": f"{family}_more_trades",
                        "family": family,
                        "total_return": 0.5,
                        "max_drawdown": -0.20,
                        "completed_trades": 5,
                    },
                    {
                        "spec_key": f"{family}_winner",
                        "family": family,
                        "total_return": 0.5,
                        "max_drawdown": -0.20,
                        "completed_trades": 3,
                    },
                ]
            )

        selected = app.select_family_winners(pd.DataFrame(rows))

        self.assertTrue(selected["spec_key"].str.endswith("_winner").all())

    def test_segment_enters_on_first_bar_when_state_is_active(self):
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        close = pd.Series([100.0, 110.0, 120.0], index=index)
        states = pd.DataFrame({"active": [True, True, True]}, index=index)
        spec = app.StrategySpec("active", "Test", "Always active", "-")

        result = app.simulate_segment(close, states, [spec], 1000.0, 0.1).iloc[0]

        self.assertEqual(result["completed_trades"], 0)
        self.assertEqual(result["open_trades"], 1)
        self.assertAlmostEqual(result["exposure"], 1.0)
        self.assertGreater(result["final_value"], 1000.0)
        self.assertLess(result["final_value"], 1200.0)

    def test_cli_defaults(self):
        args = app.build_parser().parse_args([])

        self.assertEqual(args.ticker, app.DEFAULT_TICKER)
        self.assertEqual(args.start, date(2017, 1, 1))
        self.assertEqual(args.split, date(2022, 1, 1))
        self.assertEqual(args.init_cash, 1000.0)
        self.assertEqual(args.fee_pct, 0.1)

    def test_cli_prints_all_result_sections(self):
        output = io.StringIO()
        with patch.object(app, "download_close", return_value=self.close):
            with redirect_stdout(output):
                exit_code = app.main(
                    [
                        "SPY",
                        "--start",
                        "2017-01-01",
                        "--split",
                        "2022-01-01",
                        "--end",
                        "2025-01-01",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("All training candidates", output.getvalue())
        self.assertIn("Selected training winners", output.getvalue())
        self.assertIn("Untouched test results", output.getvalue())
        self.assertIn("Outcome:", output.getvalue())

    def test_cli_accepts_and_normalizes_another_yahoo_ticker(self):
        output = io.StringIO()
        with patch.object(app, "download_close", return_value=self.close) as download:
            with redirect_stdout(output):
                exit_code = app.main(
                    [
                        "btc-usd",
                        "--start",
                        "2017-01-01",
                        "--split",
                        "2022-01-01",
                        "--end",
                        "2025-01-01",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(download.call_args.args[0], "BTC-USD")
        self.assertIn("Ticker: BTC-USD (adjusted close)", output.getvalue())

    def test_cli_rejects_invalid_date_order(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(["--start", "2023-01-01", "--split", "2022-01-01"])

        self.assertEqual(raised.exception.code, 2)

    def test_empty_test_period_is_rejected(self):
        with self.assertRaisesRegex(app.MarketDataError, "test period"):
            app.compare_strategies(
                self.close,
                start=date(2017, 1, 1),
                split=date(2025, 1, 1),
                end=date(2026, 1, 1),
            )


if __name__ == "__main__":
    unittest.main()
