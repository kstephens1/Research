import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pandas as pd

import analyze_dmac_multi as app


class AnalyzeDmacMultiTests(unittest.TestCase):
    def setUp(self):
        values = [
            3, 2, 1, 2, 3, 4, 3, 2, 1, 2,
            3, 4, 5, 4, 3, 2, 3, 4, 5, 4,
            3, 2, 1, 2, 3, 4, 3, 2, 1, 2,
        ]
        self.close = pd.Series(
            values,
            index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
            dtype=float,
            name="Close",
        )

    def test_runs_paired_arrays_once_and_preserves_order(self):
        original_run = app.vbt.MA.run
        with patch.object(app.vbt.MA, "run", wraps=original_run) as ma_run:
            results = app.analyze_dmac_multi(
                self.close, fast_windows=[2, 3], slow_windows=[4, 5]
            )

        self.assertEqual(
            [(result.fast_window, result.slow_window) for result in results],
            [(2, 4), (3, 5)],
        )
        self.assertEqual(ma_run.call_count, 2)
        self.assertEqual(ma_run.call_args_list[0].args[1], [2, 3])
        self.assertEqual(ma_run.call_args_list[1].args[1], [4, 5])
        self.assertNotEqual(results[0].total_return, results[1].total_return)

    def test_fee_reduces_each_traded_strategy(self):
        without_fee = app.analyze_dmac_multi(
            self.close, [2, 3], [4, 5], fee_pct=0.0
        )
        with_fee = app.analyze_dmac_multi(
            self.close, [2, 3], [4, 5], fee_pct=1.0
        )

        for baseline, charged in zip(without_fee, with_fee):
            self.assertLess(charged.final_value, baseline.final_value)

    def test_open_position_is_reported(self):
        rising_close = pd.Series(
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )

        result = app.analyze_dmac_multi(rising_close, [2], [3])[0]

        self.assertEqual(result.completed_trades, 0)
        self.assertEqual(result.open_trades, 1)

    def test_defaults_match_vectorbt_example(self):
        args = app.build_parser().parse_args([])

        self.assertEqual(args.fast_windows, [10, 20])
        self.assertEqual(args.slow_windows, [30, 30])

    def test_rejects_mismatched_list_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            app.validate_window_pairs([2, 3], [4])

    def test_rejects_more_than_ten_strategies(self):
        with self.assertRaisesRegex(ValueError, "maximum of 10"):
            app.validate_window_pairs(range(1, 12), range(20, 31))

    def test_accepts_ten_strategies(self):
        pairs = app.validate_window_pairs(range(1, 11), range(20, 30))

        self.assertEqual(len(pairs), 10)

    def test_rejects_duplicate_pairs(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            app.validate_window_pairs([2, 2], [4, 4])

    def test_rejects_invalid_window_order(self):
        with self.assertRaisesRegex(ValueError, "must be smaller"):
            app.validate_window_pairs([5], [5])

    def test_rejects_insufficient_history_for_largest_slow_window(self):
        with self.assertRaisesRegex(app.MarketDataError, "at least 31 observations"):
            app.analyze_dmac_multi(self.close, [10, 20], [20, 30])

    def test_cli_prints_one_table_row_per_strategy(self):
        output = io.StringIO()
        with patch.object(app, "download_close", return_value=self.close):
            with redirect_stdout(output):
                exit_code = app.main(
                    [
                        "BTC-USD",
                        "--fast-windows",
                        "2",
                        "3",
                        "--slow-windows",
                        "4",
                        "5",
                    ]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn("Final value", output.getvalue())
        self.assertIn("Return", output.getvalue())
        table_rows = [
            line
            for line in output.getvalue().splitlines()
            if line.strip().startswith(("1 ", "2 "))
        ]
        self.assertEqual(len(table_rows), 2)

    def test_cli_reports_download_failure(self):
        error = io.StringIO()
        with patch.object(app, "download_close", side_effect=RuntimeError("offline")):
            with redirect_stderr(error):
                exit_code = app.main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: offline", error.getvalue())

    def test_cli_rejects_mismatched_lists(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(
                    [
                        "--fast-windows",
                        "2",
                        "3",
                        "--slow-windows",
                        "4",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
