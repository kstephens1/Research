import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pandas as pd

import analyze_dmac as app


class AnalyzeDmacTests(unittest.TestCase):
    def setUp(self):
        self.close = pd.Series(
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0],
            index=pd.date_range("2024-01-01", periods=9, freq="D"),
            name="Close",
        )

    def test_crossover_signals_create_completed_trade(self):
        result = app.analyze_dmac(self.close, fast_window=2, slow_window=3)

        self.assertEqual(result.entry_signals, 1)
        self.assertEqual(result.exit_signals, 1)
        self.assertEqual(result.completed_trades, 1)
        self.assertEqual(result.open_trades, 0)
        self.assertAlmostEqual(result.final_value, 100.0 * 2.0 / 3.0)
        self.assertAlmostEqual(result.total_profit, -100.0 / 3.0)
        self.assertAlmostEqual(result.total_return, -1.0 / 3.0)

    def test_fee_is_applied_to_entry_and_exit(self):
        no_fee = app.analyze_dmac(self.close, 2, 3, fee_pct=0.0)
        with_fee = app.analyze_dmac(self.close, 2, 3, fee_pct=1.0)

        self.assertAlmostEqual(no_fee.final_value, 100.0 * 2.0 / 3.0)
        self.assertAlmostEqual(
            with_fee.final_value,
            100.0 / 1.01 * (2.0 / 3.0) * 0.99,
        )
        self.assertLess(with_fee.final_value, no_fee.final_value)

    def test_open_position_is_reported(self):
        rising_close = self.close.iloc[:6]

        result = app.analyze_dmac(rising_close, fast_window=2, slow_window=3)

        self.assertEqual(result.completed_trades, 0)
        self.assertEqual(result.open_trades, 1)

    def test_rejects_fast_window_not_smaller_than_slow(self):
        with self.assertRaisesRegex(ValueError, "fast window"):
            app.analyze_dmac(self.close, fast_window=3, slow_window=3)

    def test_rejects_insufficient_history(self):
        with self.assertRaisesRegex(app.MarketDataError, "at least 11 observations"):
            app.analyze_dmac(self.close, fast_window=5, slow_window=10)

    def test_cli_defaults_to_documented_strategy(self):
        args = app.build_parser().parse_args([])

        self.assertEqual(args.ticker, "BTC-USD")
        self.assertEqual(args.fast_window, 10)
        self.assertEqual(args.slow_window, 20)
        self.assertEqual(args.init_cash, 100.0)
        self.assertEqual(args.fee_pct, 0.0)

    def test_cli_prints_strategy_summary(self):
        output = io.StringIO()
        with patch.object(app, "download_close", return_value=self.close):
            with redirect_stdout(output):
                exit_code = app.main(
                    ["btc-usd", "--fast-window", "2", "--slow-window", "3"]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Ticker: BTC-USD", output.getvalue())
        self.assertIn("Fast MA window: 2 days", output.getvalue())
        self.assertIn("Completed trades: 1", output.getvalue())
        self.assertIn("Total return: -33.33%", output.getvalue())

    def test_cli_reports_download_failure(self):
        error = io.StringIO()
        with patch.object(app, "download_close", side_effect=RuntimeError("offline")):
            with redirect_stderr(error):
                exit_code = app.main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: offline", error.getvalue())

    def test_cli_rejects_invalid_window_pair(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(["--fast-window", "20", "--slow-window", "10"])

        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_malformed_date(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(["--start", "01-01-2024"])

        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_reversed_dates(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(["--start", "2024-02-01", "--end", "2024-01-01"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
