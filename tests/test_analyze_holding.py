import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from unittest.mock import patch

import pandas as pd

import analyze_holding as app


class FakeData:
    def __init__(self, close):
        self.close = close

    def get(self, column):
        if column != "Close":
            raise KeyError(column)
        return self.close


class AnalyzeHoldingTests(unittest.TestCase):
    def setUp(self):
        self.close = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
            name="Close",
        )

    def test_zero_fee_buy_and_hold_results(self):
        result = app.analyze_holding(self.close, init_cash=100.0)

        self.assertAlmostEqual(result.final_value, 120.0)
        self.assertAlmostEqual(result.total_profit, 20.0)
        self.assertAlmostEqual(result.total_return, 0.2)

    def test_fee_reduces_performance(self):
        no_fee = app.analyze_holding(self.close, fee_pct=0.0)
        with_fee = app.analyze_holding(self.close, fee_pct=1.0)

        self.assertLess(with_fee.final_value, no_fee.final_value)
        self.assertLess(with_fee.total_profit, no_fee.total_profit)

    def test_extract_close_rejects_empty_data(self):
        with self.assertRaisesRegex(app.MarketDataError, "no usable close prices"):
            app.extract_close(FakeData(pd.Series(dtype=float)))

    def test_extract_close_rejects_missing_close_column(self):
        class MissingCloseData:
            def get(self, column):
                raise KeyError(column)

        with self.assertRaisesRegex(app.MarketDataError, "Close column"):
            app.extract_close(MissingCloseData())

    def test_download_uses_utc_date_boundaries(self):
        with patch.object(
            app.vbt.YFData, "download", return_value=FakeData(self.close)
        ) as download:
            result = app.download_close(
                "BTC-USD", start=date(2024, 1, 1), end=date(2024, 1, 10)
            )

        pd.testing.assert_series_equal(result, self.close)
        download.assert_called_once_with(
            "BTC-USD",
            start="2024-01-01 00:00:00 UTC",
            end="2024-01-10 00:00:00 UTC",
        )

    def test_cli_prints_summary(self):
        output = io.StringIO()
        with patch.object(app, "download_close", return_value=self.close):
            with redirect_stdout(output):
                exit_code = app.main(["btc-usd", "--init-cash", "100"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Ticker: BTC-USD", output.getvalue())
        self.assertIn("Total profit: 20.00", output.getvalue())
        self.assertIn("Total return: 20.00%", output.getvalue())

    def test_cli_reports_download_failure(self):
        error = io.StringIO()
        with patch.object(app, "download_close", side_effect=RuntimeError("offline")):
            with redirect_stderr(error):
                exit_code = app.main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: offline", error.getvalue())

    def test_cli_rejects_nonpositive_cash(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(["--init-cash", "0"])

        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_negative_fee(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                app.main(["--fee-pct", "-0.1"])

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
