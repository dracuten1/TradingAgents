"""Tests for the Binance multi-timeframe snapshot (crypto 1h/4h/12h/1d).

All fetches are monkeypatched — no network in unit tests.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.dataflows.multi_timeframe import (
    binance_symbol,
    build_multi_timeframe_snapshot,
    fetch_klines,
)

UTC = timezone.utc


def make_klines(n: int, interval_s: int, start: datetime, base: float = 60000.0, bad_row: int | None = None):
    """Deterministic klines: closes base+10i, valid OHLC unless bad_row given."""
    rows = []
    for i in range(n):
        close = base + 10 * i
        row = {
            "open_time": start + timedelta(seconds=i * interval_s),
            "open": close - 5.0,
            "high": close + 8.0,
            "low": close - 8.0,
            "close": close,
            "volume": 100.0 + i,
        }
        if i == bad_row:
            row["close"] = row["high"] + 500.0  # Close > High → invalid
        rows.append(row)
    return rows


class TestSymbolMapping:
    def test_crypto_usd_pairs_map_to_usdt(self):
        assert binance_symbol("BTC-USD") == "BTCUSDT"
        assert binance_symbol("ETH-USD") == "ETHUSDT"

    def test_passthrough_and_non_crypto(self):
        assert binance_symbol("BTCUSDT") == "BTCUSDT"
        assert binance_symbol("AAPL") is None
        assert binance_symbol("NVDA") is None


class TestValidation:
    def test_invalid_ohlcv_rows_are_dropped_and_reported(self, monkeypatch):
        start = datetime(2026, 10, 1, tzinfo=UTC)
        rows = make_klines(60, 3600, start, bad_row=30)
        monkeypatch.setattr(
            "tradingagents.dataflows.multi_timeframe._fetch_rows",
            lambda sym, interval, limit: rows,
        )
        out = build_multi_timeframe_snapshot(
            "BTC-USD", "2026-10-13", intervals=("1h",), now=start + timedelta(hours=70)
        )
        assert "dropped 1 invalid OHLCV row" in out
        # the invalid close (high+500 = 60308+500) must not appear anywhere
        assert "60808.00" not in out


class TestFormingCandle:
    def test_forming_candle_excluded_from_indicators_and_labeled(self, monkeypatch):
        start = datetime(2026, 10, 1, tzinfo=UTC)
        n = 60
        rows = make_klines(n, 3600, start)
        # now = 50% into the last hourly candle → last is forming
        now = start + timedelta(hours=n - 1, minutes=30)
        monkeypatch.setattr(
            "tradingagents.dataflows.multi_timeframe._fetch_rows",
            lambda sym, interval, limit: rows,
        )
        out = build_multi_timeframe_snapshot("BTC-USD", "2026-10-13", intervals=("1h",), now=now)
        # forming candle labeled, with elapsed pct
        assert "FORMING (incomplete)" in out
        assert "50.0% elapsed" in out
        # last CLOSED close = 60000+10*58 = 60580 appears as a close, not the forming one (60590)
        assert "60580.00" in out
        # indicators computed on closed candles only → header states it
        assert "computed on closed candles only" in out


class TestSnapshotShape:
    def test_contains_all_intervals_summary_and_source(self, monkeypatch):
        start = datetime(2026, 9, 1, tzinfo=UTC)

        def fake(sym, interval, limit):
            step = {"1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400}[interval]
            return make_klines(60, step, start)

        monkeypatch.setattr("tradingagents.dataflows.multi_timeframe._fetch_rows", fake)
        out = build_multi_timeframe_snapshot(
            "BTC-USD", "2026-10-13", now=start + timedelta(days=60)
        )
        for tf in ("1h", "4h", "12h", "1d"):
            assert f"### {tf}" in out
        assert "Multi-timeframe summary" in out
        assert "Binance spot BTCUSDT" in out
        # 60 rows → SMA200 must be explicitly insufficient, not silently wrong
        assert "n/a (only 60 closed rows" in out

    def test_sma200_numeric_when_enough_rows(self, monkeypatch):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        rows = make_klines(250, 86400, start)

        def fake(sym, interval, limit):
            return rows

        monkeypatch.setattr("tradingagents.dataflows.multi_timeframe._fetch_rows", fake)
        out = build_multi_timeframe_snapshot("BTC-USD", "2026-10-13", intervals=("1d",), now=start + timedelta(days=251))
        assert "n/a (only" not in out
        assert "| close_200_sma |" in out


class TestLookahead:
    def test_past_analysis_date_excludes_later_candles(self, monkeypatch):
        start = datetime(2026, 10, 1, tzinfo=UTC)
        rows = make_klines(10, 86400, start)  # daily candles Oct 1..Oct 10
        monkeypatch.setattr(
            "tradingagents.dataflows.multi_timeframe._fetch_rows",
            lambda sym, interval, limit: rows,
        )
        out = build_multi_timeframe_snapshot("BTC-USD", "2026-10-05", intervals=("1d",), now=start + timedelta(days=30))
        # last candle on/before Oct 5: i=4 → close 60040
        assert "60040.00" in out
        # Oct 6+ closes must not appear
        assert "60050.00" not in out
        assert "60090.00" not in out


class TestNonCrypto:
    def test_non_crypto_returns_friendly_unavailable(self):
        out = build_multi_timeframe_snapshot("AAPL", "2026-10-13")
        assert "unavailable" in out.lower()
        assert "crypto" in out.lower()


def test_fetch_klines_parses_binance_payload(monkeypatch):
    """fetch_klines turns a raw Binance kline list into typed rows."""
    class FakeResp:
        status = 200

        def read(self):
            import json
            return json.dumps([
                [1791871200000, "83681.87", "83802.84", "83579.10", "83774.70", "564.77"],
            ]).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeUlopen:
        def __call__(self, url, timeout=0):
            return FakeResp()

    monkeypatch.setattr("tradingagents.dataflows.multi_timeframe.urlopen", FakeUlopen())
    rows = fetch_klines("BTCUSDT", "1h", 5)
    assert len(rows) == 1
    r = rows[0]
    assert r["open_time"] == datetime(2026, 10, 13, 6, 0, tzinfo=UTC)
    assert r["open"] == 83681.87
    assert r["high"] == 83802.84
    assert r["low"] == 83579.10
    assert r["close"] == 83774.70
    assert r["volume"] == 564.77
