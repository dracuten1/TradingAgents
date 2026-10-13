"""Multi-timeframe OHLCV snapshot for crypto, from the Binance public API.

Why this exists: the Yahoo-based daily chain (a) has no sub-daily timeframes
and (b) serves the *forming* intraday daily row whose live Close can exceed a
stale High (observed for BTC-USD on 2026-10-13: Close 83,921.21 > High
83,858.10). Binance klines are UTC-aligned for any interval, need no API key,
and update High/Low correctly while a candle forms.

Contract:
- Indicators are computed on CLOSED candles only; the forming candle (if any)
  is rendered separately and labeled ``FORMING (incomplete)``.
- Every row is validated (Low <= min(Open, Close) and High >= max(Open, Close));
  invalid rows are dropped and the drop is reported, never silently kept.
- Rows opening after the requested analysis date are excluded (no look-ahead
  for past-date runs).
- Deterministic: no LLM, no randomness; output is plain markdown tables.

Non-crypto symbols return a friendly "unavailable" string instead of raising —
an exception out of an agent tool would end the run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from stockstats import wrap

UTC = timezone.utc

_BINANCE_BASE = "https://api.binance.com"

INTERVAL_SECONDS: dict[str, int] = {
    "1h": 3600,
    "4h": 14400,
    "12h": 43200,
    "1d": 86400,
}
DEFAULT_INTERVALS: tuple[str, ...] = ("1h", "4h", "12h", "1d")

# Enough closed rows for SMA200 on every interval (1h: ~12.5 days).
KLINE_LIMITS: dict[str, int] = {
    "1h": 300,
    "4h": 300,
    "12h": 250,
    "1d": 300,
}

INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)

_USD_QUOTES = {"USD", "USDT", "USDC"}


def binance_symbol(symbol: str) -> str | None:
    """Map a Yahoo-style crypto pair to a Binance spot symbol; None if not crypto."""
    s = symbol.strip().upper()
    if s.endswith("USDT") or s.endswith("USDC"):
        return s
    if "-" in s:
        base, quote = s.split("-", 1)
        if quote in _USD_QUOTES and base:
            return f"{base}USDT"
    return None


def fetch_klines(pair: str, interval: str, limit: int, _retries: int = 2) -> list[dict]:
    """Fetch klines from Binance's public API and type them. Raises URLError."""
    url = f"{_BINANCE_BASE}/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}"
    last_exc: Exception | None = None
    for _ in range(_retries + 1):
        try:
            with urlopen(url, timeout=20) as resp:
                payload = json.loads(resp.read())
            return [
                {
                    "open_time": datetime.fromtimestamp(k[0] / 1000, tz=UTC),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
                for k in payload
            ]
        except (URLError, OSError, ValueError, KeyError, IndexError) as exc:
            last_exc = exc
            time.sleep(1.0)
    raise last_exc  # type: ignore[misc]


def _fetch_rows(pair: str, interval: str, limit: int) -> list[dict]:
    """Indirection seam: unit tests monkeypatch this to avoid the network."""
    return fetch_klines(pair, interval, limit)


def _validate_ohlcv(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop rows violating Low <= min(O,C) <= max(O,C) <= High (or with NaN)."""
    clean = []
    for r in rows:
        try:
            ok = (
                r["low"] <= min(r["open"], r["close"])
                and r["high"] >= max(r["open"], r["close"])
                and all(pd.notna(r[k]) for k in ("open", "high", "low", "close", "volume"))
            )
        except (KeyError, TypeError):
            ok = False
        if ok:
            clean.append(r)
    return clean, len(rows) - len(clean)


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


def _indicator_table(closed: list[dict]) -> tuple[list[str], dict]:
    """Markdown indicator rows and the latest-row indicator values (for the summary)."""
    df = pd.DataFrame(closed).rename(columns={
        "open_time": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    stock_df = wrap(df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy())
    lines = ["| Indicator | Value |", "|---|---:|"]
    latest: dict[str, object] = {}
    for name in INDICATORS:
        if name == "close_200_sma" and len(closed) < 200:
            lines.append(f"| {name} | n/a (only {len(closed)} closed rows, need 200) |")
            latest[name] = None
            continue
        try:
            stock_df[name]  # triggers stockstats calculation
            value = stock_df.iloc[-1][name]
            latest[name] = value
            lines.append(f"| {name} | {_fmt(value)} |")
        except Exception:  # noqa: BLE001 — one bad indicator must not sink the timeframe
            latest[name] = None
            lines.append(f"| {name} | N/A |")
    return lines, latest


def build_multi_timeframe_snapshot(
    symbol: str,
    as_of_date: str,
    intervals: tuple[str, ...] = DEFAULT_INTERVALS,
    now: datetime | None = None,
) -> str:
    """Render a multi-timeframe snapshot: per-interval closed-candle OHLCV +
    indicators, the forming candle labeled, and a cross-timeframe summary."""
    pair = binance_symbol(symbol)
    if pair is None:
        return (
            f"Multi-timeframe data unavailable: {symbol!r} is not a crypto pair "
            "(this tool maps <BASE>-USD tickers to Binance <BASE>USDT spot only)."
        )

    now = now or datetime.now(UTC)
    end_limit = datetime.fromisoformat(as_of_date).replace(tzinfo=UTC) + timedelta(days=1) - timedelta(microseconds=1)

    summary_rows: list[str] = []
    sections: list[str] = [
        f"## Multi-timeframe snapshot for {symbol.upper()} (Binance spot {pair})",
        "",
        f"- Requested analysis date: {as_of_date} (candles opening after it are excluded)",
        f"- Snapshot time: {now.strftime('%Y-%m-%d %H:%M')} UTC",
        "- Indicators are computed on closed candles only; the forming candle is labeled separately.",
        "",
        "## Multi-timeframe summary (closed candles)",
        "",
        "| TF | Last closed (UTC) | Close | vs SMA50 | vs SMA200 | MACD hist | RSI |",
        "|---|---|---:|---|---|---:|---:|",
    ]

    for tf in intervals:
        step = INTERVAL_SECONDS[tf]
        rows = _fetch_rows(pair, tf, KLINE_LIMITS[tf])
        rows = [r for r in rows if r["open_time"] <= end_limit]
        rows, dropped = _validate_ohlcv(rows)

        forming = [r for r in rows if r["open_time"] + timedelta(seconds=step) > now]
        closed = [r for r in rows if r["open_time"] + timedelta(seconds=step) <= now]

        if not closed:
            sections.append(f"\n### {tf}\n\nNo closed candles on or before {as_of_date}.")
            continue

        last = closed[-1]
        sections.append(
            f"\n### {tf} — last closed candle {_fmt_ts(last['open_time'])} UTC "
            "(indicators computed on closed candles only)"
        )
        drop_note = f" (dropped {dropped} invalid OHLCV row{'s' if dropped != 1 else ''})" if dropped else ""
        sections.append(f"\n| Field | Value{drop_note} |\n|---|---:|")
        sections.append(f"| Open | {last['open']:.2f} |")
        sections.append(f"| High | {last['high']:.2f} |")
        sections.append(f"| Low | {last['low']:.2f} |")
        sections.append(f"| Close | {last['close']:.2f} |")
        sections.append(f"| Volume | {last['volume']:.2f} |")

        sections.append("")
        ind_lines, ind_latest = _indicator_table(closed)
        sections.extend(ind_lines)

        sections.append(f"\nRecent closed closes (last {min(10, len(closed))}):")
        sections.append("| Open (UTC) | Close |")
        sections.append("|---|---:|")
        for r in closed[-10:]:
            sections.append(f"| {_fmt_ts(r['open_time'])} | {r['close']:.2f} |")

        for r in forming:
            elapsed = (now - r["open_time"]).total_seconds() / step * 100
            elapsed = max(0.0, min(100.0, elapsed))
            sections.append(
                f"\n- FORMING (incomplete) since {_fmt_ts(r['open_time'])} UTC: "
                f"O {r['open']:.2f} / H {r['high']:.2f} / L {r['low']:.2f} / "
                f"C(live) {r['close']:.2f} — {elapsed:.1f}% elapsed. "
                "Do NOT quote its OHLC as a final candle."
            )

        # Summary row values from the same closed-candle indicator series.
        vs50 = "n/a" if ind_latest.get("close_50_sma") is None else (
            "above" if last["close"] > float(ind_latest["close_50_sma"]) else "below"
        )
        vs200 = "n/a" if ind_latest.get("close_200_sma") is None else (
            "above" if last["close"] > float(ind_latest["close_200_sma"]) else "below"
        )
        hist = ind_latest.get("macdh")
        rsi = ind_latest.get("rsi")
        summary_rows.append(
            f"| {tf} | {_fmt_ts(last['open_time'])} | {last['close']:.2f} | {vs50} | {vs200} "
            f"| {_fmt(hist)} | {_fmt(rsi)} |"
        )

    # Insert summary rows right after the summary table header.
    header_idx = sections.index("|---|---|---:|---|---|---:|---:|")
    for i, row in enumerate(summary_rows):
        sections.insert(header_idx + 1 + i, row)

    sections.append(
        "\nSource: Binance spot public API, all timestamps UTC. The 1d close here is "
        "Binance's UTC close and can differ from Yahoo/others — if it conflicts with "
        "another tool, flag the discrepancy; do not invent a reconciled number."
    )
    return "\n".join(sections)
