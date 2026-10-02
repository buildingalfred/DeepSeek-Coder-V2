"""Loading price data (OHLCV) from CSV, plus a synthetic sample for trying the village offline."""

from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import PRICE_COLUMNS

_ALIASES = {
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "last": "close",
    "adj close": "adj_close", "adj_close": "adj_close", "vol": "volume", "<open>": "open",
    "<high>": "high", "<low>": "low", "<close>": "close", "<vol>": "volume", "<date>": "date",
    "<time>": "time", "<dtyyyymmdd>": "date", "gmt time": "datetime", "local time": "datetime",
    "date time": "datetime", "date_time": "datetime", "totalvolume": "volume", "up": "upvol",
}
_TIME_NAMES = ("datetime", "timestamp", "open_time", "date", "time", "index")
_FORMATS = ["%Y%m%d %H%M%S", "%Y%m%d %H:%M:%S", "%Y%m%d %H%M", "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
            "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d",
            "%d.%m.%Y", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M:%S", "%m-%d-%Y %H:%M",
            "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]
DATA_SUFFIXES = (".csv", ".txt")


def _sniff(path: Path) -> tuple[str, bool]:
    """Guess the separator and whether the first line is a header."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    sep = max([",", ";", "\t", "|"], key=first.count)
    if first.count(sep) == 0:
        sep = r"\s+"
    cells = [c.strip().strip('"') for c in (first.split() if sep == r"\s+" else first.split(sep))]
    header = any(any(ch.isalpha() for ch in c) and not _looks_like_date(c) for c in cells)
    return sep, header


def _looks_like_date(text: str) -> bool:
    return pd.notna(pd.to_datetime(text, errors="coerce", format="mixed"))


def _parse_times(raw: pd.Series) -> pd.Series:
    """Parse date/time text by finding the ONE format that fits every row.

    Day-first and month-first dates (14/12/2008 vs 12/14/2008) look the same until the day goes
    above 12, so every candidate is checked against the whole column. If several formats fit,
    the one that puts the bars in time order wins (bar data only moves forward)."""
    text = raw.astype(str).str.strip()
    # Quick screen on rows spread over the whole file, then confirm on all rows.
    spread = text.iloc[np.unique(np.linspace(0, len(text) - 1, min(len(text), 4000)).astype(int))]
    fits = []
    for fmt in _FORMATS:
        if pd.to_datetime(spread, format=fmt, errors="coerce").isna().any():
            continue
        full = pd.to_datetime(text, format=fmt, errors="coerce")
        if full.isna().any():
            continue
        fits.append((float((full.diff().dropna() > pd.Timedelta(0)).mean()), fmt, full))
    if fits:
        return max(fits, key=lambda f: f[0])[2]
    try:
        return pd.to_datetime(text, format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(text, format="mixed", dayfirst=True)


def _clock(raw: pd.Series) -> pd.Series:
    """Normalise a separate time column: 930 / 0930 / 093000 -> 09:30[:00]."""
    t = raw.astype(str).str.strip()
    if t.str.fullmatch(r"\d{1,6}").all():
        t = t.str.zfill(4 if t.str.len().max() <= 4 else 6)
        t = t.str[:2] + ":" + t.str[2:4] + np.where(t.str.len() == 6, ":" + t.str[4:6], "")
    return t


def load_csv(path: str | Path, tz: str = "UTC", timeframe: str | None = None) -> pd.DataFrame:
    """Read OHLCV bars from a CSV/TXT export (TradingView, Yahoo, NinjaTrader, Kibot, FirstRate,
    TradeStation, MetaTrader ...). Handles separate date and time columns, files without a
    header, and , ; tab or space separators.

    tz is the time zone the file's times are written in (used by the ICT session indicators).
    Times that carry their own offset are converted to UTC instead. timeframe (e.g. "5min",
    "15min", "1h", "4h", "1D") aggregates the bars after loading."""
    path = Path(path)
    sep, header = _sniff(path)
    df = pd.read_csv(path, sep=sep, header=0 if header else None, engine="python" if sep == r"\s+"
                     else "c", skipinitialspace=True)
    if header:
        df.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    else:
        # No header: date[, time], open, high, low, close[, volume]
        first = df.iloc[0].astype(str).tolist()
        has_clock = len(first) >= 6 and (":" in first[1] or (first[1].isdigit() and len(first[1]) <= 6
                                                            and len(first[0]) >= 8))
        names = (["date", "time"] if has_clock else ["datetime"]) + ["open", "high", "low", "close",
                                                                    "volume"]
        df = df.iloc[:, : len(names)]
        df.columns = names[: df.shape[1]]
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = df["date"].astype(str).str.strip() + " " + _clock(df["time"])
        df = df.drop(columns=["date", "time"])
    time_col = next((c for c in _TIME_NAMES if c in df.columns), None)
    if time_col:
        ts = df[time_col]
        if pd.api.types.is_numeric_dtype(ts) and ts.iloc[0] > 1e8:
            unit = "ms" if ts.iloc[0] > 1e11 else "s"
            df.index = pd.to_datetime(ts, unit=unit)
        else:
            parsed = _parse_times(ts)
            if getattr(parsed.dt, "tz", None) is not None:
                df.index = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
                tz = "UTC"
            else:
                df.index = parsed
        df = df.drop(columns=[time_col])
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Found: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])
    df = df[~df.index.duplicated()].sort_index()
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{path}: has zero or negative prices. Back-adjusted futures can go "
                         "negative far back in history; use ratio-adjusted or unadjusted data.")
    df.attrs["tz"] = tz
    if timeframe:
        df = resample(df, timeframe)
    if len(df) < 200:
        raise ValueError(f"{path}: only {len(df)} rows; need at least 200 bars to backtest")
    return df


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate bars to a bigger timeframe. Bars are labelled by their open time."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resampling needs a date/time column")
    try:
        rule = pd.tseries.frequencies.to_offset(timeframe.replace("m", "min")
                                                if timeframe.endswith("m") else timeframe)
    except ValueError as e:
        raise ValueError(f"unknown timeframe '{timeframe}', try 5min, 15min, 1h, 4h or 1D") from e
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out = out.dropna(subset=["close"])
    out.attrs["tz"] = df.attrs.get("tz", "UTC")
    return out


def fetch(symbol: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV bars from Yahoo Finance. Intraday intervals only go back a limited time."""
    import yfinance as yf  # optional dependency

    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for '{symbol}' (period={period}, "
                         f"interval={interval}). Check the symbol on finance.yahoo.com.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    df = raw[[c for c in PRICE_COLUMNS if c in raw.columns]].dropna(subset=["close"])
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.attrs["tz"] = "UTC"
    return df


def sample(n: int = 3000, seed: int = 7, freq: str = "B") -> pd.DataFrame:
    """Synthetic daily prices with trending and choppy regimes. Good for testing, not for profit."""
    rng = np.random.default_rng(seed)
    regime_len = rng.integers(60, 250, size=n // 60 + 1)
    drifts, vols = [], []
    for length in regime_len:
        drifts += [rng.choice([-0.0015, 0.0, 0.0012, 0.002])] * length
        vols += [rng.choice([0.008, 0.015, 0.025])] * length
    drift, vol = np.array(drifts[:n]), np.array(vols[:n])
    rets = drift + vol * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(vol * 0.3 * rng.standard_normal(n))
    open_[1:] = np.where(rng.random(n - 1) < 0.5, close[:-1], open_[1:])
    spread = np.abs(vol * rng.standard_normal(n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000, 10_000, size=n).astype(float)
    idx = (pd.bdate_range("2012-01-02", periods=n) if freq == "B"
           else pd.date_range("2024-01-01", periods=n, freq=freq))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume}, index=idx)
    df.attrs["tz"] = "UTC"
    return df


def bar_size(df: pd.DataFrame) -> str:
    """Typical bar length as text, e.g. '15min', '1h', '1d'."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 3:
        return "unknown"
    sec = float(pd.Series(df.index).diff().dt.total_seconds().median())
    for size, label in ((86400 * 7, "1w"), (86400, "1d"), (3600, "h"), (60, "min")):
        if sec >= size * 0.9:
            return label if size >= 86400 else f"{round(sec / size):g}{label}"
    return f"{sec:g}s"


def summary(df: pd.DataFrame, name: str = "data") -> str:
    """Short description the agents get so they know what market they are looking at."""
    rets = df["close"].pct_change().dropna()
    span = f"{df.index[0]} -> {df.index[-1]}" if isinstance(df.index, pd.DatetimeIndex) else ""
    size = bar_size(df)
    note = "" if size in ("unknown", "1d", "1w") or size.endswith("s") else \
        f" Times are {df.attrs.get('tz', 'UTC')}; session indicators convert to New York time."
    if size in ("1d", "1w"):
        note = " Daily or weekly bars: kill-zone/session indicators are meaningless here."
    return (f"{name}: {len(df)} bars of {size} {span}.{note} First close {df['close'].iloc[0]:.4g}, "
            f"last close {df['close'].iloc[-1]:.4g}, "
            f"mean bar return {rets.mean() * 100:.3f}%, bar volatility {rets.std() * 100:.3f}%.")
