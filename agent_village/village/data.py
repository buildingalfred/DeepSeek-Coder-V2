"""Loading price data (OHLCV) from CSV, plus a synthetic sample for trying the village offline."""

from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import PRICE_COLUMNS

_ALIASES = {
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    "adj close": "adj_close", "adj_close": "adj_close", "vol": "volume",
}
_TIME_NAMES = ("date", "datetime", "time", "timestamp", "open_time", "index")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a date/time column and open, high, low, close (volume optional)."""
    df = pd.read_csv(path)
    df.columns = [_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    time_col = next((c for c in _TIME_NAMES if c in df.columns), None)
    if time_col:
        ts = df[time_col]
        if pd.api.types.is_numeric_dtype(ts):
            unit = "ms" if ts.iloc[0] > 1e11 else "s"
            df.index = pd.to_datetime(ts, unit=unit)
        else:
            df.index = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
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
    if len(df) < 200:
        raise ValueError(f"{path}: only {len(df)} rows; need at least 200 bars to backtest")
    return df


def sample(n: int = 3000, seed: int = 7) -> pd.DataFrame:
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
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume}, index=idx)


def summary(df: pd.DataFrame, name: str = "data") -> str:
    """Short description the agents get so they know what market they are looking at."""
    rets = df["close"].pct_change().dropna()
    span = f"{df.index[0]} -> {df.index[-1]}" if isinstance(df.index, pd.DatetimeIndex) else ""
    return (f"{name}: {len(df)} bars {span}. First close {df['close'].iloc[0]:.4g}, "
            f"last close {df['close'].iloc[-1]:.4g}, "
            f"mean bar return {rets.mean() * 100:.3f}%, bar volatility {rets.std() * 100:.3f}%.")
