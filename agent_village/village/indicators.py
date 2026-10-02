"""Technical indicators. Every function is causal: the value at bar t only uses bars <= t."""

import json

import numpy as np
import pandas as pd

from . import ict


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window means RSI is 100.
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"line": line, "signal": sig, "hist": line - sig})


def bbands(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    dev = close.rolling(period, min_periods=period).std()
    return pd.DataFrame({"upper": mid + std * dev, "mid": mid, "lower": mid - std * dev})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return close.pct_change(period) * 100


def stoch_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low = df["low"].rolling(period, min_periods=period).min()
    high = df["high"].rolling(period, min_periods=period).max()
    return 100 * (df["close"] - low) / (high - low).replace(0, np.nan)


def zscore(close: pd.Series, period: int = 20) -> pd.Series:
    mean = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return (close - mean) / std.replace(0, np.nan)


# name -> (parameter defaults, outputs). Outputs other than "value" are addressed as "<id>.<output>".
CATALOG = {
    "sma": ({"period": 20, "source": "close"}, ["value"]),
    "ema": ({"period": 20, "source": "close"}, ["value"]),
    "rsi": ({"period": 14}, ["value"]),
    "macd": ({"fast": 12, "slow": 26, "signal": 9}, ["line", "signal", "hist"]),
    "bbands": ({"period": 20, "std": 2.0}, ["upper", "mid", "lower"]),
    "atr": ({"period": 14}, ["value"]),
    "roc": ({"period": 10}, ["value"]),
    "stoch_k": ({"period": 14}, ["value"]),
    "zscore": ({"period": 20}, ["value"]),
}


def compute(df: pd.DataFrame, kind: str, params: dict) -> pd.DataFrame:
    """Compute one indicator. Returns a frame whose columns are the indicator's outputs."""
    if kind in ict.CATALOG:
        defaults, _, fn = ict.CATALOG[kind]
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"{kind}: unknown parameter(s) {sorted(unknown)}; use {defaults}")
        # Keep each parameter's type (k=3.0 from an LLM becomes 3).
        p = {k: type(v)(params.get(k, v)) for k, v in defaults.items()}
        return fn(df, **p)
    if kind not in CATALOG:
        known = ", ".join([*CATALOG, *ict.CATALOG])
        raise ValueError(f"unknown indicator '{kind}'. Known: {known}")
    p = {**CATALOG[kind][0], **params}
    close = df["close"]
    if kind in ("sma", "ema"):
        src = p["source"]
        if src not in df.columns:
            raise ValueError(f"{kind}: unknown source '{src}'")
        fn = sma if kind == "sma" else ema
        return fn(df[src], int(p["period"])).to_frame("value")
    if kind == "rsi":
        return rsi(close, int(p["period"])).to_frame("value")
    if kind == "macd":
        return macd(close, int(p["fast"]), int(p["slow"]), int(p["signal"]))
    if kind == "bbands":
        return bbands(close, int(p["period"]), float(p["std"]))
    if kind == "atr":
        return atr(df, int(p["period"])).to_frame("value")
    if kind == "roc":
        return roc(close, int(p["period"])).to_frame("value")
    if kind == "stoch_k":
        return stoch_k(df, int(p["period"])).to_frame("value")
    return zscore(close, int(p["period"])).to_frame("value")


class Cache:
    """Remembers recently computed indicators for ONE data frame (least recently used first out).

    The tuner and analyst re-test small variations of the same strategy, so most of their
    indicators are already computed. Keep one Cache per market and never share it between frames.
    """

    def __init__(self, max_mb: float = 600):
        self.max_bytes = max_mb * 1e6
        self.items: dict = {}
        self.bytes = 0

    def get(self, df: pd.DataFrame, kind: str, params: dict) -> pd.DataFrame:
        key = (kind, json.dumps(params, sort_keys=True, default=str))
        if key in self.items:
            self.items[key] = self.items.pop(key)  # mark as recently used
            return self.items[key]
        out = compute(df, kind, params)
        self.items[key] = out
        self.bytes += out.memory_usage(index=False).sum()
        while self.bytes > self.max_bytes and len(self.items) > 1:
            old = self.items.pop(next(iter(self.items)))
            self.bytes -= old.memory_usage(index=False).sum()
        return out


def catalog_text() -> str:
    lines = []
    entries = [*CATALOG.items(), *((n, (d, o)) for n, (d, o, _) in ict.CATALOG.items())]
    for name, (defaults, outputs) in entries:
        outs = "value" if outputs == ["value"] else ", ".join(f"<id>.{o}" for o in outputs)
        lines.append(f"- {name} params={defaults} outputs: {outs}")
    return "\n".join(lines)
