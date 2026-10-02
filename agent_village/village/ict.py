"""ICT (Inner Circle Trader) concepts as testable, causal indicators.

Each function only uses bars up to and including the current one. Swing points need `k` bars on
each side, so a swing is only *known* k bars after it happened: that delay is built in, so
strategies cannot accidentally peek at the future.

Flags are 1.0 / 0.0. Zone levels are NaN when no zone is active.
"""

import numpy as np
import pandas as pd

from . import fast

NY = "America/New_York"


def swings(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Last confirmed swing high / swing low (fractal of k bars on each side)."""
    win = 2 * k + 1
    hi_c = df["high"].shift(k)
    lo_c = df["low"].shift(k)
    is_hi = hi_c == df["high"].rolling(win, min_periods=win).max()
    is_lo = lo_c == df["low"].rolling(win, min_periods=win).min()
    return pd.DataFrame({"high": hi_c.where(is_hi).ffill(), "low": lo_c.where(is_lo).ffill()})


def structure(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Break of structure (close beyond the last swing) and market structure shift (a break
    against the prevailing trend). trend is +1 after an up-break, -1 after a down-break."""
    sw = swings(df, k)
    hi, lo = sw["high"].shift(), sw["low"].shift()
    close = df["close"]
    up = (close > hi) & (close.shift() <= hi)
    down = (close < lo) & (close.shift() >= lo)
    trend = pd.Series(np.where(up, 1.0, np.where(down, -1.0, np.nan)), index=df.index).ffill()
    prev = trend.shift()
    return pd.DataFrame({
        "bos_up": up.astype(float), "bos_down": down.astype(float),
        "mss_up": (up & (prev == -1)).astype(float), "mss_down": (down & (prev == 1)).astype(float),
        "trend": trend.fillna(0.0),
    })


def sweep(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Liquidity sweep / stop hunt: price trades through the last swing low (high) but closes
    back inside. 'bull' = sell-side liquidity taken (bullish), 'bear' = buy-side taken."""
    sw = swings(df, k)
    lo, hi = sw["low"].shift(), sw["high"].shift()
    bull = (df["low"] < lo) & (df["close"] > lo)
    bear = (df["high"] > hi) & (df["close"] < hi)
    return pd.DataFrame({"bull": bull.astype(float), "bear": bear.astype(float)})


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def fvg(df: pd.DataFrame, min_atr: float = 0.0) -> pd.DataFrame:
    """Fair value gaps (3-candle imbalances). bull/bear flag the bar a gap forms; the *_top/*_bot
    columns hold the most recent gap that price has not closed through yet."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    out = fast.fvg_zones(h, l, c, _atr(df).to_numpy(float), float(min_atr))
    names = ["bull", "bear", "bull_top", "bull_bot", "bear_top", "bear_bot"]
    return pd.DataFrame(dict(zip(names, out)), index=df.index)


def order_block(df: pd.DataFrame, k: int = 3, lookback: int = 10) -> pd.DataFrame:
    """Order blocks: when structure breaks up, the last down-close candle before the break becomes
    a bullish zone (and the mirror for bearish). A zone dies when price closes through it."""
    st = structure(df, k)
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    out = fast.order_block_zones(o, h, l, c, st["bos_up"].to_numpy(float),
                                 st["bos_down"].to_numpy(float), int(lookback))
    names = ["bull_top", "bull_bot", "bear_top", "bear_bot"]
    return pd.DataFrame(dict(zip(names, out)), index=df.index)


def premium_discount(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Where the close sits inside the dealing range (last swing low -> last swing high).
    0 = at the low, 1 = at the high. Below 0.5 is discount, above is premium. ICT's optimal
    trade entry (OTE) for longs is roughly 0.21-0.38 (the 62-79% retracement)."""
    sw = swings(df, k)
    rng = (sw["high"] - sw["low"]).where(lambda r: r > 0)
    pos = (df["close"] - sw["low"]) / rng
    ote_long = pos.between(0.21, 0.38)
    ote_short = pos.between(0.62, 0.79)
    return pd.DataFrame({"pos": pos, "ote_long": ote_long.astype(float),
                         "ote_short": ote_short.astype(float)})


def displacement(df: pd.DataFrame, mult: float = 1.5, period: int = 14) -> pd.DataFrame:
    """Strong, energetic candles: body bigger than mult x ATR."""
    body = (df["close"] - df["open"]).abs()
    big = body > mult * _atr(df, period)
    return pd.DataFrame({"up": (big & (df["close"] > df["open"])).astype(float),
                         "down": (big & (df["close"] < df["open"])).astype(float)})


def _ny_time(df: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("needs a date/time column in the data")
    tz = df.attrs.get("tz", "UTC")
    idx = df.index if df.index.tz is not None else df.index.tz_localize(tz, ambiguous="NaT",
                                                                         nonexistent="NaT")
    return idx.tz_convert(NY)


def session(df: pd.DataFrame) -> pd.DataFrame:
    """ICT kill zones in New York time (needs intraday data). Bars are labelled by their open
    time. asia 20:00-24:00, london 02:00-05:00, ny_am 07:00-10:00, silver_bullet 10:00-11:00,
    ny_pm 13:30-16:00. Also the New York hour (e.g. 9.5 = 09:30) and weekday (0 = Monday)."""
    ny = _ny_time(df)
    hour = pd.Series(ny.hour + ny.minute / 60, index=df.index).astype(float)
    def box(a, b):
        return ((hour >= a) & (hour < b)).astype(float)
    return pd.DataFrame({
        "hour": hour, "weekday": pd.Series(ny.weekday, index=df.index).astype(float),
        "asia": box(20, 24), "london": box(2, 5), "ny_am": box(7, 10),
        "silver_bullet": box(10, 11), "ny_pm": box(13.5, 16),
    })


def prev_day(df: pd.DataFrame) -> pd.DataFrame:
    """Previous New York trading day's high and low (classic liquidity pools), and today's open.
    On daily data this is simply the previous bar's high/low."""
    ny = _ny_time(df)
    day = pd.Series(ny.normalize().tz_localize(None), index=df.index)
    highs = df["high"].groupby(day.values).max()
    lows = df["low"].groupby(day.values).min()
    opens = df["open"].groupby(day.values).first()
    return pd.DataFrame({
        "high": day.map(highs.shift(1)).to_numpy(float),
        "low": day.map(lows.shift(1)).to_numpy(float),
        "open": day.map(opens).to_numpy(float),   # today's first open is known from bar one
    }, index=df.index)


# name -> (defaults, outputs, function)
CATALOG = {
    "swings": ({"k": 3}, ["high", "low"], swings),
    "structure": ({"k": 3}, ["bos_up", "bos_down", "mss_up", "mss_down", "trend"], structure),
    "sweep": ({"k": 3}, ["bull", "bear"], sweep),
    "fvg": ({"min_atr": 0.0}, ["bull", "bear", "bull_top", "bull_bot", "bear_top", "bear_bot"], fvg),
    "order_block": ({"k": 3, "lookback": 10}, ["bull_top", "bull_bot", "bear_top", "bear_bot"],
                    order_block),
    "premium_discount": ({"k": 3}, ["pos", "ote_long", "ote_short"], premium_discount),
    "displacement": ({"mult": 1.5, "period": 14}, ["up", "down"], displacement),
    "session": ({}, ["hour", "weekday", "asia", "london", "ny_am", "silver_bullet", "ny_pm"],
                session),
    "prev_day": ({}, ["high", "low", "open"], prev_day),
}

GUIDE = """ICT building blocks (flags are 1 or 0, so test them with "==" 1; zone levels are empty
when no zone is active, which makes any comparison with them false):
- sweep.bull / sweep.bear: liquidity grab, price ran a swing low/high and closed back inside
- structure.bos_up/bos_down: close broke the last swing; mss_up/mss_down: break against the trend
- fvg.bull/bear (gap formed this bar); fvg.bull_top/bull_bot, fvg.bear_top/bear_bot (live gap)
- order_block.bull_top/bull_bot, bear_top/bear_bot: last opposite candle before a structure break
- premium_discount.pos (0 = range low, 1 = range high), ote_long / ote_short flags
- displacement.up/down: candle body bigger than mult x ATR
- session.london / ny_am / silver_bullet / ny_pm / asia flags, session.hour (New York time),
  session.weekday. Only meaningful on intraday data.
- prev_day.high / low (yesterday's liquidity), prev_day.open (today's open)
Use "within": N on a condition to mean "this was true at some point in the last N bars". That is
how you chain an ICT sequence, for example:
  sweep.bull == 1 within 20  ->  structure.mss_up == 1 within 10  ->  low <= fvg.bull_top now."""
