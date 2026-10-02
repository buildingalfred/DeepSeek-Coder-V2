"""Bar-by-bar backtester.

Rules are evaluated on a bar's close and filled at the NEXT bar's open, so a strategy can never
trade on information it would not have had yet. Fees are charged per unit of position change.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fast
from . import strategy as strat


@dataclass
class Result:
    train: dict
    test: dict
    full: dict
    trades: list = field(default_factory=list)
    equity: pd.Series | None = None
    returns: np.ndarray | None = None
    split: int = 0
    bars_per_year: float = 252.0


def bars_per_year(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
        if years > 30 / 365.25:
            return len(index) / years
    return 252.0


def run(df: pd.DataFrame, spec: dict, fee_bps: float = 5.0, train_frac: float = 0.7,
        cache=None) -> Result:
    """Backtest a spec. Metrics are reported separately for the train and the unseen test period.

    cache: optional indicators.Cache, so repeated backtests on the same data reuse indicators."""
    sig = strat.signals(df, spec, cache)
    open_ = df["open"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    fee = fee_bps / 10_000
    stop = (spec.get("stop_loss_pct") or 0) / 100
    take = (spec.get("take_profit_pct") or 0) / 100

    rets, pos, side, entry_i, exit_i, pnl, open_trade = fast.simulate(
        open_, close, sig["entry_long"], sig["exit_long"], sig["entry_short"], sig["exit_short"],
        fee, stop, take)
    trades = [{"side": "long" if side[i] == 1 else "short", "entry_i": int(entry_i[i]),
               "exit_i": int(exit_i[i]), "pnl_pct": round(float(pnl[i]) * 100, 3)}
              for i in range(len(side))]
    if open_trade:
        trades[-1]["open"] = True

    split = int(n * train_frac)
    bpy = bars_per_year(df.index)
    buy_hold = df["close"].pct_change().fillna(0).to_numpy()
    equity = pd.Series(np.cumprod(1 + rets), index=df.index)
    return Result(
        train=metrics(rets[:split], pos[:split], buy_hold[:split], trades, 0, split, bpy),
        test=metrics(rets[split:], pos[split:], buy_hold[split:], trades, split, n, bpy),
        full=metrics(rets, pos, buy_hold, trades, 0, n, bpy),
        trades=trades,
        equity=equity,
        returns=rets,
        split=split,
        bars_per_year=bpy,
    )


def metrics(rets, pos, buy_hold, trades, lo, hi, bpy) -> dict:
    rets = np.asarray(rets, float)
    if len(rets) == 0:
        return {}
    eq = np.cumprod(1 + rets)
    total = float(eq[-1] - 1)
    years = len(rets) / bpy
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else -1.0
    std = rets.std()
    sharpe = rets.mean() / std * np.sqrt(bpy) if std > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())
    mine = [t for t in trades if lo <= t["entry_i"] < hi]
    wins = [t["pnl_pct"] for t in mine if t["pnl_pct"] > 0]
    losses = [t["pnl_pct"] for t in mine if t["pnl_pct"] <= 0]
    pf = sum(wins) / -sum(losses) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    return {
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(float(cagr) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        # Sharpe x sqrt(years): how many standard errors from zero. Around 2+ is hard to get by luck
        # for ONE strategy; picking the best of many needs more (see luck_bar).
        "t_stat": round(float(sharpe) * float(np.sqrt(max(years, 1e-9))), 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trades": len(mine),
        "win_rate_pct": round(100 * len(wins) / len(mine), 1) if mine else 0.0,
        "profit_factor": round(float(pf), 2) if np.isfinite(pf) else 999.0,
        "exposure_pct": round(100 * float(np.mean(pos != 0)), 1),
        "buy_hold_pct": round(float(np.prod(1 + buy_hold) - 1) * 100, 2),
    }


def chunk_sharpes(result: Result, chunks: int = 3) -> list[float]:
    """Sharpe of each equal slice of the train period. A real edge should show up in most of them."""
    train = result.returns[: result.split]
    out = []
    for piece in np.array_split(train, chunks):
        std = piece.std()
        out.append(float(piece.mean() / std * np.sqrt(result.bars_per_year)) if std > 0 else 0.0)
    return out


def evaluate(markets: dict, spec: dict, fee_bps: float = 5.0, train_frac: float = 0.7,
             caches: dict | None = None) -> dict:
    """Backtest one spec on every market and combine the results.

    Returns {"train": {...}, "test": {...}, "markets": {name: {"train", "test"}}}. The combined
    train metrics include "robust_sharpe" (see robust_sharpe) which is what the leaderboard ranks.
    """
    caches = caches or {}
    results = {name: run(df, spec, fee_bps, train_frac, caches.get(name))
               for name, df in markets.items()}
    pieces = [s for r in results.values() for s in chunk_sharpes(r)]
    train = combine([r.train for r in results.values()])
    train["robust_sharpe"] = robust_sharpe(pieces)
    train["positive_periods"] = f"{sum(s > 0 for s in pieces)}/{len(pieces)}"
    return {
        "train": train,
        "test": combine([r.test for r in results.values()]),
        "markets": {n: {"train": r.train, "test": r.test} for n, r in results.items()},
    }


def robust_sharpe(pieces: list[float]) -> float:
    """Average Sharpe over slices of history, minus a penalty when they disagree.

    A strategy that earns everything in one lucky stretch scores lower than one that works
    a little in every stretch and every market.
    """
    arr = np.asarray(pieces, float)
    return round(float(arr.mean() - 0.5 * arr.std()), 3)


def combine(ms: list[dict]) -> dict:
    """Average per-market metrics; trades are summed and the drawdown is the worst one."""
    ms = [m for m in ms if m]
    if not ms:
        return {}
    if len(ms) == 1:
        return dict(ms[0])
    out = {k: round(float(np.mean([m[k] for m in ms])), 3) for k in ms[0]}
    out["trades"] = int(sum(m["trades"] for m in ms))
    out["max_drawdown_pct"] = min(m["max_drawdown_pct"] for m in ms)
    return out


def luck_bar(trials: int) -> float:
    """t-stat the best of `trials` random strategies would reach by pure luck (about sqrt(2 ln N))."""
    return round(float(np.sqrt(2 * np.log(max(trials, 2)))), 2)


MIN_TRADES_PER_MARKET = 10


def score(evaluation: dict, n_markets: int = 1) -> float:
    """Leaderboard score from the TRAIN period only, so the test period stays an honest check."""
    m = evaluation["train"]
    if not m or m["trades"] < MIN_TRADES_PER_MARKET * n_markets:
        return -99.0
    return m["robust_sharpe"]
