"""Hot loops, compiled with numba when it is installed (pip install numba), plain Python otherwise.

The results are identical either way; numba only makes them roughly 50-100x faster, which matters
for years of intraday futures data.
"""

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def simulate(open_, close, entry_long, exit_long, entry_short, exit_short, fee, stop, take):
    """Bar-by-bar position engine. Rules are read at the close of bar t-1 and filled at the open
    of bar t. Returns per-bar returns, positions and the trades as parallel arrays."""
    n = len(close)
    pos = np.zeros(n)
    rets = np.zeros(n)
    t_side = np.zeros(n, np.int64)
    t_entry = np.zeros(n, np.int64)
    t_exit = np.zeros(n, np.int64)
    t_pnl = np.zeros(n)
    nt = 0
    cur = 0
    entry_px = 0.0
    entry_i = -1
    for t in range(1, n):
        target = cur
        p = t - 1
        if cur != 0:
            move = (close[p] / entry_px - 1.0) * cur
            hit = (stop > 0 and move <= -stop) or (take > 0 and move >= take)
            if cur == 1 and (exit_long[p] or entry_short[p] or hit):
                target = -1 if (entry_short[p] and not hit) else 0
            elif cur == -1 and (exit_short[p] or entry_long[p] or hit):
                target = 1 if (entry_long[p] and not hit) else 0
        elif entry_long[p]:
            target = 1
        elif entry_short[p]:
            target = -1
        # Overnight gap belongs to the old position, the bar's own move to the new one.
        gap = open_[t] / close[p] - 1.0
        intrabar = close[t] / open_[t] - 1.0
        rets[t] = (1.0 + cur * gap) * (1.0 + target * intrabar) - 1.0 - fee * abs(target - cur)
        if target != cur:
            if cur != 0:
                t_side[nt] = cur
                t_entry[nt] = entry_i
                t_exit[nt] = t
                t_pnl[nt] = (open_[t] / entry_px - 1.0) * cur - 2.0 * fee
                nt += 1
            if target != 0:
                entry_px = open_[t]
                entry_i = t
            cur = target
        pos[t] = cur
    open_trade = 0
    if cur != 0:
        t_side[nt] = cur
        t_entry[nt] = entry_i
        t_exit[nt] = n - 1
        t_pnl[nt] = (close[n - 1] / entry_px - 1.0) * cur - 2.0 * fee
        nt += 1
        open_trade = 1
    return rets, pos, t_side[:nt], t_entry[:nt], t_exit[:nt], t_pnl[:nt], open_trade


@njit(cache=True)
def fvg_zones(h, l, c, atr, min_atr):
    n = len(c)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)
    bull_new = np.zeros(n)
    bear_new = np.zeros(n)
    bt = np.nan
    bb = np.nan
    st = np.nan
    sb = np.nan
    for t in range(n):
        # Gaps must be at least min_atr x ATR wide (no filter when min_atr is 0).
        need = min_atr * atr[t] if min_atr > 0 else 0.0
        if t >= 2 and not np.isnan(need):
            if l[t] - h[t - 2] > need:
                bt = l[t]
                bb = h[t - 2]
                bull_new[t] = 1.0
            if l[t - 2] - h[t] > need:
                st = l[t - 2]
                sb = h[t]
                bear_new[t] = 1.0
        if not np.isnan(bb) and c[t] < bb:  # closed through: the gap failed
            bt = np.nan
            bb = np.nan
        if not np.isnan(st) and c[t] > st:
            st = np.nan
            sb = np.nan
        bull_top[t] = bt
        bull_bot[t] = bb
        bear_top[t] = st
        bear_bot[t] = sb
    return bull_new, bear_new, bull_top, bull_bot, bear_top, bear_bot


@njit(cache=True)
def order_block_zones(o, h, l, c, up, down, lookback):
    n = len(c)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)
    bt = np.nan
    bb = np.nan
    rt = np.nan
    rb = np.nan
    for t in range(n):
        if up[t] > 0:
            for j in range(t - 1, max(-1, t - 1 - lookback), -1):
                if c[j] < o[j]:
                    bt = h[j]
                    bb = l[j]
                    break
        if down[t] > 0:
            for j in range(t - 1, max(-1, t - 1 - lookback), -1):
                if c[j] > o[j]:
                    rt = h[j]
                    rb = l[j]
                    break
        if not np.isnan(bb) and c[t] < bb:
            bt = np.nan
            bb = np.nan
        if not np.isnan(rt) and c[t] > rt:
            rt = np.nan
            rb = np.nan
        bull_top[t] = bt
        bull_bot[t] = bb
        bear_top[t] = rt
        bear_bot[t] = rb
    return bull_top, bull_bot, bear_top, bear_bot
