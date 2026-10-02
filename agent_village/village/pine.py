"""Export a village strategy to TradingView Pine Script (v5).

The script reproduces the village's backtester as closely as Pine allows:
- every indicator is written out with the same formula (same warm-up, same missing-value rules),
- rules are checked at the bar close and orders fill at the next bar's open,
- stop loss / take profit are checked on the close, like the village does (not intrabar),
- fees are charged as a percent per side.

So the Strategy Tester in TradingView should show the same trades. Position sizing differs
(TradingView rounds to whole contracts on futures), so compare trades and percentages rather
than dollar amounts.
"""

import json
import re

from . import features, ict, indicators
from . import strategy as strat

PRICE = {c: c for c in strat.PRICE_COLUMNS}

HELPERS = """
// ---- helpers: comparisons are false when either side is missing (same as the village) ----
f_gt(a, b) => not na(a) and not na(b) and a > b
f_lt(a, b) => not na(a) and not na(b) and a < b
f_ge(a, b) => not na(a) and not na(b) and a >= b
f_le(a, b) => not na(a) and not na(b) and a <= b
f_eq(a, b) => not na(a) and not na(b) and a == b
f_xup(a, b) => f_gt(a, b) and f_le(a[1], b[1])
f_xdn(a, b) => f_lt(a, b) and f_ge(a[1], b[1])
// true if cond held on any of the last n bars, this one included
f_within(cond, n) => nz(ta.barssince(cond), 1000000000) < n
// exponential average that starts at the first value and needs minp values (pandas ewm, adjust=False)
f_ewm(x, alpha, minp) =>
    var float e = na
    var int cnt = 0
    if not na(x)
        e := na(e) ? x : alpha * x + (1 - alpha) * e
        cnt += 1
    cnt >= minp ? e : na
f_tr() => na(close[1]) ? high - low : math.max(high - low, math.abs(high - close[1]), math.abs(low - close[1]))
f_atr(n) => f_ewm(f_tr(), 1.0 / n, n)
"""

OP_FN = {">": "f_gt", "<": "f_lt", ">=": "f_ge", "<=": "f_le", "==": "f_eq",
         "crosses_above": "f_xup", "crosses_below": "f_xdn"}


def _var(ind_id: str, output: str = "value") -> str:
    return f"i_{ind_id}" if output == "value" else f"i_{ind_id}_{output}"


def _params(kind: str, ind: dict) -> dict:
    if kind in ict.CATALOG:
        defaults = ict.CATALOG[kind][0]
        return {k: type(v)(ind.get(k, v)) for k, v in defaults.items()}
    defaults = indicators.CATALOG[kind][0]
    return {**defaults, **{k: v for k, v in ind.items() if k not in ("id", "type")}}


def _num(x) -> str:
    return repr(float(x)) if isinstance(x, float) else str(x)


def _indicator_code(ind: dict) -> str:
    i, kind = ind["id"], ind["type"]
    p = _params(kind, ind)
    v = lambda out="value": _var(i, out)  # noqa: E731
    if kind in ("sma", "ema"):
        n, src = int(p["period"]), PRICE[p["source"]]
        if kind == "sma":
            return f"{v()} = ta.sma({src}, {n})"
        return f"{v()} = f_ewm({src}, 2.0 / ({n} + 1), {n})"
    if kind == "rsi":
        n = int(p["period"])
        return (f"{i}_d = close - close[1]\n"
                f"{i}_g = f_ewm(na({i}_d) ? na : math.max({i}_d, 0), 1.0 / {n}, {n})\n"
                f"{i}_l = f_ewm(na({i}_d) ? na : math.max(-{i}_d, 0), 1.0 / {n}, {n})\n"
                f"{v()} = na({i}_g) or na({i}_l) ? na : {i}_l == 0 ? 100.0 : "
                f"100 - 100 / (1 + {i}_g / {i}_l)")
    if kind == "macd":
        f, s, g = int(p["fast"]), int(p["slow"]), int(p["signal"])
        return (f"{v('line')} = f_ewm(close, 2.0 / ({f} + 1), {f}) - f_ewm(close, 2.0 / ({s} + 1), {s})\n"
                f"{v('signal')} = f_ewm({v('line')}, 2.0 / ({g} + 1), {g})\n"
                f"{v('hist')} = {v('line')} - {v('signal')}")
    if kind == "bbands":
        n, sd = int(p["period"]), float(p["std"])
        return (f"{v('mid')} = ta.sma(close, {n})\n"
                f"{i}_dev = ta.stdev(close, {n}, false)\n"
                f"{v('upper')} = {v('mid')} + {sd} * {i}_dev\n"
                f"{v('lower')} = {v('mid')} - {sd} * {i}_dev")
    if kind == "atr":
        return f"{v()} = f_atr({int(p['period'])})"
    if kind == "roc":
        n = int(p["period"])
        return f"{v()} = (close / close[{n}] - 1) * 100"
    if kind == "stoch_k":
        n = int(p["period"])
        return (f"{i}_lo = ta.lowest(low, {n})\n{i}_hi = ta.highest(high, {n})\n"
                f"{v()} = bar_index < {n - 1} or {i}_hi == {i}_lo ? na : "
                f"100 * (close - {i}_lo) / ({i}_hi - {i}_lo)")
    if kind == "zscore":
        n = int(p["period"])
        return (f"{i}_m = ta.sma(close, {n})\n{i}_s = ta.stdev(close, {n}, false)\n"
                f"{v()} = na({i}_s) or {i}_s == 0 ? na : (close - {i}_m) / {i}_s")
    return _ict_code(i, kind, p, v)


def _swings_code(i: str, k: int) -> str:
    return (f"var float {i}_sh = na\nvar float {i}_sl = na\n"
            f"if bar_index >= {2 * k} and high[{k}] == ta.highest(high, {2 * k + 1})\n"
            f"    {i}_sh := high[{k}]\n"
            f"if bar_index >= {2 * k} and low[{k}] == ta.lowest(low, {2 * k + 1})\n"
            f"    {i}_sl := low[{k}]")


def _structure_code(i: str, k: int) -> str:
    return (_swings_code(i, k) + "\n"
            f"{i}_up = f_gt(close, {i}_sh[1]) and f_le(close[1], {i}_sh[1])\n"
            f"{i}_dn = f_lt(close, {i}_sl[1]) and f_ge(close[1], {i}_sl[1])\n"
            f"var float {i}_tr = na\n{i}_prev = {i}_tr\n"
            f"if {i}_up\n    {i}_tr := 1\nelse if {i}_dn\n    {i}_tr := -1")


def _ict_code(i: str, kind: str, p: dict, v) -> str:
    if kind == "swings":
        return _swings_code(i, p["k"]) + f"\n{v('high')} = {i}_sh\n{v('low')} = {i}_sl"
    if kind == "structure":
        return (_structure_code(i, p["k"]) + "\n"
                f"{v('bos_up')} = {i}_up ? 1.0 : 0.0\n{v('bos_down')} = {i}_dn ? 1.0 : 0.0\n"
                f"{v('mss_up')} = {i}_up and {i}_prev == -1 ? 1.0 : 0.0\n"
                f"{v('mss_down')} = {i}_dn and {i}_prev == 1 ? 1.0 : 0.0\n"
                f"{v('trend')} = nz({i}_tr)")
    if kind == "sweep":
        return (_swings_code(i, p["k"]) + "\n"
                f"{v('bull')} = f_lt(low, {i}_sl[1]) and f_gt(close, {i}_sl[1]) ? 1.0 : 0.0\n"
                f"{v('bear')} = f_gt(high, {i}_sh[1]) and f_lt(close, {i}_sh[1]) ? 1.0 : 0.0")
    if kind == "fvg":
        m = float(p["min_atr"])
        return (f"{i}_atr = f_atr(14)\n{i}_need = {m} > 0 ? {m} * {i}_atr : 0.0\n"
                f"var float {i}_bt = na\nvar float {i}_bb = na\nvar float {i}_st = na\n"
                f"var float {i}_sb = na\n{v('bull')} = 0.0\n{v('bear')} = 0.0\n"
                f"if bar_index >= 2 and not na({i}_need)\n"
                f"    if low - high[2] > {i}_need\n"
                f"        {i}_bt := low\n        {i}_bb := high[2]\n        {v('bull')} := 1.0\n"
                f"    if low[2] - high > {i}_need\n"
                f"        {i}_st := low[2]\n        {i}_sb := high\n        {v('bear')} := 1.0\n"
                f"if not na({i}_bb) and close < {i}_bb\n    {i}_bt := na\n    {i}_bb := na\n"
                f"if not na({i}_st) and close > {i}_st\n    {i}_st := na\n    {i}_sb := na\n"
                f"{v('bull_top')} = {i}_bt\n{v('bull_bot')} = {i}_bb\n"
                f"{v('bear_top')} = {i}_st\n{v('bear_bot')} = {i}_sb")
    if kind == "order_block":
        k, lb = p["k"], p["lookback"]
        return (_structure_code(i, k) + "\n"
                f"var float {i}_bt = na\nvar float {i}_bb = na\nvar float {i}_rt = na\n"
                f"var float {i}_rb = na\n"
                f"if {i}_up\n    for j = 1 to {lb}\n        if j > bar_index\n            break\n"
                f"        if close[j] < open[j]\n            {i}_bt := high[j]\n"
                f"            {i}_bb := low[j]\n            break\n"
                f"if {i}_dn\n    for j = 1 to {lb}\n        if j > bar_index\n            break\n"
                f"        if close[j] > open[j]\n            {i}_rt := high[j]\n"
                f"            {i}_rb := low[j]\n            break\n"
                f"if not na({i}_bb) and close < {i}_bb\n    {i}_bt := na\n    {i}_bb := na\n"
                f"if not na({i}_rt) and close > {i}_rt\n    {i}_rt := na\n    {i}_rb := na\n"
                f"{v('bull_top')} = {i}_bt\n{v('bull_bot')} = {i}_bb\n"
                f"{v('bear_top')} = {i}_rt\n{v('bear_bot')} = {i}_rb")
    if kind == "premium_discount":
        return (_swings_code(i, p["k"]) + "\n"
                f"{i}_rng = {i}_sh - {i}_sl\n"
                f"{v('pos')} = na({i}_rng) or {i}_rng <= 0 ? na : (close - {i}_sl) / {i}_rng\n"
                f"{v('ote_long')} = f_ge({v('pos')}, 0.21) and f_le({v('pos')}, 0.38) ? 1.0 : 0.0\n"
                f"{v('ote_short')} = f_ge({v('pos')}, 0.62) and f_le({v('pos')}, 0.79) ? 1.0 : 0.0")
    if kind == "displacement":
        mult, n = float(p["mult"]), int(p["period"])
        return (f"{i}_big = f_gt(math.abs(close - open), {mult} * f_atr({n}))\n"
                f"{v('up')} = {i}_big and close > open ? 1.0 : 0.0\n"
                f"{v('down')} = {i}_big and close < open ? 1.0 : 0.0")
    if kind == "session":
        h = f"{i}_h"
        box = lambda a, b: f"{h} >= {a} and {h} < {b} ? 1.0 : 0.0"  # noqa: E731
        return (f'{h} = hour(time, "America/New_York") + minute(time, "America/New_York") / 60.0\n'
                f"{v('hour')} = {h}\n"
                f'{v("weekday")} = (dayofweek(time, "America/New_York") + 5) % 7\n'
                f"{v('asia')} = {box(20, 24)}\n{v('london')} = {box(2, 5)}\n"
                f"{v('ny_am')} = {box(7, 10)}\n{v('silver_bullet')} = {box(10, 11)}\n"
                f"{v('ny_pm')} = {box(13.5, 16)}")
    if kind == "prev_day":
        return (f'{i}_day = dayofmonth(time, "America/New_York")\n'
                f"{i}_new = na({i}_day[1]) or {i}_day != {i}_day[1]\n"
                f"var float {i}_ch = na\nvar float {i}_cl = na\nvar float {i}_ph = na\n"
                f"var float {i}_pl = na\nvar float {i}_op = na\n"
                f"if {i}_new\n    {i}_ph := {i}_ch\n    {i}_pl := {i}_cl\n"
                f"    {i}_ch := high\n    {i}_cl := low\n    {i}_op := open\n"
                f"else\n    {i}_ch := math.max({i}_ch, high)\n    {i}_cl := math.min({i}_cl, low)\n"
                f"{v('high')} = {i}_ph\n{v('low')} = {i}_pl\n{v('open')} = {i}_op")
    raise ValueError(f"no Pine translation for indicator '{kind}'")


def _operand(x) -> str:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _num(x)
    if x in PRICE:
        return x
    ind_id, _, out = x.partition(".")
    return _var(ind_id, out or "value")


def _condition(c: dict) -> str:
    expr = f"{OP_FN[c['op']]}({_operand(c['left'])}, {_operand(c['right'])})"
    if c.get("within") and c["within"] > 1:
        expr = f"f_within({expr}, {int(c['within'])})"
    return expr


def _rule(conds: list) -> str:
    return " and ".join(_condition(c) for c in conds) if conds else "false"


def _title(text: str) -> str:
    return re.sub(r'["\\\n]', "'", str(text))[:80]


def to_pine(spec: dict, fee_bps: float = 5.0, stats: dict | None = None) -> str:
    """Return a Pine Script v5 strategy for this spec. The spec must already be valid."""
    for ind in spec.get("indicators") or []:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(ind.get("id", ""))):
            raise ValueError(f"bad indicator id {ind.get('id')!r}")
    name = _title(spec.get("name", "Village strategy"))
    stop = (spec.get("stop_loss_pct") or 0) / 100
    take = (spec.get("take_profit_pct") or 0) / 100
    head = ["//@version=5",
            f"// {name}",
            f"// Idea: {_title(spec.get('idea', ''))}",
            f"// Rules: {strat.describe(spec)}",
            "// Generated by Agent Village. Signals at the bar close, fills at the next bar open,",
            "// stop and target checked on the close: the same logic as the village backtester."]
    if stats:
        for label, m in stats.items():
            if m:
                head.append(f"// Village {label}: sharpe {m.get('sharpe')}, t-stat {m.get('t_stat')}, "
                            f"return {m.get('total_return_pct')}%, trades {m.get('trades')}, "
                            f"win rate {m.get('win_rate_pct')}%, max drawdown {m.get('max_drawdown_pct')}%")
    head.append(
        f'strategy("{name}", overlay=true, initial_capital=100000, '
        "default_qty_type=strategy.percent_of_equity, default_qty_value=100, "
        f"commission_type=strategy.commission.percent, commission_value={fee_bps / 100}, "
        "pyramiding=0, process_orders_on_close=false, calc_on_every_tick=false)")
    body = [HELPERS.strip(), "", "// ---- indicators ----"]
    for ind in spec.get("indicators") or []:
        body.append(f"// {ind['id']}: {json.dumps({k: v for k, v in ind.items() if k != 'id'})}")
        body.append(_indicator_code(ind))
    feats = spec.get("features") or []
    if feats:
        body.append("")
        body.append("// ---- invented features ----")
    for feat in feats:
        writer = features.PineWriter(_operand, f"ft_{feat['id']}")
        result = writer.write(feat["expr"])
        body.append(f"// {feat['id']} = {_title(feat['expr'])}")
        body += writer.lines
        body.append(f"{_var(feat['id'])} = {result}")
    body += ["", "// ---- rules ----",
             f"entryLong = {_rule(spec.get('entry_long'))}",
             f"exitLong = {_rule(spec.get('exit_long'))}",
             f"entryShort = {_rule(spec.get('entry_short'))}",
             f"exitShort = {_rule(spec.get('exit_short'))}",
             "",
             "// ---- position logic (same as the village) ----",
             f"stopPct = {stop}",
             f"takePct = {take}",
             "pos = strategy.position_size > 0 ? 1 : strategy.position_size < 0 ? -1 : 0",
             "move = pos != 0 ? (close / strategy.position_avg_price - 1) * pos : 0.0",
             "hit = pos != 0 and ((stopPct > 0 and move <= -stopPct) or (takePct > 0 and move >= takePct))",
             "target = pos",
             "if pos == 1 and (exitLong or entryShort or hit)",
             "    target := entryShort and not hit ? -1 : 0",
             "else if pos == -1 and (exitShort or entryLong or hit)",
             "    target := entryLong and not hit ? 1 : 0",
             "else if pos == 0",
             "    target := entryLong ? 1 : entryShort ? -1 : 0",
             "if target != pos",
             "    if target == 1",
             '        strategy.entry("Long", strategy.long)',
             "    else if target == -1",
             '        strategy.entry("Short", strategy.short)',
             "    else",
             '        strategy.close_all(comment="Exit")',
             "",
             "// ---- setups on the chart and alerts ----",
             'plotshape(entryLong, "Long setup", shape.triangleup, location.belowbar, '
             'color.new(color.teal, 0), size=size.small)',
             'plotshape(entryShort, "Short setup", shape.triangledown, location.abovebar, '
             'color.new(color.red, 0), size=size.small)',
             "if entryLong and pos <= 0",
             f'    alert("{name}: LONG setup on " + syminfo.ticker, alert.freq_once_per_bar_close)',
             "if entryShort and pos >= 0",
             f'    alert("{name}: SHORT setup on " + syminfo.ticker, alert.freq_once_per_bar_close)']
    body += _zone_plots(spec)
    return "\n".join(head + [""] + body) + "\n"


def _zone_plots(spec: dict) -> list[str]:
    """Draw the ICT zones the strategy uses, so the setups can be read on the chart."""
    out = []
    colors = {"bull_top": "color.teal", "bull_bot": "color.teal", "bear_top": "color.red",
              "bear_bot": "color.red", "high": "color.gray", "low": "color.gray"}
    for ind in spec.get("indicators") or []:
        if ind["type"] in ("fvg", "order_block", "prev_day", "swings"):
            for out_name, col in colors.items():
                if out_name in ict.CATALOG[ind["type"]][1]:
                    out.append(f'plot({_var(ind["id"], out_name)}, "{ind["id"]} {out_name}", '
                               f"color.new({col}, 60), style=plot.style_linebr)")
    return out
