"""Invented features: the villagers' imagination, written as small safe formulas.

A strategy can define its own measurements on top of prices and indicators, e.g.

  {"id": "dr", "expr": "digital_root(count(st.bos_up, 50))"}
  {"id": "wait", "expr": "minutes_since(sw.bull)"}
  {"id": "late", "expr": "ny_minute() > 600 and day_count(fvg.bull) >= 2"}

Formulas are parsed with a strict whitelist (numbers, names, arithmetic, comparisons, and the
functions below). Nothing else can run, and every function only looks backwards in time.
Conditions produce 1 (true) or 0 (false). Missing values stay missing, and a missing value in a
comparison is false.

Each formula is also translated to Pine Script (see to_pine) so exported strategies keep it.
"""

import ast

import numpy as np
import pandas as pd

NY = "America/New_York"
MAX_LEN = 400
MAX_WINDOW = 5000

FUNCS = {
    # name: (min args, max args, doc)
    "bars_since": (1, 1, "bars since cond was last true (0 = now, missing if never)"),
    "minutes_since": (1, 1, "minutes since cond was last true (intraday data)"),
    "count": (2, 2, "count(cond, n): bars in the last n where cond was true"),
    "day_count": (1, 1, "times cond was true so far today (New York day), this bar included"),
    "sum": (2, 2, "sum(x, n) over the last n bars"),
    "mean": (2, 2, "mean(x, n) over the last n bars"),
    "highest": (2, 2, "highest(x, n) over the last n bars"),
    "lowest": (2, 2, "lowest(x, n) over the last n bars"),
    "std": (2, 2, "std(x, n) over the last n bars"),
    "prev": (1, 2, "prev(x, n=1): value n bars ago"),
    "change": (1, 2, "change(x, n=1): x minus its value n bars ago"),
    "digital_root": (1, 1, "digital root of the rounded absolute value (1-9, 0 for 0)"),
    "abs": (1, 1, "absolute value"),
    "round": (1, 1, "round to the nearest whole number"),
    "floor": (1, 1, "round down"),
    "min": (2, 2, "smaller of two values"),
    "max": (2, 2, "larger of two values"),
    "where": (3, 3, "where(cond, a, b): a when cond is true, else b"),
    "ny_minute": (0, 0, "minutes since midnight New York time (570 = 09:30)"),
    "day_of_week": (0, 0, "0 = Monday ... 6 = Sunday (New York time)"),
}
WINDOW_FUNCS = {"count", "sum", "mean", "highest", "lowest", "std"}
INT_ARG = {"count": 1, "sum": 1, "mean": 1, "highest": 1, "lowest": 1, "std": 1, "prev": 1,
           "change": 1}

GUIDE = "Invented features: add \"features\": [{\"id\": \"name\", \"expr\": \"formula\"}] to a " \
        "strategy, then use the id in conditions like any indicator. Formulas may use prices, " \
        "indicator outputs (e.g. sw.bull), earlier features, numbers, + - * / %, comparisons, " \
        "and/or/not, and these functions:\n" + "\n".join(
            f"- {n}: {d}" for n, (_, _, d) in FUNCS.items())


class FeatureError(ValueError):
    pass


_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name,
            ast.Attribute, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
            ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Gt, ast.Lt, ast.GtE, ast.LtE,
            ast.Eq, ast.NotEq)


def parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise FeatureError("a feature needs a non-empty 'expr' formula")
    if len(expr) > MAX_LEN:
        raise FeatureError(f"formula longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FeatureError(f"formula is not valid: {e.msg}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise FeatureError(f"'{type(node).__name__}' is not allowed in formulas")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool)
                                               or not isinstance(node.value, (int, float))):
            raise FeatureError("only numbers are allowed as constants")
        if isinstance(node, ast.Attribute) and not isinstance(node.value, ast.Name):
            raise FeatureError("use names like 'sw.bull' (one dot only)")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                raise FeatureError(f"unknown function. Available: {', '.join(FUNCS)}")
            lo, hi, _ = FUNCS[node.func.id]
            if not lo <= len(node.args) <= hi or node.keywords:
                raise FeatureError(f"{node.func.id} takes {lo}-{hi} arguments")
            if node.func.id in INT_ARG and len(node.args) > INT_ARG[node.func.id]:
                arg = node.args[INT_ARG[node.func.id]]
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, int)
                        and 1 <= arg.value <= MAX_WINDOW) and not (
                        node.func.id in ("prev", "change") and isinstance(arg, ast.Constant)
                        and arg.value == 0):
                    raise FeatureError(f"{node.func.id}: the bar count must be a whole number "
                                       f"from 1 to {MAX_WINDOW}")
    return tree


def _name(node) -> str:
    return node.id if isinstance(node, ast.Name) else f"{node.value.id}.{node.attr}"


def _ny(index):
    if not isinstance(index, pd.DatetimeIndex):
        raise FeatureError("time functions need a date/time column in the data")
    return index


def _bool(x: pd.Series) -> pd.Series:
    return (x > 0.5).fillna(False)


def _f(x: pd.Series) -> pd.Series:
    return x.astype(float)


def evaluate(expr: str, cols: pd.DataFrame, tz: str = "UTC") -> pd.Series:
    """Evaluate a formula against the strategy's columns. Returns a float series."""
    tree = parse(expr)
    idx = cols.index

    def ny_index():
        i = _ny(idx)
        loc = i if i.tz is not None else i.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        return loc.tz_convert(NY)

    def ev(node) -> pd.Series:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return pd.Series(float(node.value), index=idx)
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = _name(node)
            if name not in cols.columns:
                raise FeatureError(f"unknown name '{name}'. Available: {', '.join(cols.columns)}")
            return _f(cols[name])
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return _f(~_bool(v))
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return (a / b).replace([np.inf, -np.inf], np.nan)
            return (a - b * np.floor(a / b)).replace([np.inf, -np.inf], np.nan)  # floor modulo
        if isinstance(node, ast.BoolOp):
            vals = [_bool(ev(v)) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return _f(out)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise FeatureError("write one comparison at a time, e.g. (a < b) and (b < c)")
            a, b, op = ev(node.left), ev(node.comparators[0]), node.ops[0]
            res = {ast.Gt: a > b, ast.Lt: a < b, ast.GtE: a >= b, ast.LtE: a <= b,
                   ast.Eq: a == b, ast.NotEq: (a != b) & a.notna() & b.notna()}[type(op)]
            return _f(res.fillna(False))
        return call(node)

    def int_arg(node, i, default):
        return node.args[i].value if len(node.args) > i else default

    def call(node) -> pd.Series:
        fn = node.func.id
        if fn == "ny_minute":
            ny = ny_index()
            return pd.Series(ny.hour * 60 + ny.minute, index=idx).astype(float)
        if fn == "day_of_week":
            return pd.Series(ny_index().weekday, index=idx).astype(float)
        x = ev(node.args[0])
        if fn in ("bars_since", "minutes_since"):
            hit = _bool(x).to_numpy()
            pos = np.where(hit, np.arange(len(hit)), -1)
            last = np.maximum.accumulate(pos) if len(pos) else pos
            if fn == "bars_since":
                out = np.where(last >= 0, np.arange(len(hit)) - last, np.nan)
                return pd.Series(out, index=idx, dtype=float)
            base = pd.Timestamp("1970-01-01", tz=_ny(idx).tz)
            t = ((idx - base) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)
            safe = np.where(last >= 0, last, 0)
            return pd.Series(np.where(last >= 0, t - t[safe], np.nan), index=idx, dtype=float)
        if fn == "day_count":
            day = pd.Series(ny_index().normalize(), index=idx)
            return _f(_bool(x)).groupby(day.values).cumsum()
        if fn in WINDOW_FUNCS:
            n = int_arg(node, 1, 1)
            src = _f(_bool(x)) if fn == "count" else x
            roll = src.rolling(n, min_periods=n)
            return {"count": roll.sum, "sum": roll.sum, "mean": roll.mean, "highest": roll.max,
                    "lowest": roll.min, "std": roll.std}[fn]()
        if fn in ("prev", "change"):
            n = int_arg(node, 1, 1)
            return x.shift(n) if fn == "prev" else x - x.shift(n)
        if fn == "digital_root":
            n = np.floor(np.abs(x) + 0.5)
            return pd.Series(np.where(n == 0, 0.0, 1 + (n - 1) % 9), index=idx).where(x.notna())
        if fn == "abs":
            return x.abs()
        if fn == "round":
            return np.sign(x) * np.floor(np.abs(x) + 0.5)  # half away from zero, like Pine
        if fn == "floor":
            return np.floor(x)
        if fn in ("min", "max"):
            y = ev(node.args[1])
            return pd.Series(np.minimum(x, y) if fn == "min" else np.maximum(x, y), index=idx)
        if fn == "where":
            a, b = ev(node.args[1]), ev(node.args[2])
            return pd.Series(np.where(_bool(x), a, b), index=idx, dtype=float)
        raise FeatureError(f"unknown function {fn}")

    return ev(tree)


# ---- Pine Script translation ---------------------------------------------------------------

class PineWriter:
    """Turns a formula into Pine lines. Every sub-expression gets its own variable so history
    references like x[3] always work."""

    def __init__(self, operand, prefix: str):
        self.operand = operand  # maps a column name to its Pine variable
        self.prefix = prefix
        self.lines: list[str] = []
        self.n = 0

    def tmp(self, code: str) -> str:
        self.n += 1
        name = f"{self.prefix}_{self.n}"
        self.lines.append(f"{name} = {code}")
        return name

    def b(self, v: str) -> str:
        return f"f_gt({v}, 0.5)"

    def write(self, expr: str) -> str:
        return self.ev(parse(expr).body)

    def ev(self, node) -> str:
        if isinstance(node, ast.Constant):
            return repr(float(node.value))
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self.operand(_name(node))
        if isinstance(node, ast.UnaryOp):
            v = self.ev(node.operand)
            if isinstance(node.op, ast.Not):
                return self.tmp(f"{self.b(v)} ? 0.0 : 1.0")
            return self.tmp(f"-{v}") if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = self.ev(node.left), self.ev(node.right)
            op = type(node.op)
            if op is ast.Div:
                return self.tmp(f"na({b}) or {b} == 0 ? na : {a} / {b}")
            if op is ast.Mod:
                return self.tmp(f"na({b}) or {b} == 0 ? na : {a} - {b} * math.floor({a} / {b})")
            sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[op]
            return self.tmp(f"{a} {sym} {b}")
        if isinstance(node, ast.BoolOp):
            join = " and " if isinstance(node.op, ast.And) else " or "
            return self.tmp("(" + join.join(self.b(self.ev(v)) for v in node.values) + ") ? 1.0 : 0.0")
        if isinstance(node, ast.Compare):
            a, b = self.ev(node.left), self.ev(node.comparators[0])
            fn = {ast.Gt: "f_gt", ast.Lt: "f_lt", ast.GtE: "f_ge", ast.LtE: "f_le",
                  ast.Eq: "f_eq"}.get(type(node.ops[0]))
            if fn:
                return self.tmp(f"{fn}({a}, {b}) ? 1.0 : 0.0")
            return self.tmp(f"not na({a}) and not na({b}) and {a} != {b} ? 1.0 : 0.0")
        return self.call(node)

    def call(self, node) -> str:
        fn = node.func.id
        if fn == "ny_minute":
            return self.tmp(f'hour(time, "{NY}") * 60 + minute(time, "{NY}")')
        if fn == "day_of_week":
            return self.tmp(f'(dayofweek(time, "{NY}") + 5) % 7')
        x = self.ev(node.args[0])
        n = node.args[1].value if len(node.args) > 1 and fn in INT_ARG else 1
        if fn == "bars_since":
            return self.tmp(f"ta.barssince({self.b(x)})")
        if fn == "minutes_since":
            last = self.tmp(f"ta.valuewhen({self.b(x)}, time, 0)")
            return self.tmp(f"na({last}) ? na : (time - {last}) / 60000.0")
        if fn == "day_count":
            day = self.tmp(f'dayofmonth(time, "{NY}")')
            c = f"{self.prefix}_c{self.n}"
            self.lines += [f"var float {c} = 0.0",
                           f"{c} := (na({day}[1]) or {day} != {day}[1] ? 0.0 : {c}) + "
                           f"({self.b(x)} ? 1.0 : 0.0)"]
            return c
        if fn in WINDOW_FUNCS:
            src = self.tmp(f"{self.b(x)} ? 1.0 : 0.0") if fn == "count" else x
            body = {"count": f"math.sum({src}, {n})", "sum": f"math.sum({src}, {n})",
                    "mean": f"ta.sma({src}, {n})", "highest": f"ta.highest({src}, {n})",
                    "lowest": f"ta.lowest({src}, {n})", "std": f"ta.stdev({src}, {n}, false)"}[fn]
            return self.tmp(f"bar_index < {n - 1} ? na : {body}")
        if fn == "prev":
            return self.tmp(f"{x}[{n}]")
        if fn == "change":
            return self.tmp(f"{x} - {x}[{n}]")
        if fn == "digital_root":
            r = self.tmp(f"na({x}) ? na : math.floor(math.abs({x}) + 0.5)")
            return self.tmp(f"na({r}) ? na : {r} == 0 ? 0.0 : 1 + ({r} - 1) % 9")
        if fn == "abs":
            return self.tmp(f"math.abs({x})")
        if fn == "round":
            return self.tmp(f"na({x}) ? na : math.sign({x}) * math.floor(math.abs({x}) + 0.5)")
        if fn == "floor":
            return self.tmp(f"math.floor({x})")
        if fn in ("min", "max"):
            y = self.ev(node.args[1])
            return self.tmp(f"na({x}) or na({y}) ? na : math.{fn}({x}, {y})")
        a, b = self.ev(node.args[1]), self.ev(node.args[2])
        return self.tmp(f"{self.b(x)} ? {a} : {b}")
