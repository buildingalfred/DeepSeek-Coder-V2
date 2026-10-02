"""Strategy specs: plain JSON that agents write and the backtester runs.

Agents never write Python. They describe a strategy as data, which keeps the village safe to run
and makes every idea easy to store, compare and mutate.

Example spec:
{
  "name": "EMA trend with RSI filter",
  "idea": "Ride trends, avoid buying when overbought",
  "indicators": [
    {"id": "fast", "type": "ema", "period": 12},
    {"id": "slow", "type": "ema", "period": 48},
    {"id": "rsi", "type": "rsi", "period": 14}
  ],
  "entry_long":  [{"left": "fast", "op": "crosses_above", "right": "slow"},
                  {"left": "rsi", "op": "<", "right": 70}],
  "exit_long":   [{"left": "fast", "op": "crosses_below", "right": "slow"}],
  "entry_short": [],
  "exit_short":  [],
  "stop_loss_pct": 5,
  "take_profit_pct": null
}

Conditions inside one list are combined with AND. Add "within": N to a condition to make it true
when it held on any of the last N bars, which chains steps into a sequence (sweep, then shift,
then retrace). "left"/"right" may be a price column
(open, high, low, close, volume), an indicator id, "<id>.<output>" for multi-output indicators
(e.g. "bb.upper", "m.hist"), or (right only) a number.
"""

import json
import re

import numpy as np
import pandas as pd

from . import features, indicators

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
OPS = [">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"]
MAX_WITHIN = 500
RULE_KEYS = ["entry_long", "exit_long", "entry_short", "exit_short"]
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SpecError(ValueError):
    """A strategy spec is malformed. The message is written to be shown back to an agent."""


def build_columns(df: pd.DataFrame, spec: dict, cache=None) -> pd.DataFrame:
    """Validate the spec and return a frame with every column the rules can reference."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    cols = df[PRICE_COLUMNS].copy()
    for ind in spec.get("indicators") or []:
        if not isinstance(ind, dict):
            raise SpecError(f"indicator entries must be objects, got {ind!r}")
        ind_id, kind = ind.get("id"), ind.get("type")
        if not ind_id or not _ID_RE.match(str(ind_id)):
            raise SpecError(f"indicator needs an 'id' made of letters, digits or _ (got {ind_id!r})")
        if ind_id in cols.columns:
            raise SpecError(f"duplicate or reserved indicator id '{ind_id}'")
        params = {k: v for k, v in ind.items() if k not in ("id", "type")}
        try:
            out = cache.get(df, kind, params) if cache else indicators.compute(df, kind, params)
        except (ValueError, TypeError, KeyError) as e:
            raise SpecError(f"indicator '{ind_id}': {e}") from e
        for name in out.columns:
            cols[ind_id if name == "value" else f"{ind_id}.{name}"] = out[name]

    for feat in spec.get("features") or []:
        if not isinstance(feat, dict):
            raise SpecError(f"feature entries must be objects, got {feat!r}")
        fid = feat.get("id")
        if not fid or not _ID_RE.match(str(fid)) or fid in cols.columns:
            raise SpecError(f"feature needs a new 'id' made of letters, digits or _ (got {fid!r})")
        try:
            cols[fid] = features.evaluate(feat.get("expr", ""), cols, df.attrs.get("tz", "UTC"))
        except features.FeatureError as e:
            raise SpecError(f"feature '{fid}': {e}") from e

    if not any(spec.get(k) for k in ("entry_long", "entry_short")):
        raise SpecError("spec needs at least one condition in entry_long or entry_short")
    for key in RULE_KEYS:
        rules = spec.get(key) or []
        if not isinstance(rules, list):
            raise SpecError(f"'{key}' must be a list of conditions")
        for cond in rules:
            _check_condition(cond, cols, key)
    for key in ("stop_loss_pct", "take_profit_pct"):
        v = spec.get(key)
        if v is not None and (not isinstance(v, (int, float)) or v <= 0):
            raise SpecError(f"'{key}' must be a positive number of percent or null")
    return cols


def _check_condition(cond, cols: pd.DataFrame, key: str) -> None:
    if not isinstance(cond, dict):
        raise SpecError(f"{key}: each condition must be an object, got {cond!r}")
    left, op, right = cond.get("left"), cond.get("op"), cond.get("right")
    available = ", ".join(cols.columns)
    if left not in cols.columns:
        raise SpecError(f"{key}: unknown left '{left}'. Available: {available}")
    if op not in OPS:
        raise SpecError(f"{key}: unknown op '{op}'. Use one of {OPS}")
    if not isinstance(right, (int, float)) and right not in cols.columns:
        raise SpecError(f"{key}: right must be a number or one of: {available} (got {right!r})")
    within = cond.get("within")
    if within is not None and (not isinstance(within, int) or isinstance(within, bool)
                               or not 1 <= within <= MAX_WITHIN):
        raise SpecError(f"{key}: 'within' must be a whole number of bars from 1 to {MAX_WITHIN}")


def _condition_mask(cond: dict, cols: pd.DataFrame) -> np.ndarray:
    left = cols[cond["left"]]
    right = cond["right"]
    right = cols[right] if isinstance(right, str) else pd.Series(float(right), index=cols.index)
    op = cond["op"]
    if op == ">":
        m = left > right
    elif op == "<":
        m = left < right
    elif op == ">=":
        m = left >= right
    elif op == "<=":
        m = left <= right
    elif op == "==":
        m = left == right
    elif op == "crosses_above":
        m = (left > right) & (left.shift() <= right.shift())
    else:
        m = (left < right) & (left.shift() >= right.shift())
    m = m.fillna(False).astype(bool)
    within = cond.get("within")
    if within and within > 1:
        # True if the condition held on any of the last `within` bars (including this one).
        m = m.astype(float).rolling(within, min_periods=1).max().astype(bool)
    return m.to_numpy(dtype=bool)


def signals(df: pd.DataFrame, spec: dict, cache=None) -> dict:
    """Boolean arrays for each rule list (AND of its conditions; empty list -> never true)."""
    cols = build_columns(df, spec, cache)
    out = {}
    for key in RULE_KEYS:
        rules = spec.get(key) or []
        mask = np.ones(len(df), dtype=bool) if rules else np.zeros(len(df), dtype=bool)
        for cond in rules:
            mask &= _condition_mask(cond, cols)
        out[key] = mask
    return out


def parse(text_or_dict) -> dict:
    """Accept a dict or JSON text (optionally wrapped in prose / code fences) and return a dict."""
    if isinstance(text_or_dict, dict):
        return text_or_dict
    obj = extract_json(text_or_dict)
    if not isinstance(obj, dict):
        raise SpecError("expected a JSON object for the strategy")
    return obj


def extract_json(text: str):
    """Pull the first JSON object or array out of an LLM reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise SpecError("could not find valid JSON in the reply")


def describe(spec: dict) -> str:
    """One-line human summary of a spec."""
    def rules(key):
        return " AND ".join(
            f"{c['left']} {c['op']} {c['right']}" + (f" within {c['within']}" if c.get("within") else "")
            for c in spec.get(key) or [])

    parts = [f"{f.get('id')} = {f.get('expr')}" for f in spec.get("features") or []]
    for key in RULE_KEYS:
        r = rules(key)
        if r:
            parts.append(f"{key}: {r}")
    for key in ("stop_loss_pct", "take_profit_pct"):
        if spec.get(key):
            parts.append(f"{key}={spec[key]}")
    return "; ".join(parts)
