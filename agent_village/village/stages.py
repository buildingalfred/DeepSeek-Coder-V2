"""Scout, then expand: find setups on a few months of data, then see which survive more history.

  1. Scout: the village hunts on a short recent window (e.g. the last 3 months before the vault).
  2. Expand: survivors move up a ladder of longer windows (6 months, 1 year, 3 years, all). At
     every step the village gets a few rounds to adapt them, and only setups that still work
     (positive robust score and positive on the step's own unseen part) climb on.
  3. Final: the last survivors seed the full run, which ends with the real vault.

Every window lies inside the training part of the data, so the final vault stays untouched.
"""

import pandas as pd

from . import backtest
from .board import Board
from .village import Village, fingerprint, short_id, verdict


def parse_span(text: str):
    """'3M' -> 3 months, '1Y', '6W', '90D', 'all' -> None (whole training part)."""
    text = text.strip().upper()
    if text in ("ALL", "MAX"):
        return None
    num, unit = text[:-1], text[-1]
    try:
        n = int(num)
    except ValueError:
        raise ValueError(f"bad window '{text}', use e.g. 3M, 6M, 1Y, 90D, 6W or all") from None
    units = {"D": "days", "W": "weeks", "M": "months", "Y": "years"}
    if unit not in units or n <= 0:
        raise ValueError(f"bad window '{text}', use e.g. 3M, 6M, 1Y, 90D, 6W or all")
    return pd.DateOffset(**{units[unit]: n})


def window(df: pd.DataFrame, span, train_frac: float) -> pd.DataFrame:
    """The last `span` of the training part of df (or all of it)."""
    train = df.iloc[: int(len(df) * train_frac)]
    if span is None:
        return train
    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("scouting by months needs dates in the data")
    out = train[train.index > train.index[-1] - span]
    out.attrs.update(df.attrs)
    return out


def ladder(markets: dict, dataset: str, board: Board, scout: str, expand: list[str],
           scout_rounds: int = 8, stage_rounds: int = 3, keep: int = 10, log=print,
           report_dir: str = "reports", **village_kw) -> tuple[list[dict], list[str]]:
    """Run scout + expansion. Returns (survivor specs, ladder table rows as text)."""
    train_frac = village_kw.get("train_frac", 0.7)
    spans = [scout] + list(expand)
    history: dict[str, dict] = {}   # fingerprint -> {"name":..., stage: "robust / vault sharpe"}
    survivors: list[dict] = []
    for step, span_text in enumerate(spans):
        span = parse_span(span_text)
        part = {n: window(df, span, train_frac) for n, df in markets.items()}
        short = [n for n, df in part.items() if len(df) < 300]
        if short:
            log(f"\n[ladder] window {span_text} has too few bars for {short}; skipping this step")
            continue
        label = f"{dataset} | {'scout' if step == 0 else 'expand'} {span_text}"
        bars = sum(len(df) for df in part.values())
        log(f"\n######## {label}: {bars:,} bars ########")
        v = Village(part, label, board, log=log, **village_kw)
        if survivors:
            v.seed(survivors, f"from {spans[step - 1]}")
        v.run(scout_rounds if step == 0 else stage_rounds, None, f"{report_dir}/ladder")
        # Survivors: positive robust score AND positive on this window's own unseen part.
        alive = []
        for row in board.leaderboard(label, 50):
            if row["score"] <= 0 or row["test"]["sharpe"] <= 0:
                continue
            fp = fingerprint(row["spec"])
            entry = history.setdefault(fp, {"name": f"[{short_id(row['spec'])}] {row['name']}"})
            entry[span_text] = (f"{row['train']['robust_sharpe']} / {row['test']['sharpe']} "
                                f"({verdict(row['train'], row['test'])})")
            alive.append(row["spec"])
            if len(alive) >= keep:
                break
        log(f"[ladder] {len(alive)} setups survive the {span_text} window")
        if not alive:
            break
        survivors = alive
    header = "| Setup | " + " | ".join(spans) + " |"
    rows = [header, "|" + "---|" * (len(spans) + 1)]
    for entry in sorted(history.values(), key=lambda e: -sum(s in e for s in spans)):
        rows.append(f"| {entry['name'][:60]} | " + " | ".join(entry.get(s, "-") for s in spans)
                    + " |")
    return survivors, rows
