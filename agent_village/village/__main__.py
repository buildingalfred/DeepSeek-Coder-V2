"""Command line: python -m village <command> --help"""

import argparse
import json
import sys
from pathlib import Path

from . import backtest, data, llm
from .agents import leaderboard_text
from .board import Board
from .village import Village


def load(path: str | None):
    if not path or path == "sample":
        return data.sample(), "sample"
    return data.load_csv(path), Path(path).name


def cmd_run(a):
    df, name = load(a.data)
    try:
        brain = llm.make(a.llm)
    except ImportError:
        sys.exit("The claude backend needs the anthropic package: pip install anthropic")
    if brain is None and a.llm == "auto":
        print("No LLM found (Ollama not running, no ANTHROPIC_API_KEY). Running heuristic villagers.")
    board = Board(a.db)
    try:
        village = Village(df, name, board, brain, quants=a.quants.split(","), fee_bps=a.fee_bps,
                          train_frac=a.train, seed=a.seed)
        village.run(a.rounds, a.papers, a.reports)
    finally:
        board.close()


def cmd_backtest(a):
    df, _ = load(a.data)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    res = backtest.run(df, spec, a.fee_bps, a.train)
    print(json.dumps({"train": res.train, "test": res.test, "full": res.full}, indent=2))


def cmd_board(a):
    board = Board(a.db)
    try:
        _, name = load(a.data)
        print(leaderboard_text(board.leaderboard(name, a.top)))
        for note in board.notes("critique", 1):
            print(f"\nLatest critique (round {note['round']}):\n{note['content']}")
    finally:
        board.close()


def cmd_sample(a):
    df = data.sample(a.bars, a.seed or 7)
    df.index.name = "date"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out)
    print(f"Wrote {len(df)} synthetic bars to {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="village", description="A village of agents that research "
                                "trading strategies together.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", default="sample",
                        help="OHLCV CSV file, or 'sample' for synthetic data (default)")
        sp.add_argument("--fee-bps", type=float, default=5.0, help="fee per side in basis points")
        sp.add_argument("--train", type=float, default=0.7,
                        help="fraction of bars the agents may see (rest is the vault)")

    r = sub.add_parser("run", help="run the village for some rounds")
    common(r)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--papers", default="papers", help="folder of PDFs / .txt / .md to read")
    r.add_argument("--llm", default="auto", help="auto | none | ollama[:model] | claude[:model]")
    r.add_argument("--quants", default="trend,reversion,breakout",
                   help="comma-separated quant styles: trend, reversion, breakout")
    r.add_argument("--db", default="village.db", help="the village's memory file")
    r.add_argument("--reports", default="reports")
    r.add_argument("--seed", type=int, default=None)
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("backtest", help="backtest one strategy JSON file")
    common(b)
    b.add_argument("spec")
    b.set_defaults(fn=cmd_backtest)

    lb = sub.add_parser("board", help="show the leaderboard and latest critique")
    lb.add_argument("--data", default="sample")
    lb.add_argument("--db", default="village.db")
    lb.add_argument("--top", type=int, default=10)
    lb.set_defaults(fn=cmd_board)

    s = sub.add_parser("sample", help="write synthetic price data to a CSV")
    s.add_argument("--out", default="data/sample.csv")
    s.add_argument("--bars", type=int, default=3000)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(fn=cmd_sample)

    a = p.parse_args(argv)
    for name in ("quants",):
        if hasattr(a, name):
            bad = [q for q in getattr(a, name).split(",") if q not in ("trend", "reversion", "breakout")]
            if bad:
                p.error(f"unknown quant style(s): {bad}")
    a.fn(a)


if __name__ == "__main__":
    main()
