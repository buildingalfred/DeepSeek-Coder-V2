"""Command line: python -m village <command> --help"""

import argparse
import json
import sys
from pathlib import Path

from . import backtest, data, llm
from .agents import ICT_MISSION, STYLES, TEAMS, leaderboard_text
from .board import Board
from .village import Village


def load_markets(paths: list[str], tz: str = "UTC"):
    """Turn --data arguments (files, folders, or 'sample') into ({name: frame}, dataset name)."""
    if not paths or paths == ["sample"]:
        return {"sample": data.sample()}, "sample"
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"No CSV files found in {paths}")
    markets = {}
    for f in files:
        try:
            markets[f.name] = data.load_csv(f, tz)
        except (OSError, ValueError) as e:
            sys.exit(f"Could not load {f}: {e}")
    return markets, "+".join(sorted(markets))


def load(path):
    markets, name = load_markets(path if isinstance(path, list) else [path])
    if len(markets) > 1:
        sys.exit("this command takes a single data file")
    return next(iter(markets.values())), name


def cmd_run(a):
    markets, name = load_markets(a.data, a.tz)
    try:
        brain = llm.make(a.llm)
    except ImportError:
        sys.exit("The claude backend needs the anthropic package: pip install anthropic")
    if brain is None and a.llm == "auto":
        print("No LLM found (Ollama not running, no ANTHROPIC_API_KEY). Running heuristic villagers.")
    board = Board(a.db)
    try:
        quants = a.quants.split(",") if a.quants else TEAMS[a.team]
        mission = a.mission if a.mission is not None else (ICT_MISSION if "ict" in a.team else "")
        village = Village(markets, name, board, brain, quants=quants, fee_bps=a.fee_bps,
                          train_frac=a.train, seed=a.seed, tuner=not a.no_tuner, mission=mission)
        village.run(a.rounds, a.papers, a.reports)
    finally:
        board.close()


def cmd_backtest(a):
    markets, _ = load_markets(a.data, a.tz)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    print(json.dumps(backtest.evaluate(markets, spec, a.fee_bps, a.train), indent=2))


def cmd_board(a):
    board = Board(a.db)
    try:
        name = load_markets(a.data)[1]
        print(leaderboard_text(board.leaderboard(name, a.top)))
        for note in board.notes("critique", 1, name):
            print(f"\nLatest critique (round {note['round']}):\n{note['content']}")
    finally:
        board.close()


def cmd_sample(a):
    df = data.sample(a.bars, a.seed or 7, a.freq)
    df.index.name = "date"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out)
    print(f"Wrote {len(df)} synthetic bars to {a.out}")


def cmd_fetch(a):
    try:
        df = data.fetch(a.symbol, a.period, a.interval)
    except ImportError:
        sys.exit("fetch needs the yfinance package: pip install yfinance")
    except ValueError as e:
        sys.exit(str(e))
    out = Path(a.out or f"data/{a.symbol.replace('/', '-').replace('^', '')}_{a.interval}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = "date"
    df.to_csv(out)
    print(f"Wrote {len(df)} bars of {a.symbol} to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="village", description="A village of agents that research "
                                "trading strategies together.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", nargs="+", default=["sample"],
                        help="one or more OHLCV CSV files or folders of them, or 'sample' "
                             "for synthetic data (default). Several markets = tougher test.")
        sp.add_argument("--fee-bps", type=float, default=5.0, help="fee per side in basis points")
        sp.add_argument("--tz", default="UTC",
                        help="time zone the CSV times are in, e.g. UTC, America/New_York, "
                             "Europe/London (used by the ICT kill-zone indicators)")
        sp.add_argument("--train", type=float, default=0.7,
                        help="fraction of bars the agents may see (rest is the vault)")

    r = sub.add_parser("run", help="run the village for some rounds")
    common(r)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--papers", default="papers", help="folder of PDFs / .txt / .md to read")
    r.add_argument("--llm", default="auto", help="auto | none | ollama[:model] | claude[:model]")
    r.add_argument("--team", default="default", choices=sorted(TEAMS),
                   help="default (trend, reversion, breakout), ict (three ICT hunters) or mixed")
    r.add_argument("--quants", default=None,
                   help=f"comma-separated quant styles instead of a team: {', '.join(STYLES)}")
    r.add_argument("--mission", default=None,
                   help="what the whole team should work on (the ict team has one built in)")
    r.add_argument("--db", default="village.db", help="the village's memory file")
    r.add_argument("--reports", default="reports")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--no-tuner", action="store_true", help="leave out the tuner villager")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("backtest", help="backtest one strategy JSON file")
    common(b)
    b.add_argument("spec")
    b.set_defaults(fn=cmd_backtest)

    lb = sub.add_parser("board", help="show the leaderboard and latest critique")
    lb.add_argument("--data", nargs="+", default=["sample"])
    lb.add_argument("--db", default="village.db")
    lb.add_argument("--top", type=int, default=10)
    lb.set_defaults(fn=cmd_board)

    s = sub.add_parser("sample", help="write synthetic price data to a CSV")
    s.add_argument("--out", default="data/sample.csv")
    s.add_argument("--bars", type=int, default=3000)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--freq", default="B", help="bar size: B (business days, default), 1h, 15min ...")
    s.set_defaults(fn=cmd_sample)

    f = sub.add_parser("fetch", help="download prices from Yahoo Finance (needs yfinance)")
    f.add_argument("symbol", help="e.g. SPY, AAPL, BTC-USD, EURUSD=X, ^GSPC")
    f.add_argument("--period", default="max", help="e.g. 5y, 10y, max (default)")
    f.add_argument("--interval", default="1d", help="1d (default), 1h, 1wk ...")
    f.add_argument("--out", default=None, help="CSV path (default data/<symbol>_<interval>.csv)")
    f.set_defaults(fn=cmd_fetch)

    a = p.parse_args(argv)
    if getattr(a, "quants", None):
        bad = [q for q in a.quants.split(",") if q not in STYLES]
        if bad:
            p.error(f"unknown quant style(s): {bad}. Choose from {', '.join(STYLES)}")
    a.fn(a)


if __name__ == "__main__":
    main()
