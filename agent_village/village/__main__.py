"""Command line: python -m village <command> --help"""

import argparse
import json
import sys
from pathlib import Path

from . import backtest, data, llm
from .agents import ALGO_MISSION, ICT_MISSION, STYLES, TEAMS, leaderboard_text
from .board import Board
from .village import Village


def data_files(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        files += (sorted(f for f in p.iterdir() if f.suffix.lower() in data.DATA_SUFFIXES)
                  if p.is_dir() else [p])
    if not files:
        sys.exit(f"No .csv or .txt files found in {paths}")
    return files


def dataset_name(paths: list[str], timeframe: str | None = None) -> str:
    """The village's memory is kept per dataset: these files at this timeframe."""
    if not paths or paths == ["sample"]:
        name = "sample"
    else:
        name = "+".join(sorted(f.name for f in data_files(paths)))
    return f"{name} @{timeframe}" if timeframe else name


def load_markets(paths: list[str], tz: str = "UTC", timeframe: str | None = None):
    """Turn --data arguments (files, folders, or 'sample') into ({name: frame}, dataset name)."""
    if not paths or paths == ["sample"]:
        df = data.sample()
        return {"sample": data.resample(df, timeframe) if timeframe else df}, \
            dataset_name(paths, timeframe)
    files = data_files(paths)
    markets = {}
    from_folder = any(Path(p).is_dir() for p in paths)
    for f in files:
        try:
            print(f"Loading {f.name} ...", flush=True)
            markets[f.name] = data.load_csv(f, tz, timeframe)
        except (OSError, ValueError, KeyError, IndexError) as e:
            if not from_folder:
                sys.exit(f"Could not load {f}: {e}")
            print(f"  skipping {f.name}: not price data ({str(e)[:120]})")
    if not markets:
        sys.exit(f"No price data could be loaded from {paths}")
    name = "+".join(sorted(markets))
    return markets, f"{name} @{timeframe}" if timeframe else name


def load(path):
    markets, name = load_markets(path if isinstance(path, list) else [path])
    if len(markets) > 1:
        sys.exit("this command takes a single data file")
    return next(iter(markets.values())), name


def cmd_run(a):
    url = None
    if a.watch:
        from .live import serve
        Board(a.db).close()  # make sure the database exists before the page asks for it
        url = serve(a.db, a.port, open_browser=True, background=True)
        print(f"Live village at {url}")
    markets, name = load_markets(a.data, a.tz, a.timeframe)
    try:
        brain = llm.make(a.llm)
    except ImportError:
        sys.exit("The claude backend needs the anthropic package: pip install anthropic")
    if brain is None and a.llm == "auto":
        print("No LLM found (Ollama not running, no ANTHROPIC_API_KEY). Running heuristic villagers.")
    board = Board(a.db)
    try:
        quants = a.quants.split(",") if a.quants else TEAMS[a.team]
        default_mission = {"ict": ICT_MISSION + " " + ALGO_MISSION, "mixed": ALGO_MISSION}
        mission = a.mission if a.mission is not None else default_mission.get(a.team, "")
        kw = dict(llm=brain, quants=quants, fee_bps=a.fee_bps, train_frac=a.train, seed=a.seed,
                  tuner=not a.no_tuner, mission=mission)
        survivors, table = [], []
        if a.scout:
            from .stages import ladder
            if a.papers:  # read the library once, before scouting
                Village(markets, name, board, **kw).librarian.study(a.papers, board, board.last_round())
            survivors, table = ladder(markets, name, board, a.scout,
                                      [x for x in a.expand.split(",") if x], a.scout_rounds,
                                      a.stage_rounds, report_dir=a.reports, **kw)
            print("\nThe ladder (robust score / unseen-part sharpe at each window):")
            print("\n".join(table))
        village = Village(markets, name, board, **kw)
        if survivors:
            village.seed(survivors, "ladder survivor")
            board.post(board.last_round(), "ladder", "ladder", "\n".join(table), name)
        village.run(a.rounds, a.papers, a.reports)
    finally:
        board.close()
    if url:
        print(f"\nDone. The live village stays open at {url} - press Ctrl+C to quit.")
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def cmd_watch(a):
    from .live import serve
    serve(a.db, a.port, open_browser=not a.no_browser)


def cmd_backtest(a):
    markets, _ = load_markets(a.data, a.tz, a.timeframe)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    print(json.dumps(backtest.evaluate(markets, spec, a.fee_bps, a.train), indent=2))


def cmd_board(a):
    board = Board(a.db)
    try:
        name = dataset_name(a.data, a.timeframe)
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


def cmd_pine(a):
    from . import pine, strategy
    if a.id is not None:
        board = Board(a.db)
        try:
            row = board.db.execute("SELECT * FROM strategies WHERE id = ?", (a.id,)).fetchone()
        finally:
            board.close()
        if row is None:
            sys.exit(f"No strategy #{a.id} in {a.db}")
        spec, stats = json.loads(row["spec"]), {"train": json.loads(row["train"] or "null"),
                                                "vault": json.loads(row["test"] or "null")}
    else:
        spec, stats = json.loads(Path(a.spec).read_text(encoding="utf-8")), None
    # Validate against sample data first, so a broken spec fails here and not in TradingView.
    try:
        strategy.signals(data.sample(400, freq="15min"), spec)
    except strategy.SpecError as e:
        sys.exit(f"Invalid strategy: {e}")
    out = Path(a.out or (Path(a.spec).with_suffix(".pine") if a.spec else f"strategy_{a.id}.pine"))
    out.write_text(pine.to_pine(spec, a.fee_bps, stats), encoding="utf-8")
    print(f"Wrote {out}. In TradingView: Pine Editor -> paste -> Add to chart -> Strategy Tester.")


def cmd_transcribe(a):
    from .pdfs import transcribe
    try:
        n = transcribe(a.folder, a.out, a.model)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"Done: {n} new transcript(s) in {a.out}. The Librarian reads them on the next run.")


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
        sp.add_argument("--timeframe", default=None,
                        help="resample after loading, e.g. 5min, 15min, 1h, 4h, 1D "
                             "(recommended for years of 1-minute data)")
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
    r.add_argument("--watch", action="store_true",
                   help="open the live village in your browser while it works")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--scout", default=None,
                   help="start small: hunt on this much recent data first, e.g. 3M or 90D")
    r.add_argument("--expand", default="6M,1Y,3Y,all",
                   help="then test survivors on longer windows (default 6M,1Y,3Y,all)")
    r.add_argument("--scout-rounds", type=int, default=8)
    r.add_argument("--stage-rounds", type=int, default=3,
                   help="rounds the team gets to adapt setups at each expansion step")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("backtest", help="backtest one strategy JSON file")
    common(b)
    b.add_argument("spec")
    b.set_defaults(fn=cmd_backtest)

    lb = sub.add_parser("board", help="show the leaderboard and latest critique")
    lb.add_argument("--data", nargs="+", default=["sample"])
    lb.add_argument("--timeframe", default=None)
    lb.add_argument("--db", default="village.db")
    lb.add_argument("--top", type=int, default=10)
    lb.set_defaults(fn=cmd_board)

    s = sub.add_parser("sample", help="write synthetic price data to a CSV")
    s.add_argument("--out", default="data/sample.csv")
    s.add_argument("--bars", type=int, default=3000)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--freq", default="B", help="bar size: B (business days, default), 1h, 15min ...")
    s.set_defaults(fn=cmd_sample)

    w = sub.add_parser("watch", help="open the live village (characters, whiteboard, leaderboard)")
    w.add_argument("--db", default="village.db")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true")
    w.set_defaults(fn=cmd_watch)

    pn = sub.add_parser("pine", help="turn a strategy into a TradingView Pine Script")
    pn.add_argument("spec", nargs="?", default="reports/best_strategy.json",
                    help="strategy JSON file (default reports/best_strategy.json)")
    pn.add_argument("--id", type=int, default=None, help="or a strategy number from the board")
    pn.add_argument("--db", default="village.db")
    pn.add_argument("--fee-bps", type=float, default=5.0)
    pn.add_argument("--out", default=None)
    pn.set_defaults(fn=cmd_pine)

    tr = sub.add_parser("transcribe", help="turn videos into text for the Librarian (offline)")
    tr.add_argument("folder", nargs="?", default="videos")
    tr.add_argument("--out", default="papers/transcripts")
    tr.add_argument("--model", default="small",
                    help="Whisper size: tiny, base, small (default), medium, large-v3")
    tr.set_defaults(fn=cmd_transcribe)

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
