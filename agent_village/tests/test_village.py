import json
import random

import numpy as np
import pandas as pd
import pytest

from village import agents, backtest, data, indicators
from village import strategy as strat
from village.board import Board
from village.village import Village


@pytest.fixture
def df():
    return data.sample(800, seed=3)


def always_long():
    return {"name": "hold", "indicators": [],
            "entry_long": [{"left": "close", "op": ">", "right": 0}], "exit_long": []}


def test_rsi_bounds(df):
    r = indicators.rsi(df["close"], 14).dropna()
    assert r.between(0, 100).all()


def test_always_long_matches_buy_and_hold_minus_fees(df):
    res = backtest.run(df, always_long(), fee_bps=0, train_frac=1.0)
    # Enters at the open of bar 1 and holds to the end.
    expected = df["close"].iloc[-1] / df["open"].iloc[1] - 1
    assert res.full["total_return_pct"] == pytest.approx(expected * 100, abs=0.01)
    assert res.full["trades"] == 1


def test_fees_reduce_returns(df):
    spec = {"name": "x", "indicators": [{"id": "r", "type": "rsi", "period": 2}],
            "entry_long": [{"left": "r", "op": "<", "right": 30}],
            "exit_long": [{"left": "r", "op": ">", "right": 60}]}
    cheap = backtest.run(df, spec, fee_bps=0).full["total_return_pct"]
    pricey = backtest.run(df, spec, fee_bps=50).full["total_return_pct"]
    assert pricey < cheap


def test_no_lookahead(df):
    """Changing future prices must not change past strategy returns."""
    spec = agents.random_spec("trend", random.Random(0))
    base = backtest.run(df, spec, train_frac=1.0).equity
    future = df.copy()
    future.iloc[500:, :4] *= 1.5
    changed = backtest.run(future, spec, train_frac=1.0).equity
    # Bar 499's return can differ only through the next bar's fill; compare strictly earlier bars.
    assert np.allclose(base.iloc[:499], changed.iloc[:499])


def test_spec_errors_are_readable(df):
    bad = {"indicators": [{"id": "a", "type": "nope"}], "entry_long": []}
    with pytest.raises(strat.SpecError, match="unknown indicator"):
        backtest.run(df, bad)
    bad = {"indicators": [], "entry_long": [{"left": "foo", "op": ">", "right": 1}]}
    with pytest.raises(strat.SpecError, match="unknown left 'foo'"):
        backtest.run(df, bad)


def test_multi_output_indicator_columns(df):
    spec = {"indicators": [{"id": "bb", "type": "bbands", "period": 20}],
            "entry_long": [{"left": "close", "op": "crosses_above", "right": "bb.upper"}],
            "exit_long": [{"left": "close", "op": "crosses_below", "right": "bb.mid"}]}
    assert backtest.run(df, spec).full["trades"] > 0


def test_random_and_mutated_specs_are_valid(df):
    rng = random.Random(1)
    for style in agents.STYLES:
        for _ in range(10):
            spec = agents.random_spec(style, rng)
            backtest.run(df, agents.mutate(spec, rng))


def test_extract_json_from_chatty_reply():
    reply = 'Sure! Here it is:\n```json\n{"name": "a", "entry_long": []}\n```\nGood luck.'
    assert strat.extract_json(reply)["name"] == "a"
    assert strat.extract_json('blah {"x": 1} blah')["x"] == 1


def test_load_csv_roundtrip(tmp_path, df):
    path = tmp_path / "p.csv"
    out = df.copy()
    out.index.name = "Date"
    out.columns = [c.title() for c in out.columns]
    out.to_csv(path)
    loaded = data.load_csv(path)
    assert list(loaded.columns) == strat.PRICE_COLUMNS
    assert len(loaded) == len(df)


class FakeLLM:
    """Scripted brain: one broken reply and one duplicate, to exercise the retries."""
    name = "fake"

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def chat(self, system, user, json_mode=False):
        self.calls += 1
        self.prompts.append(user)
        if "Librarian" in system:
            return '{"ideas": ["buy when RSI(2) < 10, sell when > 70"]}'
        if "Critic" in system or "Mayor" in system:
            return "- keep it simple"
        if self.calls == 2:
            return "not json at all"
        # Call 4 repeats call 3's rules, which the village must reject as a duplicate.
        level = 10 + 3 * (self.calls if self.calls != 4 else 3)
        return json.dumps({"name": "rsi2", "indicators": [{"id": "r", "type": "rsi", "period": 2}],
                           "entry_long": [{"left": "r", "op": "<", "right": level}],
                           "exit_long": [{"left": "r", "op": ">", "right": 70}]})


def test_village_end_to_end_with_fake_llm(tmp_path, df):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "notes.md").write_text("Short-term RSI mean reversion works on indices.")
    board = Board(tmp_path / "v.db")
    llm = FakeLLM()
    v = Village(df, "sample", board, llm=llm, seed=0, tuner=False, log=lambda *_: None)
    report = v.run(rounds=2, papers=str(papers), report_dir=str(tmp_path / "reports"))
    assert report.exists() and "rsi2" in report.read_text()
    assert list((tmp_path / "reports").glob("*.html"))
    assert board.notes("idea")
    assert any("already tested" in p for p in llm.prompts)
    rows = [r for r in board.all_strategies("sample") if "quant" in r["author"]]
    assert len(rows) == 6
    assert len({json.dumps(r["spec"]["entry_long"]) for r in rows}) == 6
    # Nova grafts invented features onto leaders and writes on the whiteboard.
    assert any(r["author"].startswith("Nova") for r in board.all_strategies("sample"))
    assert board.notes("whiteboard", 50, "sample")
    # A second run continues the round numbering instead of starting over.
    v.run(rounds=1, report_dir=str(tmp_path / "reports"))
    assert board.last_round() == 3
    board.close()


def test_village_runs_without_llm_on_two_markets(tmp_path, df):
    markets = {"a.csv": df, "b.csv": data.sample(900, seed=11)}
    board = Board(tmp_path / "v.db")
    Village(markets, "a.csv+b.csv", board, llm=None, seed=0, log=lambda *_: None).run(
        rounds=3, report_dir=str(tmp_path / "reports"))
    authors = [r["author"] for r in board.in_round("a.csv+b.csv", 3)]
    assert sum("quant" in a for a in authors) == 3
    best = board.leaderboard("a.csv+b.csv", 1)[0]
    assert set(best["markets"]) == {"a.csv", "b.csv"}
    # Tuner variations that were thrown away still count as tries.
    assert board.trial_count("a.csv+b.csv") >= len(board.all_strategies("a.csv+b.csv"))
    html = next((tmp_path / "reports").glob("*.html")).read_text()
    assert "<svg" in html and "vault" in html
    board.close()


def test_robust_sharpe_prefers_consistency():
    steady = backtest.robust_sharpe([0.8, 0.9, 1.0])
    lucky = backtest.robust_sharpe([3.0, -0.5, -0.4])
    assert steady > lucky


def test_evaluate_combines_markets(df):
    other = data.sample(900, seed=11)
    spec = agents.random_spec("trend", random.Random(2))
    ev = backtest.evaluate({"a": df, "b": other}, spec)
    a, b = backtest.run(df, spec), backtest.run(other, spec)
    assert ev["train"]["trades"] == a.train["trades"] + b.train["trades"]
    assert ev["train"]["max_drawdown_pct"] == min(a.train["max_drawdown_pct"],
                                                  b.train["max_drawdown_pct"])
    assert "robust_sharpe" in ev["train"]


def test_tuner_only_keeps_clear_improvements():
    tuner = agents.Tuner("t", rng=random.Random(0), tries=5)
    spec = agents.random_spec("trend", random.Random(0))
    best, tried = tuner.tune(spec, 1.0, lambda s: ({"train": {}}, 1.01))
    assert best is None and tried == 5
    best, _ = tuner.tune(spec, 1.0, lambda s: ({"train": {}}, 2.0))
    assert best[0]["name"].endswith("(tuned)")


def test_old_database_is_upgraded(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE notes (id INTEGER PRIMARY KEY, round INTEGER, author TEXT, kind TEXT,
                            content TEXT, created REAL);
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, round INTEGER, author TEXT, name TEXT,
            spec TEXT, train TEXT, test TEXT, score REAL, error TEXT, dataset TEXT, created REAL);
        INSERT INTO strategies (round, author, name, spec, train, test, score, dataset)
            VALUES (1, 'x', 'old', '{}', '{}', '{}', 0.5, 'sample');
    """)
    db.commit()
    db.close()
    board = Board(path)
    assert board.leaderboard("sample")[0]["markets"] is None
    board.post(2, "c", "critique", "hi", "sample")
    assert board.notes("critique", 5, "sample")[0]["content"] == "hi"
    board.close()


def test_fetch_flattens_yahoo_columns(monkeypatch, df):
    import sys
    import types
    raw = df.rename(columns=str.title)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["SPY"]])
    fake = types.SimpleNamespace(download=lambda *a, **k: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = data.fetch("SPY")
    assert list(out.columns) == strat.PRICE_COLUMNS and len(out) == len(df)
    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame()))
    with pytest.raises(ValueError, match="no data"):
        data.fetch("NOPE")


# ---- ICT building blocks -----------------------------------------------------------------

from village import ict  # noqa: E402

ICT_KINDS = [k for k in ict.CATALOG]


@pytest.fixture
def intraday():
    return data.sample(1200, seed=5, freq="15min")


@pytest.mark.parametrize("kind", ICT_KINDS)
def test_ict_indicators_never_look_ahead(kind, intraday):
    cut = 700
    full = indicators.compute(intraday, kind, {})
    future = intraday.copy()
    future.iloc[cut:, :4] = future.iloc[cut:, :4].to_numpy()[::-1]  # scramble the future
    changed = indicators.compute(future, kind, {})
    pd.testing.assert_frame_equal(full.iloc[:cut], changed.iloc[:cut])


def candles(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 0.0
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")
    return df


def test_fvg_finds_a_bullish_gap_and_drops_it_when_closed_through():
    df = candles([(10, 11, 9, 10.5), (10.5, 14, 10.4, 13.8), (13.8, 15, 12, 14.5),
                  (14.5, 14.6, 11.5, 12.5), (12.5, 12.6, 10, 10.2)])
    g = ict.fvg(df)
    assert g["bull"].tolist() == [0, 0, 1, 0, 0]
    assert g["bull_bot"].iloc[2] == 11 and g["bull_top"].iloc[2] == 12   # high[t-2]..low[t]
    assert g["bull_top"].iloc[3] == 12      # still live while price trades into it
    assert np.isnan(g["bull_top"].iloc[4])  # closed below the gap: gone


def test_sweep_needs_a_wick_through_and_a_close_back_inside():
    rows = [(10, 10.5, 9.5, 10)] * 3 + [(10, 10.5, 8, 10)] + [(10, 10.5, 9.5, 10)] * 3
    rows += [(10, 10.2, 7.5, 9)]  # wick below the swing low at 8 and close back above it
    s = ict.sweep(candles(rows), k=3)
    assert s["bull"].iloc[-1] == 1 and s["bull"].iloc[:-1].sum() == 0


def test_session_uses_new_york_time(intraday):
    s = ict.session(intraday)
    # 2024-01-01 07:00 UTC is 02:00 in New York (winter): start of the London kill zone.
    t = pd.Timestamp("2024-01-01 07:00")
    assert s.loc[t, "hour"] == 2 and s.loc[t, "london"] == 1
    ny = intraday.copy()
    ny.attrs["tz"] = "America/New_York"
    assert ict.session(ny).loc[t, "ny_am"] == 1


def test_within_chains_events_over_time():
    df = data.sample(400, seed=1)
    spec = {"indicators": [{"id": "r", "type": "rsi", "period": 2}],
            "entry_long": [{"left": "r", "op": "<", "right": 10}], "exit_long": []}
    now = strat.signals(df, spec)["entry_long"]
    spec["entry_long"][0]["within"] = 5
    later = strat.signals(df, spec)["entry_long"]
    assert later.sum() > now.sum() and (later | ~now).all()
    with pytest.raises(strat.SpecError, match="within"):
        spec["entry_long"][0]["within"] = 0
        strat.signals(df, spec)


def test_random_ict_specs_run_and_mutation_keeps_flags_exact(intraday):
    rng = random.Random(4)
    for style in ("ict_liquidity", "ict_blocks", "ict_time"):
        for _ in range(4):
            spec = agents.random_spec(style, rng, intraday=True)
            backtest.run(intraday, spec)
            m = agents.mutate(spec, rng)
            for key in strat.RULE_KEYS:
                for cond in m.get(key) or []:
                    if cond["op"] == "==":
                        assert cond["right"] == 1
            backtest.run(intraday, m)


def test_analyst_finds_the_piece_that_matters():
    spec = {"entry_long": [{"left": "a", "op": ">", "right": 0},
                           {"left": "b", "op": ">", "right": 0}], "exit_long": []}

    def score(s):  # only condition "a" carries the edge
        return 1.0 if any(c["left"] == "a" for c in s["entry_long"]) else 0.0

    rows = agents.Analyst.ablate(spec, score)
    assert rows[0]["condition"].startswith("a") and rows[0]["impact"] == 1.0
    assert "ESSENTIAL" in agents.Analyst.summarise("x", 1.0, rows)


def test_ict_team_runs_end_to_end(tmp_path, intraday):
    board = Board(tmp_path / "v.db")
    v = Village(intraday, "fx", board, llm=None, quants=agents.TEAMS["ict"], seed=1,
                mission=agents.ICT_MISSION, log=lambda *_: None)
    report = v.run(rounds=3, report_dir=str(tmp_path / "r"))
    assert {r["author"] for r in board.all_strategies("fx")} >= {"Ivy the liquidity hunter"}
    assert report.exists()
    board.close()


def test_short_noisy_vault_is_flagged_as_possible_luck():
    from village.village import verdict
    train = {"sharpe": 1.0}
    assert verdict(train, {"sharpe": 0.9, "trades": 30, "t_stat": 3.0}) == "held up"
    assert verdict(train, {"sharpe": 0.9, "trades": 30, "t_stat": 1.1}) == "held up, but could be luck"
    assert backtest.luck_bar(1000) > backtest.luck_bar(10)


# ---- Pine Script export --------------------------------------------------------------------

from village import pine  # noqa: E402


def _all_indicator_spec():
    inds = [{"id": f"x{i}", "type": k} for i, k in enumerate([*indicators.CATALOG, *ict.CATALOG])]
    return {"name": 'all "kinds"', "indicators": inds,
            "entry_long": [{"left": "close", "op": "crosses_above", "right": "x0", "within": 3}],
            "exit_long": [{"left": "x2", "op": ">", "right": 70}],
            "entry_short": [{"left": "x12.bear", "op": "==", "right": 1}], "exit_short": [],
            "stop_loss_pct": 2, "take_profit_pct": 4}


def test_pine_covers_every_indicator_and_rule():
    src = pine.to_pine(_all_indicator_spec(), 5)
    assert src.startswith("//@version=5") and "strategy(" in src
    assert 'all \'kinds\'' in src            # quotes in names cannot break the script
    assert "f_within(f_xup(close, i_x0), 3)" in src
    assert "commission_value=0.05" in src and "stopPct = 0.02" in src


@pytest.mark.skipif(not __import__("os").environ.get("VILLAGE_SLOW"),
                    reason="slow (about a minute); run with VILLAGE_SLOW=1")
def test_pine_is_valid_syntax():
    parser = pytest.importorskip("pynescript.ast")
    rng = random.Random(0)
    # The parser is slow, so check the spec that uses everything plus two random sequences.
    specs = [agents.random_spec(s, rng, intraday=True) for s in ("ict_liquidity", "ict_time")]
    nova = agents.Inventor("n", rng=random.Random(5), intraday=True)
    every_function = {"name": "fx", "indicators": [{"id": "sw", "type": "sweep"}], "features": [
        {"id": f"f{i}", "expr": e} for i, e in enumerate([
            "bars_since(sw.bull)", "minutes_since(sw.bear)", "count(sw.bull, 9)", "day_count(sw.bull)",
            "sum(close, 3) + mean(close, 3) - highest(high, 5) * lowest(low, 5) / std(close, 5)",
            "prev(close, 2) % 7 + change(close)", "digital_root(f2) + abs(-1) + round(1.5) + floor(2.2)",
            "min(open, close) - max(open, close)", "where(not sw.bull, 1, -1)",
            "ny_minute() >= 570 and day_of_week() <= 4 or close != open",
            "minute_of_hour() + bars_today() + session_high(120, 300) - session_low(0, 570)",
            "value_at(open, 0) + value_at(close * 2, 570)"])],
        "entry_long": [{"left": "f9", "op": "==", "right": 1}], "exit_long": []}
    specs += [nova.graft(specs[0])[0], every_function]
    for spec in specs + [_all_indicator_spec()]:
        parser.parse(pine.to_pine(spec))


def test_formats_from_common_exporters_load(tmp_path):
    df = data.sample(600, seed=3, freq="1min")
    d = df.index
    o, h, l, c, v = (df[x].round(2) for x in strat.PRICE_COLUMNS)
    files = {
        "tradestation.txt": (pd.DataFrame({"Date": d.strftime("%m/%d/%Y"), "Time": d.strftime("%H%M"),
                                           "Open": o, "High": h, "Low": l, "Close": c, "Up": v}), {}),
        "ninja.txt": (pd.DataFrame({0: d.strftime("%Y%m%d %H%M%S"), 1: o, 2: h, 3: l, 4: c, 5: v}),
                      {"sep": ";", "header": False}),
        "kibot.txt": (pd.DataFrame({0: d.strftime("%m/%d/%Y"), 1: d.strftime("%H:%M"), 2: o, 3: h,
                                    4: l, 5: c, 6: v}), {"header": False}),
        "mt5.csv": (pd.DataFrame({"<DATE>": d.strftime("%Y.%m.%d"), "<TIME>": d.strftime("%H:%M:%S"),
                                  "<OPEN>": o, "<HIGH>": h, "<LOW>": l, "<CLOSE>": c}), {"sep": "\t"}),
    }
    for name, (frame, kw) in files.items():
        frame.to_csv(tmp_path / name, index=False, **kw)
        x = data.load_csv(tmp_path / name)
        assert len(x) == 600 and x.index[0] == d[0] and x.index[-1] == d[-1], name
        assert len(data.load_csv(tmp_path / name, timeframe="2min")) == 300


# ---- invented features, whiteboard, ladder ---------------------------------------------------

from village import features  # noqa: E402


def _cols(values, freq="15min"):
    idx = pd.date_range("2024-01-01 03:00", periods=len(values), freq=freq)
    return pd.DataFrame({"flag": np.array(values, float), "close": np.arange(len(values)) + 1.0},
                        index=idx)


def test_feature_functions():
    cols = _cols([0, 1, 0, 0, 1, 0])
    assert features.evaluate("bars_since(flag)", cols).tolist()[1:] == [0, 1, 2, 0, 1]
    assert np.isnan(features.evaluate("bars_since(flag)", cols).iloc[0])
    assert features.evaluate("minutes_since(flag)", cols).tolist()[1:] == [0, 15, 30, 0, 15]
    assert features.evaluate("count(flag, 3)", cols).tolist()[2:] == [1, 1, 1, 1]
    assert features.evaluate("prev(close, 2)", cols).iloc[2] == 1
    dr = features.evaluate("digital_root(close * 9 + 1)", cols)   # 10, 19, 28 ... -> 1
    assert set(dr) == {1.0}
    assert features.evaluate("digital_root(close - 1)", cols).iloc[0] == 0
    both = features.evaluate("flag == 1 and close > 3", cols)
    assert both.tolist() == [0, 0, 0, 0, 1, 0]


def test_day_count_restarts_each_new_york_day():
    cols = _cols([1] * 8, freq="2h")   # 03:00 UTC = 22:00 New York the day before
    counts = features.evaluate("day_count(flag)", cols)
    ny = cols.index.tz_localize("UTC").tz_convert("America/New_York")
    first_new_day = list(ny.day).index(ny.day[-1])
    assert counts.iloc[first_new_day] == 1                       # restarted at NY midnight
    assert counts.iloc[-1] == len(cols) - first_new_day          # and kept counting that day


@pytest.mark.parametrize("bad", ["__import__('os')", "close.__class__", "open[1]", "x if y else z",
                                 "count(close, n)", "foo(1)", "'text'", "(lambda: 1)()"])
def test_formulas_cannot_run_code(bad):
    with pytest.raises(features.FeatureError):
        features.evaluate(bad, _cols([0, 1, 0]))


def test_features_never_look_ahead(intraday):
    spec = {"indicators": [{"id": "sw", "type": "sweep"}],
            "features": [{"id": "a", "expr": "digital_root(count(sw.bull, 30)) + minutes_since(sw.bear)"},
                         {"id": "b", "expr": "day_count(sw.bull) * ny_minute() + change(close, 4)"}],
            "entry_long": [{"left": "a", "op": ">", "right": 3}], "exit_long": []}
    cut = 700
    full = strat.build_columns(intraday, spec)
    future = intraday.copy()
    future.iloc[cut:, :4] = future.iloc[cut:, :4].to_numpy()[::-1]
    pd.testing.assert_frame_equal(full.iloc[:cut], strat.build_columns(future, spec).iloc[:cut])


def test_pine_translates_features():
    spec = {"name": "f", "indicators": [{"id": "sw", "type": "sweep"}],
            "features": [{"id": "dr", "expr": "digital_root(count(sw.bull, 50))"},
                         {"id": "w", "expr": "minutes_since(sw.bull) + day_count(sw.bear)"}],
            "entry_long": [{"left": "dr", "op": "==", "right": 7}], "exit_long": []}
    src = pine.to_pine(spec)
    assert "i_dr = ft_dr_" in src and "ta.valuewhen" in src and "math.sum" in src


def test_never_trading_and_lookalike_strategies_are_rejected(tmp_path, df):
    v = Village(df, "s", Board(tmp_path / "v.db"), llm=None, log=lambda *_: None)
    never = {"indicators": [], "entry_long": [{"left": "close", "op": "<", "right": 0}],
             "exit_long": []}
    assert "never trades" in v._try(never)[1]
    base = always_long()
    ev, err = v._try(base)
    v._record(1, "x", base, ev)
    same = dict(base, entry_long=base["entry_long"] + [{"left": "close", "op": ">", "right": -1}])
    assert "trades exactly like" in v._try(same)[1]


def test_inventor_grafts_a_valid_feature(intraday):
    nova = agents.Inventor("n", rng=random.Random(3), intraday=True)
    spec = agents.random_spec("ict_liquidity", random.Random(1))
    for _ in range(10):
        grafted, idea = nova.graft(spec)
        assert grafted["features"] and "What if" in idea
        strat.build_columns(intraday, grafted)


def test_ladder_scouts_then_expands(tmp_path):
    from village import stages
    df = data.sample(12000, seed=4, freq="15min")
    board = Board(tmp_path / "v.db")
    survivors, table = stages.ladder({"m": df}, "m", board, "30D", ["60D", "all"], scout_rounds=2,
                                     stage_rounds=1, log=lambda *_: None,
                                     report_dir=str(tmp_path / "r"), quants=agents.TEAMS["ict"],
                                     seed=1)
    assert table[0] == "| Setup | 30D | 60D | all |"
    windows = {r["dataset"] for r in board.db.execute("SELECT dataset FROM strategies")}
    assert windows == {"m | scout 30D", "m | expand 60D", "m | expand all"}
    assert survivors and len(table) > 2
    # Survivors of each step were carried into the next window's board.
    seeded = board.db.execute("SELECT COUNT(*) FROM strategies WHERE author LIKE 'from %'").fetchone()
    assert seeded[0] > 0
    # Every window ends exactly where the vault begins, so the vault stays unseen.
    train_end = df.index[int(len(df) * 0.7) - 1]
    for span in ("30D", "60D", "all"):
        assert stages.window(df, stages.parse_span(span), 0.7).index[-1] == train_end
    board.close()


def test_library_reads_pine_and_subtitles(tmp_path):
    from village import pdfs
    (tmp_path / "ind.pine").write_text("//@version=5\nindicator('x')\nplot(ta.sma(close, 20))")
    (tmp_path / "talk.srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nPrice runs the stops\n\n"
                                       "2\n00:01:05,500 --> 00:01:07,000\n<i>then it shifts</i>\n")
    docs = {p.name for p in pdfs.find_documents(tmp_path)}
    assert docs == {"ind.pine", "talk.srt"}
    text = pdfs.read_document(tmp_path / "talk.srt")
    assert "Price runs the stops" in text and "then it shifts" in text and "-->" not in text
    assert "[00:01]" in text
    assert pdfs.is_pine(tmp_path / "ind.pine")


def test_librarian_uses_pine_prompt_for_indicators(tmp_path):
    (tmp_path / "ind.pine").write_text("//@version=5\nindicator('x')")
    seen = []

    class LLM:
        name = "spy"

        def chat(self, system, user, json_mode=False):
            seen.append(system)
            return '{"ideas": ["sma 20 cross"]}'

    board = Board(tmp_path / "v.db")
    assert agents.Librarian("L", LLM()).study(tmp_path, board, 1) == 1
    assert "Pine Script source code" in seen[0]
    board.close()


# ---- clock functions, crossbreeding, live view -----------------------------------------------

def _hourly():
    idx = pd.date_range("2024-01-02 05:00", periods=24, freq="1h")   # 00:00 New York time
    base = np.arange(24) + 100.0
    return pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base + 0.5},
                        index=idx)


def test_clock_functions():
    df = _hourly()
    lon_hi = features.evaluate("session_high(120, 300)", df)
    assert lon_hi.iloc[:2].isna().all()                    # before 02:00 New York: no value yet
    assert lon_hi.iloc[2:5].tolist() == [103, 104, 105]    # running high inside the window
    assert lon_hi.iloc[10] == 105                          # kept after the window closes
    assert features.evaluate("session_low(120, 300)", df).iloc[6] == 101
    assert set(features.evaluate("value_at(open, 0)", df)) == {100.0}      # midnight open
    nine = features.evaluate("value_at(open, 570)", df)
    assert nine.iloc[:10].isna().all() and nine.iloc[10] == 110            # first bar at/after 09:30
    assert features.evaluate("bars_today()", df).tolist()[:4] == [0, 1, 2, 3]
    with pytest.raises(features.FeatureError):
        features.parse("session_high(300, 120)")


def test_pine_translates_clock_functions():
    spec = {"name": "clock", "indicators": [],
            "features": [{"id": "a", "expr": "close > value_at(open, 0) and low < session_low(120, 300)"}],
            "entry_long": [{"left": "a", "op": "==", "right": 1}], "exit_long": []}
    src = pine.to_pine(spec)
    assert 'hour(time, "America/New_York") * 60' in src and "math.min(" in src


def test_crossbreed_combines_two_parents(intraday):
    nova = agents.Inventor("n", rng=random.Random(2), intraday=True)
    a = agents.random_spec("ict_liquidity", random.Random(1), intraday=True)
    b = agents.random_spec("ict_blocks", random.Random(2), intraday=True)
    child, idea = nova.crossbreed(a, b)
    assert "crossbreed" in idea and " x " in child["name"]
    assert any(i["id"].startswith("x") for i in child["indicators"])
    strat.build_columns(intraday, child)


def test_nova_designs_a_strategy_from_her_hypothesis(tmp_path, intraday):
    class LLM:
        name = "fake"

        def chat(self, system, user, json_mode=False):
            if "Hypothesis to test" in user:
                return json.dumps({"name": "macro test", "indicators": [], "features": [
                    {"id": "m", "expr": "ny_minute() >= 590 and ny_minute() < 610"}],
                    "entry_long": [{"left": "m", "op": "==", "right": 1}],
                    "exit_long": [{"left": "m", "op": "==", "right": 0}]})
            if "hypotheses" in user:
                return '{"hypotheses": ["the 09:50 macro delivers up"]}'
            if "Librarian" in system or "Critic" in system or "Mayor" in system:
                return "- ok"
            return json.dumps(always_long())

    board = Board(tmp_path / "v.db")
    Village(intraday, "fx", board, llm=LLM(), seed=0, tuner=False, log=lambda *_: None).run(
        rounds=1, report_dir=str(tmp_path / "r"))
    names = [r["name"] for r in board.all_strategies("fx") if r["author"].startswith("Nova")]
    assert "macro test" in names
    assert any("Hypothesis" in n["content"] for n in board.notes("whiteboard", 50, "fx"))
    board.close()


def test_live_view_serves_the_village(tmp_path, intraday):
    import urllib.request
    from village import live
    board = Board(tmp_path / "v.db")
    Village(intraday, "s", board, llm=None, seed=0, log=lambda *_: None).run(
        rounds=2, report_dir=str(tmp_path / "r"))
    board.close()
    st = live.state(str(tmp_path / "v.db"))
    assert st["ready"] and st["round"] == 2 and st["tested"] > 0
    names = {v["name"] for v in st["villagers"]}
    assert {"Tom", "Rita", "Bo", "Tess", "Ada", "Nova"} <= names
    assert any(v["text"] for v in st["villagers"])          # speech bubbles have something to say
    assert st["whiteboard"] and st["leaders"]
    assert not live.state(str(tmp_path / "none.db"))["ready"]
    url = live.serve(str(tmp_path / "v.db"), port=8890, open_browser=False, background=True)
    page = urllib.request.urlopen(url).read().decode()
    assert "Agent Village" in page and "whiteboard" in page.lower()
    data_ = json.loads(urllib.request.urlopen(url + "state").read())
    assert data_["dataset"] == "s"


def test_auto_brain_uses_an_installed_ollama_model(monkeypatch):
    from village import llm
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm.OllamaLLM, "installed", staticmethod(lambda host=None: ["qwen2.5:14b"]))
    assert llm.make("auto").model == "qwen2.5:14b"
    monkeypatch.setattr(llm.OllamaLLM, "installed", staticmethod(lambda host=None: []))
    assert llm.make("auto") is None


def test_day_first_dates_are_not_mistaken_for_month_first(tmp_path):
    # The first ~1300 rows only have days 1-12, so they look month-first; row 1337 does not.
    idx = pd.date_range("2008-12-11 18:00", periods=3000, freq="15min")
    for fmt in ("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        pd.DataFrame({"Date": idx.strftime(fmt), "Open": 1.0, "High": 2.0, "Low": 0.5,
                      "Close": 1.5}).to_csv(tmp_path / "x.csv", index=False)
        x = data.load_csv(tmp_path / "x.csv")
        assert x.index[0] == idx[0] and x.index[-1] == idx[-1] and len(x) == 3000, fmt
