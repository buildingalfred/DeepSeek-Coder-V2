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
        period = 2 + (self.calls if self.calls != 4 else 3)
        return json.dumps({"name": "rsi2", "indicators": [{"id": "r", "type": "rsi", "period": period}],
                           "entry_long": [{"left": "r", "op": "<", "right": 10}],
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
    rows = board.all_strategies("sample")
    assert len(rows) == 6
    assert len({json.dumps(r["spec"]["indicators"]) for r in rows}) == 6
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
