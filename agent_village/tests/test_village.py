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
