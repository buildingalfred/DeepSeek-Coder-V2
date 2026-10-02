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
    """Scripted brain: first reply is broken, later replies are valid, to exercise the retry."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    def chat(self, system, user, json_mode=False):
        self.calls += 1
        if "Librarian" in system:
            return '{"ideas": ["buy when RSI(2) < 10, sell when > 70"]}'
        if "Critic" in system or "Mayor" in system:
            return "- keep it simple"
        if self.calls == 2:
            return "not json at all"
        return json.dumps({"name": "rsi2", "indicators": [{"id": "r", "type": "rsi", "period": 2}],
                           "entry_long": [{"left": "r", "op": "<", "right": 10}],
                           "exit_long": [{"left": "r", "op": ">", "right": 70}]})


def test_village_end_to_end_with_fake_llm(tmp_path, df):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "notes.md").write_text("Short-term RSI mean reversion works on indices.")
    board = Board(tmp_path / "v.db")
    v = Village(df, "sample", board, llm=FakeLLM(), seed=0, log=lambda *_: None)
    report = v.run(rounds=2, papers=str(papers), report_dir=str(tmp_path / "reports"))
    assert report.exists() and "rsi2" in report.read_text()
    assert board.notes("idea")
    assert len(board.leaderboard("sample")) >= 5
    # A second run continues the round numbering instead of starting over.
    v.run(rounds=1, report_dir=str(tmp_path / "reports"))
    assert board.last_round() == 3
    board.close()


def test_village_runs_without_llm(tmp_path, df):
    board = Board(tmp_path / "v.db")
    Village(df, "sample", board, llm=None, seed=0, log=lambda *_: None).run(
        rounds=2, report_dir=str(tmp_path / "reports"))
    assert len(board.in_round("sample", 2)) == 3
    board.close()
