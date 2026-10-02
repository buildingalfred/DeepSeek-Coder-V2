"""The village loop: read, propose, backtest, critique, tune, repeat. Then open the vault."""

import json
import random
import time
from pathlib import Path

from . import backtest, data, report
from . import strategy as strat
from .agents import Critic, Librarian, Mayor, Quant, Tuner, compact, leaderboard_text
from .board import Board

MAX_ATTEMPTS_LLM = 3        # an LLM quant gets this many tries to produce a valid, new strategy
MAX_ATTEMPTS_HEURISTIC = 8  # random villagers are cheap, let them retry more on duplicates


def fingerprint(spec: dict) -> str:
    """Identity of a strategy's rules, ignoring its name and description."""
    core = {k: v for k, v in spec.items() if k not in ("name", "idea") and not k.startswith("_")}
    for key in strat.RULE_KEYS:
        core.setdefault(key, [])
    core.setdefault("stop_loss_pct", None)
    core.setdefault("take_profit_pct", None)
    return json.dumps(core, sort_keys=True)


class Village:
    def __init__(self, markets, dataset: str, board: Board, llm=None, quants=None, fee_bps=5.0,
                 train_frac=0.7, seed=None, tuner=True, log=print):
        self.markets = markets if isinstance(markets, dict) else {dataset: markets}
        self.dataset = dataset
        self.board = board
        self.llm = llm
        self.fee_bps = fee_bps
        self.train_frac = train_frac
        self.log = log
        rng = random.Random(seed)
        self.librarian = Librarian("Lena the Librarian", llm, log)
        self.critic = Critic("Carl the Critic", llm, log)
        self.mayor = Mayor("Maya the Mayor", llm, log)
        self.tuner = Tuner("Tess the Tuner", None, log, random.Random(rng.random())) if tuner else None
        styles = quants or ["trend", "reversion", "breakout"]
        names = {"trend": "Tom", "reversion": "Rita", "breakout": "Bo"}
        self.quants = []
        for i, s in enumerate(styles):
            name = f"{names.get(s, s.title())} the {s} quant"
            if styles[:i].count(s):
                name += f" {styles[:i].count(s) + 1}"
            self.quants.append(Quant(name, s, llm, log, random.Random(rng.random())))
        # Agents only ever see the train period; the rest stays sealed in the vault.
        self.data_summary = "\n".join(
            data.summary(df.iloc[: int(len(df) * train_frac)], f"{name} (train period only)")
            for name, df in self.markets.items())
        self.seen = {fingerprint(r["spec"]) for r in board.all_strategies(dataset) if r["spec"]}

    def evaluate(self, spec: dict) -> dict:
        return backtest.evaluate(self.markets, spec, self.fee_bps, self.train_frac)

    def score(self, evaluation: dict) -> float:
        return backtest.score(evaluation, len(self.markets))

    def run(self, rounds: int = 5, papers: str | None = None, report_dir: str = "reports") -> Path:
        start_round = self.board.last_round()
        brain = self.llm.name if self.llm else "no LLM (heuristic villagers)"
        self.log(f"Village waking up. Brain: {brain}.\nData: {self.data_summary}")
        for r in range(start_round + 1, start_round + rounds + 1):
            self.log(f"\n=== Round {r} ===")
            if papers:
                n = self.librarian.study(papers, self.board, r)
                if n:
                    self.log(f"  [{self.librarian.name}] posted {n} new ideas")
            for quant in self.quants:
                self._turn(quant, r)
            if self.tuner:
                self._tune(r)
            critique = self.critic.review(self.board.in_round(self.dataset, r),
                                          leaderboard_text(self.board.leaderboard(self.dataset, 5)))
            self.board.post(r, self.critic.name, "critique", critique, self.dataset)
            self.log(f"  [{self.critic.name}]\n" + _indent(critique))
        return self.report(report_dir)

    def _context(self, quant: Quant) -> dict:
        top = self.board.leaderboard(self.dataset, 5)
        mine = []
        for row in self.board.by_author(self.dataset, quant.name, 3):
            result = f"ERROR: {row['error']}" if row["error"] else str(row["train"])
            mine.append(f"- {compact(row['spec'])}\n  -> {result}")
        critiques = self.board.notes("critique", 1, self.dataset)
        return {
            "data_summary": self.data_summary,
            "ideas": "\n".join(f"- {n['content']}" for n in self.board.notes("idea", 15)),
            "leaderboard": leaderboard_text(top),
            "critique": critiques[-1]["content"] if critiques else "",
            "mine": "\n".join(mine),
            "best_specs": [row["spec"] for row in top],
        }

    def _turn(self, quant: Quant, round_: int) -> None:
        ctx = self._context(quant)
        attempts = MAX_ATTEMPTS_LLM if quant.llm is not None else MAX_ATTEMPTS_HEURISTIC
        error = previous = None
        for attempt in range(attempts):
            spec = quant.propose(ctx, error=error, previous=previous)
            previous = (spec.get("_raw") or json.dumps(spec))[:4000]
            evaluation, error = self._try(spec)
            if not error:
                break
            if attempt + 1 < attempts and quant.llm is not None:
                self.log(f"  [{quant.name}] rejected ({error[:100]}), retrying")
        clean = {k: v for k, v in spec.items() if not k.startswith("_")}
        if error:
            self.board.add_strategy(round_, quant.name, clean, None, None, None, error, self.dataset)
            self.log(f"  [{quant.name}] gave up this round: {error[:160]}")
            return
        self._record(round_, quant.name, clean, evaluation)

    def _try(self, spec: dict):
        if "_parse_error" in spec:
            return None, spec["_parse_error"]
        if fingerprint(spec) in self.seen:
            return None, ("these exact rules were already tested; change the indicators, "
                          "parameters or conditions to try something new")
        try:
            return self.evaluate(spec), None
        except strat.SpecError as e:
            return None, str(e)

    def _record(self, round_: int, author: str, spec: dict, evaluation: dict) -> int:
        score = self.score(evaluation)
        sid = self.board.add_strategy(round_, author, spec, evaluation["train"], evaluation["test"],
                                      score, None, self.dataset, evaluation["markets"])
        self.seen.add(fingerprint(spec))
        t = evaluation["train"]
        self.log(f"  [{author}] #{sid} '{spec.get('name', 'unnamed')}': robust {t['robust_sharpe']}, "
                 f"sharpe {t['sharpe']}, return {t['total_return_pct']}%, trades {t['trades']}, "
                 f"good periods {t['positive_periods']} (train)")
        return sid

    def _tune(self, round_: int) -> None:
        top = [r for r in self.board.leaderboard(self.dataset, 3) if r["score"] > -99]
        if not top:
            return
        parent = top[(round_ - 1) % len(top)]
        best, tried = self.tuner.tune(parent["spec"], parent["score"], self._tuner_eval)
        # Discarded variations still count as tries (the kept one is counted as a strategy).
        discarded = tried - (best is not None)
        self.board.post(round_, self.tuner.name, "trials", str(discarded), self.dataset)
        if best is None:
            self.log(f"  [{self.tuner.name}] tried {tried} variations of #{parent['id']}, "
                     "none beat it")
            return
        spec, evaluation = best
        self._record(round_, self.tuner.name, spec, evaluation)

    def _tuner_eval(self, spec: dict):
        if fingerprint(spec) in self.seen:
            return None
        try:
            evaluation = self.evaluate(spec)
        except strat.SpecError:
            return None
        return evaluation, self.score(evaluation)

    def report(self, report_dir: str = "reports", top_n: int = 10) -> Path:
        rows = self.board.leaderboard(self.dataset, top_n)
        trials = self.board.trial_count(self.dataset)
        for r in rows:
            r["verdict"] = verdict(r["train"], r["test"])
        table = "\n".join(
            f"{i}. '{r['name']}' ({strat.describe(r['spec'])})\n   train: {r['train']}\n"
            f"   test: {r['test']}\n   verdict: {r['verdict']}" for i, r in enumerate(rows, 1))
        story = self.mayor.summarise(table) if rows else None
        held = [r for r in rows if r["verdict"] == "held up"]
        positive = [r for r in rows if r["train"]["sharpe"] > 0]
        best = held[0] if held else (positive[0] if positive else None)
        brain = self.llm.name if self.llm else "none (heuristic)"

        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        md = report.markdown(self.dataset, rows, best, story, trials, brain, self.fee_bps)
        path = out / f"report-{stamp}.md"
        path.write_text(md, encoding="utf-8")
        charts = [(r, self._curves(r["spec"])) for r in (rows[:3] if rows else [])]
        html_path = out / f"report-{stamp}.html"
        html_path.write_text(report.html(self.dataset, rows, best, story, trials, brain,
                                         self.fee_bps, charts, self.train_frac), encoding="utf-8")
        if best:
            (out / "best_strategy.json").write_text(json.dumps(best["spec"], indent=2), "utf-8")
        self.log(f"\nReport written to {html_path} (open it in a browser) and {path}")
        return path

    def _curves(self, spec: dict) -> dict:
        """Equity curve per market (plus buy & hold when there is a single market)."""
        curves = {}
        for name, df in self.markets.items():
            res = backtest.run(df, spec, self.fee_bps, self.train_frac)
            curves[name] = res.equity
        if len(self.markets) == 1:
            df = next(iter(self.markets.values()))
            curves["buy & hold"] = df["close"] / df["close"].iloc[0]
        return curves


def verdict(train: dict, test: dict) -> str:
    if train["sharpe"] <= 0:
        return "never worked"
    if test["trades"] < 3:
        return "too few test trades"
    if test["sharpe"] <= 0:
        return "failed in vault"
    if test["sharpe"] < 0.5 * train["sharpe"]:
        return "weaker out of sample"
    return "held up"


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())
