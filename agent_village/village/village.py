"""The village loop: read, propose, backtest, critique, repeat. Then open the vault."""

import json
import random
import time
from pathlib import Path

from . import backtest, data
from . import strategy as strat
from .agents import Critic, Librarian, Mayor, Quant, compact, leaderboard_text
from .board import Board


class Village:
    def __init__(self, df, dataset: str, board: Board, llm=None, quants=None, fee_bps=5.0,
                 train_frac=0.7, seed=None, log=print):
        self.df = df
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
        styles = quants or ["trend", "reversion", "breakout"]
        names = {"trend": "Tom", "reversion": "Rita", "breakout": "Bo"}
        self.quants = [Quant(f"{names.get(s, s.title())} the {s} quant", s, llm, log,
                             random.Random(rng.random())) for s in styles]
        # Agents only ever see the train period; the rest stays sealed in the vault.
        train_df = df.iloc[: int(len(df) * train_frac)]
        self.data_summary = data.summary(train_df, f"{dataset} (train period only)")

    def run(self, rounds: int = 5, papers: str | None = None, report_dir: str = "reports") -> Path:
        start_round = self.board.last_round()
        brain = self.llm.name if self.llm else "no LLM (heuristic villagers)"
        self.log(f"Village waking up. Brain: {brain}. Data: {self.data_summary}")
        for r in range(start_round + 1, start_round + rounds + 1):
            self.log(f"\n=== Round {r} ===")
            if papers:
                n = self.librarian.study(papers, self.board, r)
                if n:
                    self.log(f"  [{self.librarian.name}] posted {n} new ideas")
            for quant in self.quants:
                self._turn(quant, r)
            critique = self.critic.review(self.board.in_round(self.dataset, r),
                                          leaderboard_text(self.board.leaderboard(self.dataset, 5)))
            self.board.post(r, self.critic.name, "critique", critique)
            self.log(f"  [{self.critic.name}]\n" + _indent(critique))
        return self.report(report_dir)

    def _context(self, quant: Quant) -> dict:
        top = self.board.leaderboard(self.dataset, 5)
        mine = []
        for row in self.board.by_author(self.dataset, quant.name, 3):
            result = f"ERROR: {row['error']}" if row["error"] else str(row["train"])
            mine.append(f"- {compact(row['spec'])}\n  -> {result}")
        critiques = self.board.notes("critique", 1)
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
        spec = quant.propose(ctx)
        result, error = self._try(spec)
        if error and quant.llm is not None:
            self.log(f"  [{quant.name}] spec rejected ({error[:100]}), retrying")
            previous = spec.get("_raw") or json.dumps(spec)
            spec = quant.propose(ctx, error=error, previous=previous[:4000])
            result, error = self._try(spec)
        clean = {k: v for k, v in spec.items() if not k.startswith("_")}
        if error:
            self.board.add_strategy(round_, quant.name, clean, None, None, None, error, self.dataset)
            self.log(f"  [{quant.name}] invalid strategy: {error[:160]}")
            return
        score = backtest.score(result)
        sid = self.board.add_strategy(round_, quant.name, clean, result.train, result.test, score,
                                      None, self.dataset)
        self.log(f"  [{quant.name}] #{sid} '{clean.get('name', 'unnamed')}': "
                 f"sharpe {result.train['sharpe']}, return {result.train['total_return_pct']}%, "
                 f"trades {result.train['trades']} (train)")

    def _try(self, spec: dict):
        if "_parse_error" in spec:
            return None, spec["_parse_error"]
        try:
            return backtest.run(self.df, spec, self.fee_bps, self.train_frac), None
        except strat.SpecError as e:
            return None, str(e)

    def report(self, report_dir: str = "reports", top_n: int = 10) -> Path:
        rows = self.board.leaderboard(self.dataset, top_n)
        lines = [f"# Village report: {self.dataset}", "",
                 f"{time.strftime('%Y-%m-%d %H:%M')} · brain: "
                 f"{self.llm.name if self.llm else 'none (heuristic)'} · fees {self.fee_bps} bps/side",
                 "", "The villagers only saw the **train** period. The **test** (vault) period is "
                 "later data nobody optimised on: it is the honest check.", "",
                 "| # | Strategy | Author | Train Sharpe | Train return | Test Sharpe | Test return "
                 "| Test trades | Test buy&hold | Verdict |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        table = []
        for i, r in enumerate(rows, 1):
            tr, te = r["train"], r["test"]
            verdict = _verdict(tr, te)
            lines.append(f"| {i} | {r['name']} | {r['author']} | {tr['sharpe']} | "
                         f"{tr['total_return_pct']}% | {te['sharpe']} | {te['total_return_pct']}% | "
                         f"{te['trades']} | {te['buy_hold_pct']}% | {verdict} |")
            table.append(f"{i}. '{r['name']}' ({strat.describe(r['spec'])})\n"
                         f"   train: {tr}\n   test: {te}\n   verdict: {verdict}")
        if not rows:
            lines.append("| - | no valid strategies yet | | | | | | | | |")
        story = self.mayor.summarise("\n".join(table)) if rows else None
        if story:
            lines += ["", "## The Mayor's summary", "", story.strip()]
        held = [r for r in rows if _verdict(r["train"], r["test"]) == "held up"]
        positive = [r for r in rows if r["train"]["sharpe"] > 0]
        best = held[0] if held else (positive[0] if positive else None)
        if best:
            lines += ["", f"## Best candidate: {best['name']}", "",
                      "```json", json.dumps(best["spec"], indent=2), "```"]
        lines += ["", "_Backtests are not promises. Paper-trade anything before risking money._"]

        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if best:
            (out / "best_strategy.json").write_text(json.dumps(best["spec"], indent=2), "utf-8")
        self.log(f"\nReport written to {path}")
        return path


def _verdict(train: dict, test: dict) -> str:
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
