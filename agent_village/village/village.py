"""The village loop: read, propose, backtest, critique, tune, repeat. Then open the vault."""

import json
import random
import re
import time
from pathlib import Path

from . import backtest, data, indicators, pine, report
from . import strategy as strat
from .agents import (Analyst, Critic, Inventor, Librarian, Mayor, Quant, Tuner, base_name,
                     compact, leaderboard_text)
from .board import Board

MAX_ATTEMPTS_LLM = 3        # an LLM quant gets this many tries to produce a valid, new strategy
MAX_ATTEMPTS_HEURISTIC = 8  # random villagers are cheap, let them retry more on duplicates


def behaviour(evaluation: dict) -> str:
    """Two strategies with the same trades and returns are the same strategy, whatever the rules."""
    t, v = evaluation["train"], evaluation["test"]
    return json.dumps([t.get("trades"), t.get("total_return_pct"), t.get("sharpe"),
                       v.get("trades"), v.get("total_return_pct")])


def short_id(spec: dict) -> str:
    import hashlib
    return hashlib.sha1(fingerprint(spec).encode()).hexdigest()[:4]


def fingerprint(spec: dict) -> str:
    """Identity of a strategy's rules, ignoring its name and description."""
    core = {k: v for k, v in spec.items()
            if k not in ("name", "idea", "whiteboard") and not k.startswith("_")}
    for key in strat.RULE_KEYS:
        core.setdefault(key, [])
    core.setdefault("stop_loss_pct", None)
    core.setdefault("take_profit_pct", None)
    return json.dumps(core, sort_keys=True)


class Village:
    def __init__(self, markets, dataset: str, board: Board, llm=None, quants=None, fee_bps=5.0,
                 train_frac=0.7, seed=None, tuner=True, mission: str = "", log=print):
        self.markets = markets if isinstance(markets, dict) else {dataset: markets}
        self.dataset = dataset
        self.board = board
        self.llm = llm
        self.fee_bps = fee_bps
        self.train_frac = train_frac
        self.round = board.last_round()
        self._print = log
        log = self.log  # every villager reports through the village, so the live view sees it
        self.mission = mission
        rng = random.Random(seed)
        intraday = any(data.bar_size(df) not in ("1d", "1w", "unknown") for df in self.markets.values())
        self.librarian = Librarian("Lena the Librarian", llm, log, mission)
        self.analyst = Analyst("Ada the Analyst", None, log)
        self.inventor = Inventor("Nova the Inventor", llm, log, random.Random(rng.random()),
                                 intraday, mission)
        self.critic = Critic("Carl the Critic", llm, log)
        self.mayor = Mayor("Maya the Mayor", llm, log)
        self.tuner = Tuner("Tess the Tuner", None, log, random.Random(rng.random())) if tuner else None
        styles = quants or ["trend", "reversion", "breakout"]
        names = {"trend": "Tom the trend quant", "reversion": "Rita the reversion quant",
                 "breakout": "Bo the breakout quant", "ict_liquidity": "Ivy the liquidity hunter",
                 "ict_blocks": "Ian the order-block hunter", "ict_time": "Iris the time hunter"}
        self.quants = []
        for i, s in enumerate(styles):
            name = names.get(s, f"{s.title()} the quant")
            if styles[:i].count(s):
                name += f" {styles[:i].count(s) + 1}"
            self.quants.append(Quant(name, s, llm, log, random.Random(rng.random()), mission,
                                     intraday))
        # Agents only ever see the train period; the rest stays sealed in the vault.
        self.data_summary = "\n".join(
            data.summary(df.iloc[: int(len(df) * train_frac)], f"{name} (train period only)")
            for name, df in self.markets.items())
        self.caches = {name: indicators.Cache() for name in self.markets}
        rows = board.all_strategies(dataset)
        self.seen = {fingerprint(r["spec"]) for r in rows if r["spec"]}
        self.behaviours = {behaviour(r) for r in rows if r["train"] and r["test"]}

    def log(self, text: str) -> None:
        """Print a line, and record '[Name] did something' lines as events for the live view."""
        self._print(text)
        m = re.match(r"\s*\[([^\]]+)\]\s*(.*)", text, re.S)
        if m:
            self.board.event(self.dataset, self.round, m.group(1), m.group(2).strip()[:600])

    def evaluate(self, spec: dict) -> dict:
        return backtest.evaluate(self.markets, spec, self.fee_bps, self.train_frac, self.caches)

    def score(self, evaluation: dict) -> float:
        return backtest.score(evaluation, len(self.markets))

    def run(self, rounds: int = 5, papers: str | None = None, report_dir: str = "reports") -> Path:
        start_round = self.board.last_round()
        brain = self.llm.name if self.llm else "no LLM (heuristic villagers)"
        self.log(f"Village waking up. Brain: {brain}.\nData: {self.data_summary}")
        for r in range(start_round + 1, start_round + rounds + 1):
            self.round = r
            self.log(f"\n=== Round {r} ===")
            self.board.event(self.dataset, r, "village", f"Round {r} begins")
            if papers:
                n = self.librarian.study(papers, self.board, r)
                if n:
                    self.log(f"  [{self.librarian.name}] posted {n} new ideas")
            for quant in self.quants:
                self._turn(quant, r)
            if self.tuner:
                self._tune(r)
            self._invent(r)
            analysis = self._analyse(r)
            critique = self.critic.review(self.board.in_round(self.dataset, r),
                                          leaderboard_text(self.board.leaderboard(self.dataset, 5))
                                          + (f"\n\nAnalyst:\n{analysis}" if analysis else ""))
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
        analyses = self.board.notes("analysis", 1, self.dataset)
        return {
            "whiteboard": self.whiteboard_text(14),
            "data_summary": self.data_summary,
            "ideas": "\n".join(f"- {n['content']}" for n in self.board.notes("idea", 15)),
            "leaderboard": leaderboard_text(top),
            "critique": critiques[-1]["content"] if critiques else "",
            "analysis": analyses[-1]["content"] if analyses else "",
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
        note = spec.get("whiteboard")
        if isinstance(note, str) and note.strip():
            self.write(round_, quant.name, note.strip()[:300])
        clean = {k: v for k, v in spec.items() if not k.startswith("_") and k != "whiteboard"}
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
            evaluation = self.evaluate(spec)
        except strat.SpecError as e:
            return None, str(e)
        if not evaluation["train"].get("trades"):
            return None, ("this strategy never trades on the train data; its conditions are "
                          "never all true together. Loosen them or use 'within' to chain steps")
        if behaviour(evaluation) in self.behaviours:
            return None, ("this trades exactly like a strategy already tested (same trades and "
                          "returns), so the extra rules change nothing; try a real difference")
        return evaluation, None

    def write(self, round_: int, author: str, text: str) -> None:
        """Put a note on the shared whiteboard."""
        self.board.post(round_, author, "whiteboard", text, self.dataset)

    def whiteboard_text(self, n: int = 14) -> str:
        notes = self.board.notes("whiteboard", n, self.dataset)
        return "\n".join(f"- [r{x['round']} {x['author'].split(' ')[0]}] {x['content']}" for x in notes)

    def _record(self, round_: int, author: str, spec: dict, evaluation: dict) -> int:
        score = self.score(evaluation)
        top = self.board.leaderboard(self.dataset, 1)
        if score > -99 and (not top or score > top[0]["score"] + 0.05):
            t = evaluation["train"]
            self.write(round_, author, f"AHA: new leader '{spec.get('name', 'unnamed')}' "
                                       f"(robust {t['robust_sharpe']}, t {t['t_stat']}, "
                                       f"{t['trades']} trades). Rules: {strat.describe(spec)}")
        sid = self.board.add_strategy(round_, author, spec, evaluation["train"], evaluation["test"],
                                      score, None, self.dataset, evaluation["markets"])
        self.seen.add(fingerprint(spec))
        self.behaviours.add(behaviour(evaluation))
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

    def _analyse(self, round_: int) -> str:
        """Ada takes the current leader apart and posts which conditions carry the edge."""
        top = self.board.leaderboard(self.dataset, 1)
        if not top or top[0]["score"] <= -99:
            return ""
        leader = top[0]
        last = self.board.notes("analysis", 1, self.dataset)
        tag = f"#{leader['id']}"
        if last and last[-1]["content"].startswith(tag):
            return last[-1]["content"]  # same leader as last round, nothing new to learn
        rows = self.analyst.ablate(leader["spec"], self._ablation_score)
        text = f"{tag} " + self.analyst.summarise(leader["name"], leader["score"], rows)
        self.board.post(round_, self.analyst.name, "analysis", text, self.dataset)
        self.board.post(round_, self.analyst.name, "trials", str(len(rows)), self.dataset)
        self.log(f"  [{self.analyst.name}]\n" + _indent(text))
        essential = [r for r in rows if r["impact"] > 0.2][:2]
        for row in rows:
            if row in essential:
                self.write(round_, self.analyst.name,
                           f"AHA: '{row['condition']}' is ESSENTIAL in the leader (without it "
                           f"{row['without']} vs {leader['score']}). Keep it, build around it.")
            elif row["impact"] < -0.05:
                self.write(round_, self.analyst.name,
                           f"Dead end: '{row['condition']}' HURTS the leader; drop it.")
        # If dropping a piece clearly helps, submit the simpler version for the team to build on.
        worst = rows[-1] if rows else None
        if worst and worst["impact"] < -0.05:
            spec = dict(worst["variant"], name=f"{base_name(leader['name'])} (simplified)")
            evaluation, error = self._try(spec)
            if not error:
                self._record(round_, self.analyst.name, spec, evaluation)
        return text

    def _invent(self, round_: int) -> None:
        """Nova writes hypotheses on the whiteboard and tests up to two inventions per round:
        with an AI, her best hypothesis as a strategy; always, a graft or a crossbreed."""
        top = [r for r in self.board.leaderboard(self.dataset, 3) if r["score"] > -99]
        ctx = self._context(self.quants[0])
        hypotheses = self.inventor.brainstorm(ctx)
        for h in hypotheses:
            self.write(round_, self.inventor.name, f"Hypothesis: {h}")
        candidates = []
        if hypotheses:
            spec = self.inventor.design(ctx, hypotheses[0], top[0]["spec"] if top else None)
            if spec:
                candidates.append((spec, f"Testing my hypothesis: {hypotheses[0][:160]}"))
        rng = self.inventor.rng
        for _ in range(6 if top else 0):
            if len(top) >= 2 and rng.random() < 0.4:
                a, b = rng.sample(top, 2)
                candidates.append(self.inventor.crossbreed(a["spec"], b["spec"]))
            else:
                candidates.append(self.inventor.graft(rng.choice(top)["spec"]))
        done = 0
        for spec, idea in candidates:
            evaluation, error = self._try(spec)
            if error:
                continue
            clean = {k: v for k, v in spec.items() if not k.startswith("_") and k != "whiteboard"}
            sid = self._record(round_, self.inventor.name, clean, evaluation)
            t = evaluation["train"]
            self.write(round_, self.inventor.name,
                       f"{idea} Tested as #{sid}: robust {t['robust_sharpe']}, {t['trades']} trades.")
            done += 1
            if done == 2:
                return

    def seed(self, specs: list[dict], author: str) -> int:
        """Bring strategies found elsewhere (e.g. a smaller window) onto this board."""
        round_ = self.board.last_round()
        added = 0
        for spec in specs:
            evaluation, error = self._try(spec)
            if not error:
                self._record(round_, author, spec, evaluation)
                added += 1
        return added

    def _ablation_score(self, spec: dict):
        try:
            return self.score(self.evaluate(spec))
        except strat.SpecError:
            return None

    def _tuner_eval(self, spec: dict):
        evaluation, error = self._try(spec)
        if error:
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
        analyses = self.board.notes("analysis", 1, self.dataset)
        analysis = analyses[-1]["content"] if analyses else ""
        if analysis:
            table += f"\n\nAnalyst's breakdown of the leader:\n{analysis}"
        story = self.mayor.summarise(table, self.mission) if rows else None
        # Only a strategy that survived the vault can be the best candidate; never fall back.
        held = [r for r in rows if r["verdict"] == "held up"]
        maybe = [r for r in rows if r["verdict"].startswith("held up")]
        best = held[0] if held else (maybe[0] if maybe else None)
        brain = self.llm.name if self.llm else "none (heuristic)"

        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        board_text = self.whiteboard_text(20)
        ladders = self.board.notes("ladder", 1, self.dataset)
        if ladders:
            analysis = (analysis + "\n\n" if analysis else "") + \
                "Scout-and-expand ladder (robust / unseen sharpe per window):\n" + ladders[-1]["content"]
        md = report.markdown(self.dataset, rows, best, story, trials, brain, self.fee_bps, analysis,
                             board_text)
        path = out / f"report-{stamp}.md"
        path.write_text(md, encoding="utf-8")
        charts = [(r, self._curves(r["spec"])) for r in (rows[:3] if rows else [])]
        html_path = out / f"report-{stamp}.html"
        html_path.write_text(report.html(self.dataset, rows, best, story, trials, brain,
                                         self.fee_bps, charts, self.train_frac, analysis,
                                         board_text),
                             encoding="utf-8")
        if best:
            (out / "best_strategy.json").write_text(json.dumps(best["spec"], indent=2), "utf-8")
            code = pine.to_pine(best["spec"], self.fee_bps, {"train": best["train"],
                                                            "vault": best["test"]})
            (out / "best_strategy.pine").write_text(code, "utf-8")
            self.log(f"Pine Script for TradingView: {out / 'best_strategy.pine'}")
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
    if test.get("t_stat", 99) < 2:
        return "held up, but could be luck"
    return "held up"


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())
