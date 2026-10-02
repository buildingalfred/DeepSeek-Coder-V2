"""The villagers.

Librarian - reads PDFs and notes, posts trading ideas to the board.
Quant     - proposes strategies (one per round) in its own style, learning from the board.
Critic    - reviews each round and posts lessons for the quants.
Mayor     - writes the final report and opens the vault (the unseen test period).

Every agent works with or without an LLM. Without one they use simple heuristics, which turns the
village into a random search: a useful baseline the LLM agents should beat.
"""

import copy
import json
import random
import re

from . import indicators, pdfs
from . import strategy as strat
from .llm import LLMError

SPEC_GUIDE = f"""A strategy is a JSON object:
{{
  "name": "short name",
  "idea": "one sentence: why this should work",
  "indicators": [{{"id": "fast", "type": "ema", "period": 12}}, ...],
  "entry_long":  [condition, ...],   // all must be true to go long
  "exit_long":   [condition, ...],   // all must be true to close a long
  "entry_short": [condition, ...],   // optional, [] for long-only
  "exit_short":  [condition, ...],
  "stop_loss_pct": number or null,   // e.g. 5 means exit after a 5% loss
  "take_profit_pct": number or null
}}
A condition is {{"left": X, "op": OP, "right": Y}} where OP is one of {strat.OPS}.
X is a price column ({", ".join(strat.PRICE_COLUMNS)}), an indicator id, or "<id>.<output>" for
indicators with several outputs. Y is the same kind of name, or a number.
Indicators available:
{indicators.catalog_text()}
Signals are checked at each bar's close and filled at the next bar's open. Fees are charged."""


def _fmt_metrics(m: dict) -> str:
    if not m:
        return "n/a"
    return (f"sharpe {m['sharpe']}, return {m['total_return_pct']}%, maxDD {m['max_drawdown_pct']}%, "
            f"trades {m['trades']}, win {m['win_rate_pct']}%, PF {m['profit_factor']}, "
            f"buy&hold {m['buy_hold_pct']}%")


def leaderboard_text(rows: list[dict]) -> str:
    if not rows:
        return "(empty - nobody has scored yet)"
    out = []
    for i, r in enumerate(rows, 1):
        out.append(f"{i}. #{r['id']} '{r['name']}' by {r['author']}: {_fmt_metrics(r['train'])}\n"
                   f"   rules: {strat.describe(r['spec'])}")
    return "\n".join(out)


class Agent:
    role = "villager"

    def __init__(self, name: str, llm=None, log=print):
        self.name = name
        self.llm = llm
        self.log = log

    def ask(self, system: str, user: str, json_mode: bool = False) -> str | None:
        """Ask the LLM; on failure log it and return None so the caller can fall back."""
        if self.llm is None:
            return None
        try:
            return self.llm.chat(system, user, json_mode=json_mode)
        except LLMError as e:
            self.log(f"  [{self.name}] LLM problem: {e}")
            return None


class Librarian(Agent):
    role = "librarian"
    SYSTEM = ("You are the Librarian of a village of trading researchers. You read papers, books and "
              "notes and extract concrete, testable trading ideas. Only report ideas that could be "
              "expressed with standard technical indicators on OHLCV price data. Be concise.")

    def study(self, folder, board, round_: int, max_chunks: int = 6) -> int:
        """Read every document not read before and post its ideas. Returns number of new ideas."""
        found = 0
        for path in pdfs.find_documents(folder):
            source = str(path.name)
            if board.has_source(source):
                continue
            try:
                text = pdfs.read_document(path)
            except Exception as e:  # broken PDFs should not stop the village
                self.log(f"  [{self.name}] could not read {source}: {e}")
                continue
            self.log(f"  [{self.name}] reading {source} ({len(text):,} chars)")
            ideas = []
            for piece in pdfs.chunks(text)[:max_chunks]:
                ideas += self._ideas_from(piece, source)
            if not ideas:
                ideas = self._keyword_ideas(text, source)
            for idea in ideas:
                board.post(round_, self.name, "idea", f"[{source}] {idea}")
            board.post(round_, self.name, "source", source)
            found += len(ideas)
        return found

    def _ideas_from(self, piece: str, source: str) -> list[str]:
        reply = self.ask(self.SYSTEM,
                         f"Text from '{source}':\n\n{piece}\n\nReturn JSON: "
                         '{"ideas": ["idea with concrete rule and parameters", ...]} '
                         "(at most 5, empty list if none).", json_mode=True)
        if not reply:
            return []
        try:
            data = strat.extract_json(reply)
        except strat.SpecError:
            return []
        items = data.get("ideas", []) if isinstance(data, dict) else data
        return [str(i)[:400] for i in items if i][:5] if isinstance(items, list) else []

    @staticmethod
    def _keyword_ideas(text: str, source: str) -> list[str]:
        lower = text.lower()
        hits = {
            "moving average": "moving-average trend following (crossovers)",
            "momentum": "momentum / rate of change",
            "rsi": "RSI overbought/oversold",
            "relative strength": "RSI overbought/oversold",
            "macd": "MACD signal-line crosses",
            "bollinger": "Bollinger band breakouts or reversion",
            "mean reversion": "mean reversion to an average",
            "breakout": "price breakouts",
            "volatility": "volatility filters (ATR)",
            "stochastic": "stochastic oscillator",
        }
        found = sorted({v for k, v in hits.items() if k in lower})
        return [f"mentions {', '.join(found)}"] if found else []


STYLES = {
    "trend": "trend follower: you believe prices that move keep moving. Crossovers, MACD, momentum.",
    "reversion": "mean-reversion trader: you fade extremes. RSI, z-score, Bollinger bands.",
    "breakout": "breakout and volatility trader: you buy strength escaping a range, with stops.",
}


class Quant(Agent):
    role = "quant"

    def __init__(self, name, style, llm=None, log=print, rng=None):
        super().__init__(name, llm, log)
        self.style = style
        self.rng = rng or random.Random()

    def system_prompt(self) -> str:
        return (f"You are {self.name}, a quant in a village of trading researchers. You are a "
                f"{STYLES[self.style]} You work with others: build on good ideas on the board, "
                "learn from the critic, avoid repeating failures, and keep strategies simple "
                "(few rules generalise better; 20+ trades are needed to trust a result).\n\n"
                + SPEC_GUIDE + "\n\nReply with ONLY the JSON strategy object.")

    def propose(self, ctx: dict, error: str | None = None, previous: str | None = None) -> dict:
        prompt = (f"Market: {ctx['data_summary']}\n\n"
                  f"Ideas from the library:\n{ctx['ideas'] or '(none)'}\n\n"
                  f"Leaderboard (train period):\n{ctx['leaderboard']}\n\n"
                  f"Critic's latest notes:\n{ctx['critique'] or '(none yet)'}\n\n"
                  f"Your recent attempts:\n{ctx['mine'] or '(none)'}\n\n"
                  "Propose ONE new strategy that you think will beat the leaderboard.")
        if error:
            prompt += (f"\n\nYour previous reply was rejected by the backtester:\n{error}\n"
                       f"Previous reply:\n{previous}\nFix it and reply with the corrected JSON.")
        reply = self.ask(self.system_prompt(), prompt, json_mode=True)
        if reply is None:
            return self.heuristic(ctx.get("best_specs", []))
        try:
            spec = strat.parse(reply)
        except strat.SpecError as e:
            spec = {"_raw": reply, "_parse_error": str(e)}
        return spec

    # ---- heuristics used when there is no LLM -----------------------------------------
    def heuristic(self, best_specs: list[dict]) -> dict:
        if best_specs and self.rng.random() < 0.5:
            spec = mutate(self.rng.choice(best_specs[:3]), self.rng)
            base = re.sub(r" \(tweaked by .*\)$", "", spec.get("name", "leader"))
            spec["name"] = f"{base} (tweaked by {self.name.split()[0]})"
            return spec
        return random_spec(self.style, self.rng, self.name)


class Critic(Agent):
    role = "critic"
    SYSTEM = ("You are the Critic of a village of trading researchers. You review backtests "
              "skeptically: overfitting, too few trades, drawdowns, not beating buy & hold, rules "
              "that only work by luck. Give short, concrete advice the quants can act on next round.")

    def review(self, round_rows: list[dict], leaderboard: str) -> str:
        lines = []
        for r in round_rows:
            if r["error"]:
                lines.append(f"- {r['author']}: INVALID ({r['error'][:150]})")
            else:
                lines.append(f"- {r['author']} '{r['name']}': {_fmt_metrics(r['train'])}\n"
                             f"  rules: {strat.describe(r['spec'])}")
        results = "\n".join(lines)
        reply = self.ask(self.SYSTEM,
                         f"This round's results (train period):\n{results}\n\nLeaderboard:\n"
                         f"{leaderboard}\n\nWrite at most 6 bullet points of advice.")
        return reply.strip() if reply else self.heuristic(round_rows)

    @staticmethod
    def heuristic(round_rows: list[dict]) -> str:
        notes = []
        for r in round_rows:
            m = r["train"]
            if r["error"]:
                notes.append(f"- {r['author']}: spec was invalid, keep rules to known columns.")
            elif m["trades"] < 20:
                notes.append(f"- {r['author']}: only {m['trades']} trades, too few to trust.")
            elif m["max_drawdown_pct"] < -35:
                notes.append(f"- {r['author']}: drawdown {m['max_drawdown_pct']}% is too deep, add a stop.")
            elif m["total_return_pct"] < m["buy_hold_pct"]:
                notes.append(f"- {r['author']}: lost to buy & hold ({m['total_return_pct']}% vs "
                             f"{m['buy_hold_pct']}%).")
            elif m["sharpe"] > 1:
                notes.append(f"- {r['author']}: promising (sharpe {m['sharpe']}); others try variations.")
        return "\n".join(notes) or "- Nothing stood out this round."


class Mayor(Agent):
    role = "mayor"
    SYSTEM = ("You are the Mayor of a village of trading researchers. Summarise what the village "
              "found for the person who owns it, in plain language. Be honest: a strategy that "
              "fails on the unseen test period is probably overfit, and past results do not "
              "guarantee future returns.")

    def summarise(self, table: str) -> str | None:
        return self.ask(self.SYSTEM,
                        f"Top strategies. 'train' is what the villagers optimised on; 'test' is the "
                        f"sealed later period nobody saw:\n\n{table}\n\nWrite a short report (under "
                        "300 words): what worked, what is likely overfit, what to try next.")


# ---- random strategy generation and mutation (the no-LLM villagers) ---------------------

def random_spec(style: str, rng: random.Random, author: str = "") -> dict:
    stop = rng.choice([None, 3, 5, 8, 12])
    if style == "trend":
        kind = rng.choice(["ema", "sma"])
        fast = rng.randint(5, 30)
        slow = fast + rng.randint(10, 120)
        spec = {
            "name": f"{kind.upper()} {fast}/{slow} cross",
            "idea": "follow the trend when the fast average leads the slow one",
            "indicators": [{"id": "fast", "type": kind, "period": fast},
                           {"id": "slow", "type": kind, "period": slow}],
            "entry_long": [{"left": "fast", "op": "crosses_above", "right": "slow"}],
            "exit_long": [{"left": "fast", "op": "crosses_below", "right": "slow"}],
            "entry_short": [], "exit_short": [],
        }
        if rng.random() < 0.4:
            spec["indicators"].append({"id": "rsi", "type": "rsi", "period": 14})
            spec["entry_long"].append({"left": "rsi", "op": "<", "right": rng.choice([65, 70, 75])})
        if rng.random() < 0.3:
            spec["entry_short"] = [{"left": "fast", "op": "crosses_below", "right": "slow"}]
            spec["exit_short"] = [{"left": "fast", "op": "crosses_above", "right": "slow"}]
    elif style == "reversion":
        if rng.random() < 0.5:
            p, lo, hi = rng.choice([2, 3, 5, 7, 14]), rng.randint(10, 35), rng.randint(50, 80)
            spec = {
                "name": f"RSI{p} {lo}/{hi} reversion",
                "idea": "buy short-term oversold dips, sell the bounce",
                "indicators": [{"id": "rsi", "type": "rsi", "period": p}],
                "entry_long": [{"left": "rsi", "op": "<", "right": lo}],
                "exit_long": [{"left": "rsi", "op": ">", "right": hi}],
            }
        else:
            p, z = rng.randint(10, 60), round(rng.uniform(1.0, 2.5), 2)
            spec = {
                "name": f"z-score {p} at -{z}",
                "idea": "fade stretched moves away from the average",
                "indicators": [{"id": "z", "type": "zscore", "period": p}],
                "entry_long": [{"left": "z", "op": "<", "right": -z}],
                "exit_long": [{"left": "z", "op": ">", "right": 0}],
            }
        if rng.random() < 0.4:
            spec["indicators"].append({"id": "trend", "type": "sma", "period": rng.choice([100, 150, 200])})
            spec["entry_long"].append({"left": "close", "op": ">", "right": "trend"})
    else:
        p, sd = rng.randint(10, 50), rng.choice([1.5, 2.0, 2.5])
        spec = {
            "name": f"Bollinger {p}/{sd} breakout",
            "idea": "buy when price escapes the upper band, exit back at the middle",
            "indicators": [{"id": "bb", "type": "bbands", "period": p, "std": sd}],
            "entry_long": [{"left": "close", "op": "crosses_above", "right": "bb.upper"}],
            "exit_long": [{"left": "close", "op": "crosses_below", "right": "bb.mid"}],
        }
        if rng.random() < 0.4:
            spec["indicators"].append({"id": "mom", "type": "roc", "period": rng.randint(5, 30)})
            spec["entry_long"].append({"left": "mom", "op": ">", "right": 0})
        stop = stop or rng.choice([4, 6, 8])
    spec.setdefault("entry_short", [])
    spec.setdefault("exit_short", [])
    spec["stop_loss_pct"] = stop
    spec["take_profit_pct"] = None
    return spec


def mutate(spec: dict, rng: random.Random) -> dict:
    """Jitter the numbers in a spec by up to about 25%."""
    spec = copy.deepcopy(spec)
    for ind in spec.get("indicators", []):
        for k, v in list(ind.items()):
            if k in ("id", "type", "source") or not isinstance(v, (int, float)):
                continue
            if isinstance(v, int):
                ind[k] = max(2, int(round(v * rng.uniform(0.75, 1.25))))
            else:
                ind[k] = round(max(0.1, v * rng.uniform(0.8, 1.2)), 2)
    for key in strat.RULE_KEYS:
        for cond in spec.get(key) or []:
            if isinstance(cond.get("right"), (int, float)) and cond["right"] != 0:
                cond["right"] = round(cond["right"] * rng.uniform(0.85, 1.15), 2)
    if spec.get("stop_loss_pct") and rng.random() < 0.5:
        spec["stop_loss_pct"] = round(spec["stop_loss_pct"] * rng.uniform(0.7, 1.3), 1)
    return spec


def compact(spec: dict) -> str:
    return re.sub(r"\s+", " ", json.dumps(spec))
