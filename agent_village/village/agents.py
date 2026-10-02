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

from . import features, ict, indicators, pdfs
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
A condition is {{"left": X, "op": OP, "right": Y}} or {{"left": X, "op": OP, "right": Y, "within": N}}
where OP is one of {strat.OPS}.
X is a price column ({", ".join(strat.PRICE_COLUMNS)}), an indicator id, or "<id>.<output>" for
indicators with several outputs. Y is the same kind of name, or a number. "within": N makes the
condition true if it held on any of the last N bars, so conditions can form a sequence.
Indicators available:
{indicators.catalog_text()}

{ict.GUIDE}

{features.GUIDE}

Optionally add "whiteboard": "one short note for the team" (an observation, a hunch, an "aha").

Signals are checked at each bar's close and filled at the next bar's open. Fees are charged."""


def _fmt_metrics(m: dict) -> str:
    if not m:
        return "n/a"
    robust = f"robust {m['robust_sharpe']} (good periods {m['positive_periods']}), " \
        if "robust_sharpe" in m else ""
    return (f"{robust}sharpe {m['sharpe']}, return {m['total_return_pct']}%, maxDD {m['max_drawdown_pct']}%, "
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
    SYSTEM = ("You are the Librarian of a village of trading researchers. You read papers, bank and "
              "central-bank research, books and notes, and extract concrete, testable rules about "
              "how price moves: exact conditions, sequences of events, times of day (say which "
              "time zone), price levels, thresholds and parameters. Look closely for details that "
              "are easy to miss: footnotes, tables, timing of fixes and auctions, liquidity and "
              "order-flow behaviour, stop clusters, session opens. Say which building block each "
              "idea maps to (e.g. liquidity sweep, fair value gap, order block, market structure "
              "shift, kill zone, previous-day high/low, moving average, RSI). Only report ideas "
              "that can be tested on OHLCV price data. Be concise and precise.")

    def __init__(self, name, llm=None, log=print, mission: str = ""):
        super().__init__(name, llm, log)
        self.mission = mission

    PINE = ("This document is TradingView Pine Script source code. Read it like an engineer: "
            "state exactly what it computes and when it signals (conditions, lookbacks, "
            "thresholds, sessions, time zones), then say how each part maps onto the village's "
            "building blocks or a formula. Note anything clever or unusual it does.")

    def study(self, folder, board, round_: int, max_chunks: int = 12) -> int:
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
            pine = pdfs.is_pine(path, text)
            for piece in pdfs.chunks(text)[:max_chunks]:
                ideas += self._ideas_from(piece, source, pine)
            if not ideas:
                ideas = self._keyword_ideas(text, source)
            for idea in ideas:
                board.post(round_, self.name, "idea", f"[{source}] {idea}")
            board.post(round_, self.name, "source", source)
            found += len(ideas)
        return found

    def _ideas_from(self, piece: str, source: str, pine: bool = False) -> list[str]:
        system = self.SYSTEM + (f"\n\nThe village's mission: {self.mission}" if self.mission else "")
        if pine:
            system += "\n\n" + self.PINE
        reply = self.ask(system,
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
            "fair value gap": "fair value gaps (imbalances)",
            "imbalance": "fair value gaps (imbalances)",
            "order block": "order blocks",
            "liquidity": "liquidity pools and sweeps",
            "stop hunt": "liquidity sweeps (stop hunts)",
            "kill zone": "ICT kill zones (session timing)",
            "killzone": "ICT kill zones (session timing)",
            "london open": "session timing (London open)",
            "market structure": "market structure shifts",
            "displacement": "displacement candles",
            "premium": "premium/discount of the dealing range",
            "optimal trade entry": "OTE retracement (62-79%)",
            "silver bullet": "silver bullet window (10-11 New York)",
            "fix": "timing around benchmark fixes",
        }
        found = sorted({v for k, v in hits.items() if k in lower})
        return [f"mentions {', '.join(found)}"] if found else []


STYLES = {
    "trend": "trend follower: you believe prices that move keep moving. Crossovers, MACD, momentum.",
    "reversion": "mean-reversion trader: you fade extremes. RSI, z-score, Bollinger bands.",
    "breakout": "breakout and volatility trader: you buy strength escaping a range, with stops.",
    "ict_liquidity": ("ICT liquidity specialist: price seeks liquidity. You chain sequences: a "
                      "sweep of a swing low/high, then a market structure shift, then entry on the "
                      "retrace into the fair value gap left by the displacement."),
    "ict_blocks": ("ICT order-block specialist: you buy bullish order blocks in discount and sell "
                   "bearish ones in premium, using the dealing range and the OTE retracement."),
    "ict_time": ("ICT time-and-price specialist: you believe the algorithm runs on a clock. You "
                 "use kill zones (London, New York AM, silver bullet) and raids on the previous "
                 "day's high/low."),
}
TEAMS = {
    "default": ["trend", "reversion", "breakout"],
    "ict": ["ict_liquidity", "ict_blocks", "ict_time"],
    "mixed": ["ict_liquidity", "ict_blocks", "ict_time", "trend", "reversion"],
}
ALGO_MISSION = (
    "Treat price as the output of a delivery algorithm, and reverse-engineer its rules. Think "
    "like someone decoding a machine, not like a textbook trader: standard indicator strategies "
    "assume a normal, random market and are not what we are looking for. Hunt for rules about "
    "TIME (session opens, the midnight open, macro windows like xx:50-xx:10, quarter-hour cycles, "
    "minutes between events), NUMBERS (round levels, counts of swings or gaps, digital roots), "
    "SEQUENCE (raid of liquidity -> shift -> gap -> delivery to the next pool) and SYMMETRY (equal "
    "times or ranges). Every idea must be a precise, falsifiable rule; the vault decides what is "
    "real. Write your 'aha' moments and hunches on the whiteboard so the others can build on them.")
ICT_MISSION = ("Find out whether ICT's model of price delivery holds up: test liquidity sweeps, "
               "market structure shifts, fair value gaps, order blocks, premium/discount and "
               "kill zones, and above all how they connect into one sequence. Work as a team: "
               "build on whatever piece the analyst shows is actually adding value, and drop "
               "pieces that add nothing.")


class Quant(Agent):
    role = "quant"

    def __init__(self, name, style, llm=None, log=print, rng=None, mission: str = "",
                 intraday: bool = False):
        super().__init__(name, llm, log)
        self.style = style
        self.rng = rng or random.Random()
        self.mission = mission
        self.intraday = intraday

    def system_prompt(self) -> str:
        mission = f"The village's mission: {self.mission}\n" if self.mission else ""
        return (f"You are {self.name}, a quant in a village of trading researchers. {mission}You are "
                f"a {STYLES[self.style]} You work with others: build on good ideas on the board, "
                "learn from the critic, avoid repeating failures, and keep strategies simple "
                "(few rules generalise better; 20+ trades are needed to trust a result). The "
                "leaderboard ranks by 'robust' Sharpe: the strategy must work in every slice of "
                "history and every market, not just one lucky stretch. Short or noisy data makes big "
                "Sharpe ratios easy to get by luck; the t-stat says how solid a result is.\n\n"
                + SPEC_GUIDE + "\n\nReply with ONLY the JSON strategy object.")

    def propose(self, ctx: dict, error: str | None = None, previous: str | None = None) -> dict:
        prompt = (f"Market: {ctx['data_summary']}\n\n"
                  f"Ideas from the library:\n{ctx['ideas'] or '(none)'}\n\n"
                  f"Leaderboard (train period):\n{ctx['leaderboard']}\n\n"
                  f"Critic's latest notes:\n{ctx['critique'] or '(none yet)'}\n\n"
                  f"Analyst's findings (which pieces of the leaders matter):\n"
                  f"{ctx.get('analysis') or '(none yet)'}\n\n"
                  f"The team whiteboard (hunches, findings, dead ends):\n"
                  f"{ctx.get('whiteboard') or '(empty)'}\n\n"
                  f"Your recent attempts:\n{ctx['mine'] or '(none)'}\n\n"
                  "Propose ONE new strategy that you think will beat the leaderboard. Test a "
                  "hypothesis from the whiteboard, or build on what the analyst found essential. "
                  "Be inventive: you can create your own features with formulas.")
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
            spec["name"] = f"{base_name(spec.get('name', 'leader'))} (tweaked by {self.name.split()[0]})"
            return spec
        return random_spec(self.style, self.rng, self.name, intraday=self.intraday)


# Nova's idea pool for the no-LLM village: (family, formula, op, thresholds). The formulas assume
# price is delivered by an algorithm with rules about TIME, NUMBERS, SEQUENCE and SYMMETRY.
INVENTED = [
    # counting and numerology
    ("number", "digital_root(count({flag}, {n}))", "==", [1, 3, 5, 6, 7, 9]),
    ("number", "count({flag}, {n}) % 3", "==", [0, 1, 2]),
    ("number", "digital_root(round(close))", "==", [1, 3, 6, 9]),
    ("number", "round(close) % {step}", "<=", [1, 2, 5]),
    ("number", "{step} - round(close) % {step}", "<=", [1, 2, 5]),
    ("number", "digital_root(bars_since({flag}))", "==", [3, 6, 9]),
    ("number", "bars_since({flag}) % {fib}", "==", [0]),
    # sequence and memory
    ("sequence", "bars_since({flag})", "<=", [1, 2, 3, 5, 8]),
    ("sequence", "bars_since({flag2}) - bars_since({flag})", ">", [0, 2, 5]),
    ("sequence", "count({flag}, {n})", ">=", [2, 3]),
    ("sequence", "where(bars_since({flag}) < bars_since({flag2}), 1, 0)", "==", [1]),
    # symmetry: the time from A to B equals the time from B to now
    ("symmetry", "abs(bars_since({flag2}) - 2 * bars_since({flag}))", "<=", [0, 1, 2]),
    ("symmetry", "abs((high - low) - prev(high - low, {m}))", "<", [0.5, 1, 2]),
    # range and ratio
    ("range", "(high - low) / mean(high - low, {n})", ">", [1.5, 2.0, 3.0]),
    ("range", "(close - lowest(low, {n})) / (highest(high, {n}) - lowest(low, {n}))", "<", [0.21, 0.38, 0.5]),
    ("range", "(close - lowest(low, {n})) / (highest(high, {n}) - lowest(low, {n}))", ">", [0.5, 0.62, 0.79]),
]
INVENTED_TIME = [
    # the clock: macros, quarter cycles, session opens and session ranges
    ("time", "minute_of_hour() % 15", "<=", [0, 2, 5]),
    ("time", "minute_of_hour() >= 50 or minute_of_hour() <= 10", "==", [1]),
    ("time", "ny_minute() >= {macro_a} and ny_minute() < {macro_b}", "==", [1]),
    ("time", "digital_root(minute_of_hour())", "==", [3, 6, 9]),
    ("time", "digital_root(bars_today())", "==", [1, 3, 7, 9]),
    ("time", "day_of_week()", "==", [0, 1, 2, 3, 4]),
    ("time", "minutes_since({flag})", "<=", [15, 30, 45, 60, 90]),
    ("open", "close > value_at(open, 0)", "==", [0, 1]),
    ("open", "close > value_at(open, 510)", "==", [0, 1]),
    ("open", "close > value_at(open, 570)", "==", [0, 1]),
    ("open", "abs(close - value_at(open, 0)) / mean(high - low, 50)", ">", [1, 2, 3]),
    ("session", "low < session_low(0, 300) and ny_minute() >= 420", "==", [1]),
    ("session", "high > session_high(0, 300) and ny_minute() >= 420", "==", [1]),
    ("session", "low < session_low(120, 300) and ny_minute() >= 570", "==", [1]),
    ("session", "high > session_high(120, 300) and ny_minute() >= 570", "==", [1]),
    ("session", "(close - session_low(0, 570)) / (session_high(0, 570) - session_low(0, 570))", "<", [0.21, 0.38, 0.5]),
    ("session", "(close - session_low(0, 570)) / (session_high(0, 570) - session_low(0, 570))", ">", [0.5, 0.62, 0.79]),
    ("power of 3", "ny_minute() >= 600 and close > value_at(open, 0) and low < session_low(0, 570)", "==", [1]),
]
MACROS = [(470, 490), (530, 550), (590, 610), (650, 670), (710, 730), (790, 820), (915, 945)]
FLAGS = [("sweep", ["bull", "bear"]), ("structure", ["bos_up", "bos_down", "mss_up", "mss_down"]),
         ("fvg", ["bull", "bear"]), ("displacement", ["up", "down"])]


class Inventor(Agent):
    """Nova the Inventor: the village's imagination. With an LLM she writes new hypotheses on the
    whiteboard (with invented features). She also takes the leader and bolts on one invented
    filter, so every round at least one wild idea gets tested."""
    role = "inventor"
    SYSTEM = ("You are Nova, the Inventor in a village of trading researchers. You believe price "
              "is delivered by an algorithm and your job is imagination: invent new, specific, "
              "testable hypotheses about the algorithm's rules - never textbook indicator ideas. "
              "Examples of the spirit: 'the NY AM macro 09:50-10:10 delivers to the London high "
              "when the midnight open was below it', 'after the 3rd structure break of the day "
              "the next FVG holds', 'moves end when the minutes since the sweep has a digital root "
              "of 9', 'equal time from raid to shift and shift to target'. Invent new ones about how price behaves, "
              "especially hidden structure in timing and counting (minutes between events, how "
              "many swings formed before a move, digital roots, which weekday or minute of the "
              "session, sequences of sweeps and shifts). Build on the whiteboard and the "
              "analyst's findings; do not repeat dead ends. Each hypothesis must be expressible "
              "with the village's formula language.\n\n" + features.GUIDE + "\n\n" + ict.GUIDE)

    def __init__(self, name, llm=None, log=print, rng=None, intraday=False, mission=""):
        super().__init__(name, llm, log)
        self.rng = rng or random.Random()
        self.intraday = intraday
        self.mission = mission

    def brainstorm(self, ctx: dict) -> list[str]:
        mission = f"Mission: {self.mission}\n\n" if self.mission else ""
        reply = self.ask(self.SYSTEM,
                         f"{mission}Market: {ctx['data_summary']}\n\nWhiteboard:\n"
                         f"{ctx.get('whiteboard') or '(empty)'}\n\nLeaderboard:\n{ctx['leaderboard']}"
                         f"\n\nAnalyst:\n{ctx.get('analysis') or '(none)'}\n\nLibrary ideas:\n"
                         f"{ctx['ideas'] or '(none)'}\n\nReturn JSON: {{\"hypotheses\": [\"...\"]}} with "
                         "3 hypotheses, each naming the formula to test.", json_mode=True)
        if not reply:
            return []
        try:
            data = strat.extract_json(reply)
        except strat.SpecError:
            return []
        items = data.get("hypotheses", []) if isinstance(data, dict) else data
        return [str(h)[:400] for h in items if h][:3] if isinstance(items, list) else []

    def design(self, ctx: dict, hypothesis: str, base: dict | None) -> dict | None:
        """Turn one hypothesis into a strategy (LLM only), optionally on top of a leader."""
        base_txt = json.dumps(base) if base else "(none: design from scratch)"
        reply = self.ask(self.SYSTEM + "\n\n" + SPEC_GUIDE + "\n\nReply with ONLY the JSON strategy.",
                         f"Hypothesis to test: {hypothesis}\n\nMarket: {ctx['data_summary']}\n\n"
                         f"You may build on this leading strategy:\n{base_txt}\n\nWrite the strategy "
                         "that tests the hypothesis as directly as possible, with invented features "
                         "where needed. Name it after the hypothesis.", json_mode=True)
        if not reply:
            return None
        try:
            return strat.parse(reply)
        except strat.SpecError:
            return None

    def _flag(self, spec: dict) -> str:
        """A random event flag (sweep, structure break, gap, displacement), adding its indicator."""
        kind, outs = self.rng.choice(FLAGS)
        inds = spec.setdefault("indicators", [])
        have = next((i["id"] for i in inds if i.get("type") == kind and set(i) <= {"id", "type", "k"}),
                    None)
        if not have:
            have = f"nv{len(inds)}"
            inds.append({"id": have, "type": kind})
        return f"{have}.{self.rng.choice(outs)}"

    def graft(self, spec: dict) -> tuple[dict, str]:
        """Copy a strategy and add one invented feature filter to its entry. Returns (spec, idea)."""
        spec = copy.deepcopy(spec)
        side = "entry_long" if spec.get("entry_long") else "entry_short"
        pool = INVENTED + (INVENTED_TIME * 2 if self.intraday else [])
        family, formula, op, values = self.rng.choice(pool)
        macro = self.rng.choice(MACROS)
        fill = {"n": self.rng.choice([8, 13, 21, 34, 55]), "m": self.rng.choice([3, 5, 8, 13]),
                "step": self.rng.choice([10, 25, 50, 100]), "fib": self.rng.choice([3, 5, 8, 13]),
                "macro_a": macro[0], "macro_b": macro[1]}
        if "{flag}" in formula:
            fill["flag"] = self._flag(spec)
        if "{flag2}" in formula:
            fill["flag2"] = self._flag(spec)
        expr = formula.format(**fill)
        fid = f"nova{len(spec.get('features') or [])}"
        spec.setdefault("features", []).append({"id": fid, "expr": expr})
        threshold = self.rng.choice(values)
        spec[side] = list(spec.get(side) or []) + [{"left": fid, "op": op, "right": threshold}]
        spec["name"] = f"{base_name(spec.get('name', 'leader'))} + {expr} {op} {threshold}"[:110]
        return spec, f"[{family}] What if '{expr} {op} {threshold}' is one of the algorithm's rules?"

    def crossbreed(self, a: dict, b: dict) -> tuple[dict, str]:
        """Child of two good setups: a's rules plus one of b's entry conditions (with everything
        that condition needs from b, renamed so nothing clashes)."""
        child = copy.deepcopy(a)
        side = "entry_long" if a.get("entry_long") else "entry_short"
        donor = b.get(side) or b.get("entry_long") or b.get("entry_short") or []
        if not donor:
            return self.graft(a)
        ids = [i["id"] for i in b.get("indicators") or []] + [f["id"] for f in b.get("features") or []]
        rename = {i: f"x{i}" for i in ids}

        def ren(name):
            if not isinstance(name, str):
                return name
            head, dot, tail = name.partition(".")
            return rename.get(head, head) + dot + tail

        def ren_expr(expr):
            for old, new in sorted(rename.items(), key=lambda kv: -len(kv[0])):
                expr = re.sub(rf"\b{re.escape(old)}\b(?!\s*\()", new, expr)
            return expr

        taken = {i["id"] for i in child.get("indicators") or []} | \
            {f["id"] for f in child.get("features") or []}
        if taken & set(rename.values()):
            return self.graft(a)
        child.setdefault("indicators", []).extend(dict(i, id=rename[i["id"]]) for i in b.get("indicators") or [])
        child.setdefault("features", []).extend({"id": rename[f["id"]], "expr": ren_expr(f["expr"])}
                                                for f in b.get("features") or [])
        cond = copy.deepcopy(self.rng.choice(donor))
        cond["left"], cond["right"] = ren(cond["left"]), ren(cond["right"])
        child[side] = list(child.get(side) or []) + [cond]
        child["name"] = f"{base_name(a.get('name', 'A'))} x {base_name(b.get('name', 'B'))}"[:110]
        return child, (f"[crossbreed] Child of '{base_name(a.get('name', 'A'))}' and "
                       f"'{base_name(b.get('name', 'B'))}': does {_cond_text(cond)} add to it?")


class Analyst(Agent):
    """Takes the leading strategy apart: removes one condition at a time and measures the damage.
    This shows which pieces of a setup actually carry the edge. No LLM needed."""
    role = "analyst"

    @staticmethod
    def ablate(spec: dict, evaluate) -> list[dict]:
        """evaluate(spec) -> score or None. Returns one row per condition, most important first."""
        base = evaluate(spec)
        if base is None:
            return []
        rows = []
        for key in strat.RULE_KEYS:
            conds = spec.get(key) or []
            for i, cond in enumerate(conds):
                if key.startswith("entry") and len(conds) == 1:
                    continue  # removing the only entry rule leaves nothing to test
                variant = copy.deepcopy(spec)
                variant[key] = conds[:i] + conds[i + 1:]
                score = evaluate(variant)
                if score is None:
                    continue
                rows.append({"rule": key, "condition": _cond_text(cond), "without": score,
                             "impact": round(base - score, 3), "variant": variant})
        return sorted(rows, key=lambda r: -r["impact"])

    @staticmethod
    def summarise(name: str, base: float, rows: list[dict]) -> str:
        if not rows:
            return f"'{name}' has no condition that can be removed for testing."
        lines = [f"Pieces of '{name}' (robust score {base}):"]
        for r in rows:
            if r["impact"] > 0.05:
                verdict = "ESSENTIAL" if r["impact"] > 0.2 else "helps"
            elif r["impact"] < -0.05:
                verdict = "HURTS, drop it"
            else:
                verdict = "adds nothing"
            lines.append(f"- {r['rule']}: {r['condition']} -> without it {r['without']} "
                         f"({verdict})")
        return "\n".join(lines)


def _cond_text(c: dict) -> str:
    return f"{c['left']} {c['op']} {c['right']}" + (f" within {c['within']}" if c.get("within") else "")


class Tuner(Agent):
    """Fine-tunes a leading strategy by trying small variations of its numbers. No LLM needed."""
    role = "tuner"

    def __init__(self, name, llm=None, log=print, rng=None, tries: int = 12):
        super().__init__(name, llm, log)
        self.rng = rng or random.Random()
        self.tries = tries

    def tune(self, spec: dict, parent_score: float, evaluate):
        """evaluate(spec) -> (evaluation, score) or None. Returns ((spec, evaluation) or None, tried)."""
        best, best_score, tried = None, parent_score, 0
        base = base_name(spec.get("name", "strategy"))
        for _ in range(self.tries):
            candidate = mutate(spec, self.rng)
            candidate["name"] = f"{base} (tuned)"
            got = evaluate(candidate)
            if got is None:
                continue
            tried += 1
            evaluation, score = got
            # Demand a clear improvement: tiny gains from tuning are usually noise.
            if score > best_score + 0.05:
                best, best_score = (candidate, evaluation), score
        return best, tried


class Critic(Agent):
    role = "critic"
    SYSTEM = ("You are the Critic of a village of trading researchers. You review backtests "
              "(including any trading theory the village is testing, such as ICT) "
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
            elif m["sharpe"] > 0 and m.get("robust_sharpe", 0) < 0:
                notes.append(f"- {r['author']}: profitable overall but only in "
                             f"{m['positive_periods']} periods; looks like luck, not an edge.")
            elif m["sharpe"] > 1:
                notes.append(f"- {r['author']}: promising (sharpe {m['sharpe']}); others try variations.")
        return "\n".join(notes) or "- Nothing stood out this round."


class Mayor(Agent):
    role = "mayor"
    SYSTEM = ("You are the Mayor of a village of trading researchers. Summarise what the village "
              "found for the person who owns it, in plain language. Be honest: a strategy that "
              "fails on the unseen test period is probably overfit, and past results do not "
              "guarantee future returns.")

    def summarise(self, table: str, mission: str = "") -> str | None:
        system = self.SYSTEM + (f" The village's mission was: {mission} Say plainly what the "
                                "evidence says about it." if mission else "")
        return self.ask(system,
                        f"Top strategies. 'train' is what the villagers optimised on; 'test' is the "
                        f"sealed later period nobody saw:\n\n{table}\n\nWrite a short report (under "
                        "300 words): what worked, what is likely overfit, what to try next.")


# ---- random strategy generation and mutation (the no-LLM villagers) ---------------------

def random_spec(style: str, rng: random.Random, author: str = "", intraday: bool = False) -> dict:
    if style.startswith("ict"):
        return random_ict_spec(style, rng, intraday)
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


def random_ict_spec(style: str, rng: random.Random, intraday: bool = False) -> dict:
    """Random ICT-style sequences, so the village can explore ICT even without an LLM."""
    k = rng.choice([2, 3, 5])
    stop = rng.choice([1, 2, 3, 5]) if intraday else rng.choice([3, 5, 8])
    take = round(stop * rng.choice([1.5, 2, 3]), 1)
    long_side = rng.random() < 0.6
    up, side = ("bull", "long") if long_side else ("bear", "short")
    shift = "mss_up" if long_side else "mss_down"
    inds = [{"id": "sw", "type": "sweep", "k": k}, {"id": "st", "type": "structure", "k": k}]
    if style == "ict_liquidity":
        inds.append({"id": "gap", "type": "fvg", "min_atr": rng.choice([0.0, 0.25, 0.5])})
        entry = [{"left": "sw." + up, "op": "==", "right": 1, "within": rng.choice([10, 20, 40])},
                 {"left": "st." + shift, "op": "==", "right": 1, "within": rng.choice([5, 10, 20])},
                 ({"left": "low", "op": "<=", "right": "gap.bull_top"} if long_side
                  else {"left": "high", "op": ">=", "right": "gap.bear_bot"})]
        name = f"sweep -> MSS -> FVG {side}"
    elif style == "ict_blocks":
        inds += [{"id": "ob", "type": "order_block", "k": k, "lookback": rng.choice([5, 10, 20])},
                 {"id": "pd", "type": "premium_discount", "k": k}]
        entry = ([{"left": "low", "op": "<=", "right": "ob.bull_top"},
                  {"left": "pd.pos", "op": "<", "right": rng.choice([0.4, 0.5])}] if long_side else
                 [{"left": "high", "op": ">=", "right": "ob.bear_bot"},
                  {"left": "pd.pos", "op": ">", "right": rng.choice([0.5, 0.6])}])
        if rng.random() < 0.5:
            entry.append({"left": "st." + shift, "op": "==", "right": 1, "within": rng.choice([10, 30])})
        name = f"order block in {'discount' if long_side else 'premium'} {side}"
    else:
        inds.append({"id": "pdl", "type": "prev_day"})
        entry = [({"left": "low", "op": "<", "right": "pdl.low"} if long_side
                  else {"left": "high", "op": ">", "right": "pdl.high"}),
                 {"left": "st." + shift, "op": "==", "right": 1, "within": rng.choice([3, 6, 12])}]
        name = f"previous-day raid + MSS {side}"
        if intraday:
            inds.append({"id": "t", "type": "session"})
            zone = rng.choice(["london", "ny_am", "silver_bullet"])
            entry.append({"left": "t." + zone, "op": "==", "right": 1})
            name += f" in {zone}"
    spec = {"name": name, "idea": "ICT sequence generated by the village's random search",
            "indicators": inds, "entry_long": [], "exit_long": [], "entry_short": [],
            "exit_short": [], "stop_loss_pct": stop, "take_profit_pct": take}
    spec["entry_" + side] = entry
    # Exit on a shift the other way, if stop or target did not hit first.
    other = "mss_down" if long_side else "mss_up"
    spec["exit_" + side] = [{"left": "st." + other, "op": "==", "right": 1}]
    return spec


def mutate(spec: dict, rng: random.Random) -> dict:
    """Jitter the numbers in a spec by up to about 25%."""
    spec = copy.deepcopy(spec)
    for ind in spec.get("indicators", []):
        for k, v in list(ind.items()):
            if k in ("id", "type", "source") or not isinstance(v, (int, float)) or v == 0:
                continue
            if isinstance(v, int):
                ind[k] = max(2, int(round(v * rng.uniform(0.75, 1.25))))
            else:
                ind[k] = round(max(0.1, v * rng.uniform(0.8, 1.2)), 2)
    for key in strat.RULE_KEYS:
        for cond in spec.get(key) or []:
            # Flag checks ("== 1") must stay exact; thresholds can move.
            if (isinstance(cond.get("right"), (int, float)) and cond["right"] != 0
                    and cond.get("op") != "=="):
                cond["right"] = round(cond["right"] * rng.uniform(0.85, 1.15), 2)
            if cond.get("within"):
                cond["within"] = max(1, min(strat.MAX_WITHIN,
                                            int(round(cond["within"] * rng.uniform(0.7, 1.3)))))
    for key in ("stop_loss_pct", "take_profit_pct"):
        if spec.get(key) and rng.random() < 0.5:
            spec[key] = round(spec[key] * rng.uniform(0.7, 1.3), 1)
    return spec


_SUFFIX = re.compile(r"\s*\((tuned|simplified|tweaked by [^)]*)\)\s*$")


def base_name(name: str) -> str:
    """'EMA cross (tuned) (tweaked by Bo)' -> 'EMA cross'."""
    while _SUFFIX.search(name):
        name = _SUFFIX.sub("", name)
    return name


def compact(spec: dict) -> str:
    return re.sub(r"\s+", " ", json.dumps(spec))
