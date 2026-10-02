# Agent Village

A village of AI agents that research trading strategies together. They read your PDFs, invent
strategies from indicators, backtest them on your price data, criticise each other, and get better
round after round. It runs on your own computer.

## Quick start for your own research (futures data, PDFs, videos, Pine indicators)

Put your files in these folders inside `agent_village`:

```
data/      NQ, ES, YM price files (.csv or .txt; 1-minute is fine, any common export format)
papers/    PDFs, notes, your Pine Script indicators (.pine or .txt), subtitles (.srt/.vtt)
videos/    videos or audio (.mp4, .mkv, .mov, .mp3 ...)
```

Then double-click **`research.bat`** (Windows), or run the steps yourself:

```
pip install -r requirements.txt -r requirements-extra.txt
python -m village transcribe videos                 # videos -> text in papers/transcripts
python -m village run --data data --timeframe 15min --tz America/New_York --team mixed ^
       --scout 3M --expand 6M,1Y,3Y,10Y,all --rounds 20
```

(On Mac/Linux write the command on one line, or use `\` instead of `^`.) Set `--tz` to the time
zone your data's times are in: many futures exports use exchange time (`America/Chicago`) or New
York time. When it's done, open the newest `.html` file in `reports/`, and paste
`reports/best_strategy.pine` into TradingView.

A browser window opens with **the live village**: every villager is a character in a little
town (library, trading floor, workshop, inventor's lab, analyst, town hall). Speech bubbles show
what each one just did, villagers walk to **the whiteboard** when they write on it, and the
side panels show every whiteboard note, the leaderboard, the best score per round and what the
library found. You can open it on its own any time with `python -m village watch` (it reads
`village.db`, also while a run is going).

What happens:

1. **The Librarian reads everything** in `papers/`: PDFs, transcripts, and your Pine indicators
   (for those she works out exactly what they compute and when they signal).
2. **Scout:** the team hunts on the last 3 months before the vault. Small data, fast rounds.
3. **Expand:** setups that work climb a ladder: 6 months, 1 year, 3 years, 10 years, all. At each
   step the team gets a few rounds to adapt them, and only setups that still work, including on
   the step's own unseen part, climb on. The report shows the ladder.
4. **The final run** on all data starts from the survivors and ends by opening the vault: the
   most recent 30% of your history, which nobody touched until then.
5. **Pine Script:** the best setup that held up is written to `reports/best_strategy.pine`.
   It plots every setup on the chart, sends alerts, and runs in TradingView's Strategy Tester
   with the same logic as the village.

## Who lives in the village

| Villager | Job |
|---|---|
| **Lena the Librarian** | Reads every PDF, `.txt` and `.md` file in `papers/` and posts trading ideas to the board. |
| **Tom** (trend), **Rita** (mean reversion), **Bo** (breakout) | Quants. Each round, each one proposes a strategy in their own style, building on the board's ideas, the leaderboard and the critic's notes. |
| **The Backtester** | Not an AI. It runs every strategy on your data, with fees, and fills orders at the next bar's open so nobody can cheat by seeing the future. |
| **Tess the Tuner** | Not an AI. Each round she tries a dozen small variations of a leading strategy's numbers and keeps one only if it is clearly better. |
| **Nova the Inventor** | The imagination. She treats price as an algorithm and hunts its rules. With an AI she writes hypotheses on the whiteboard and turns the best one into a strategy herself. Every round she also tests two inventions: grafting an idea onto a leader, or crossbreeding two leaders into a child. Her idea families: **time** (macros like 09:50-10:10, quarter-hour cycles, minutes since an event), **opens** (midnight, 08:30, 09:30), **sessions** (raids of the London or pre-market high/low), **numbers** (round levels, digital roots, counts), **sequence**, **symmetry** (equal times) and **power of 3**. |
| **Carl the Critic** | Reviews each round: too few trades, deep drawdowns, losing to buy & hold, overfitting. |
| **Maya the Mayor** | Writes the final report and opens **the vault**. |

**The vault:** the agents only ever see the first 70% of your data (the *train* period). The last
30% stays sealed. The final report shows how each strategy did on that unseen data. A strategy
that only works on the train period was overfit, meaning it memorised noise. This is the honest
check that tells you whether the village found something real.

**The whiteboard:** a shared board where everyone writes what they notice: "AHA: new leader...",
"this piece is ESSENTIAL", "dead end: this hurts", Nova's hypotheses, and the AI agents' own notes.
Everyone reads it every round, so the team builds on each other's findings. It is in the report.

**Invented features:** agents can create their own measurements with small formulas, for
example `digital_root(count(st.bos_up, 50))`, `minutes_since(sw.bull)`, `day_count(fvg.bull)`
(how many FVGs so far today), `ny_minute()` and `minute_of_hour()` (the clock),
`value_at(open, 0)` (midnight open), `session_high(120, 300)` (London high), `bars_today()`,
`bars_since(...)`, `prev(...)`, `change(...)`, rolling `mean/sum/highest/lowest/std`,
`where(...)`, arithmetic and comparisons.
Only these building blocks are allowed (no code), they only look backwards in time, and they are
translated into the Pine Script too.

**The robust score:** the leaderboard does not just reward the most profit. It splits the train
period into slices and ranks strategies by how well they did in *every* slice (and every market),
minus a penalty when the slices disagree. A strategy that made all its money in one lucky year
ranks below one that earned steadily. The village also refuses to test the exact same rules
twice, and the report says how many strategies were tried in total. The more tries, the more
likely the winner is just lucky.

The village also rejects strategies that never trade, and ones that trade *exactly* like one it
already has (extra rules that change nothing are not a discovery).

Everything the village learns is stored in `village.db`. Run it again and it continues where it
stopped.

## Setup (once)

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/)). On Windows, tick
"Add Python to PATH" during install.

Open a terminal in this `agent_village` folder and run:

```
pip install -r requirements.txt
```

On Windows you can instead double-click **`start.bat`**. It installs everything and runs five
rounds.

### Give the villagers a brain (pick one)

1. **Free and local, recommended: Ollama.** Install it from [ollama.com](https://ollama.com),
   then download a model:
   ```
   ollama pull deepseek-coder-v2
   ```
   Any Ollama model works. General chat models often reason about markets better than coding
   models, for example `ollama pull qwen2.5:14b` and then `--llm ollama:qwen2.5:14b`. A bigger
   model gives smarter villagers but runs slower.
2. **Claude (paid API).** Set an `ANTHROPIC_API_KEY` from
   [console.anthropic.com](https://console.anthropic.com) and use `--llm claude`. API usage is
   billed separately from a Claude.ai subscription.
3. **No brain.** `--llm none` runs simple built-in rules (random search). It is good for trying
   things out, and it is the baseline the AI villagers should beat.

By default (`--llm auto`) the village uses Ollama if it is running, then Claude if a key is set,
then no brain.

## Run it

Try it on built-in synthetic data first:

```
python -m village run --rounds 5
```

With your own data and papers:

1. Get price data. Download it from Yahoo Finance in one line:
   ```
   python -m village fetch SPY
   python -m village fetch BTC-USD
   ```
   Or put your own CSV in `data/`. It needs a date/time column and `open, high, low, close`
   (`volume` is optional). Exports from Yahoo Finance and TradingView usually work as they are.
2. Put PDFs, books or notes in `papers/`.
3. Run:
   ```
   python -m village run --data data/BTC-USD_1d.csv --rounds 10
   ```

**Test on several markets at once:** pass several files or a whole folder, for example
`--data data/` or `--data data/SPY_1d.csv data/QQQ_1d.csv`. A strategy then has to work on all
of them to rank well. This is the strongest protection against fooling yourself.

The report lands in `reports/` in two versions. Open the `.html` one in your browser: it has the
leaderboard, equity charts with the vault period shaded, and the Mayor's summary. The best
strategy that held up in the vault is saved to `reports/best_strategy.json`.

### Other commands

```
python -m village board --data data/SPY_1d.csv        # leaderboard + latest critique
python -m village backtest my_strategy.json --data data/SPY_1d.csv
python -m village fetch EURUSD=X --period 10y        # download more data
python -m village pine reports/best_strategy.json    # strategy -> TradingView Pine Script
python -m village pine --id 42                       # or any strategy from the board
python -m village transcribe videos                  # videos -> text for the Librarian
python -m village watch                              # open the live village
python -m village sample --out data/sample.csv        # write synthetic data
python -m village run --help                          # every option
```

Useful options for `run`: `--watch` (open the live village), `--fee-bps 10` (fees per side; crypto and forex often cost more than
the default 5), `--quants trend,trend,reversion` (choose who lives in the village), and
`--train 0.6` (seal a bigger vault), `--no-tuner` (leave Tess out).

## The ICT team: testing the "algorithm"

ICT says price is delivered by an algorithm: it runs to liquidity (stops above highs and below
lows), leaves imbalances (fair value gaps), shifts structure, and then delivers to the next
target, on a clock (kill zones). Every part of that claim can be written as an exact rule, so
the village can test it piece by piece and as a full sequence.

```
python -m village run --team ict --data data/EURUSD_15m.csv --tz UTC --rounds 20
```

| Villager | Hunts for |
|---|---|
| **Ivy** the liquidity hunter | sweep of a swing high/low → market structure shift → entry on the retrace into the fair value gap |
| **Ian** the order-block hunter | order blocks in discount (longs) or premium (shorts), OTE retracements |
| **Iris** the time hunter | kill zones (London, NY AM, silver bullet) and raids on the previous day's high/low |
| **Ada** the Analyst | takes the leader apart one condition at a time and reports which pieces are ESSENTIAL, which add nothing and which HURT. If dropping a piece helps, she submits the simpler version herself. |

With `--team ict` or `--team mixed` the whole team also gets the algorithm mission: treat price
as the output of a delivery algorithm and reverse-engineer its rules (time, numbers, sequence,
symmetry) instead of trading textbook indicators. Every rule still has to pass the vault.

The whole team shares one mission and reads Ada's findings every round, so they converge on the
pieces that carry the edge and drop the decoration. `--team mixed` adds the trend and reversion
quants as a control group: if ICT setups cannot beat simple moving averages, that tells you
something too. `--mission "..."` gives the team your own instructions.

ICT building blocks the agents can use: `sweep`, `structure` (BOS / MSS), `fvg`,
`order_block`, `swings`, `premium_discount` (with OTE), `displacement`, `session` (kill zones in
New York time) and `prev_day` (previous day high/low). A condition can carry `"within": N` ("this
happened in the last N bars"), which is how steps are chained into a sequence:

```json
"entry_long": [
  {"left": "sw.bull",   "op": "==", "right": 1, "within": 20},
  {"left": "st.mss_up", "op": "==", "right": 1, "within": 10},
  {"left": "low",       "op": "<=", "right": "gap.bull_top"}
]
```

Kill zones need **intraday** data and the right time zone: pass `--tz` with the zone your CSV's
times are in (default UTC). Yahoo only keeps short intraday history
(`python -m village fetch EURUSD=X --interval 1h --period 2y`); for more, export from your broker
or TradingView.

**The luck check.** On short intraday samples, big numbers come easily by chance: on two months
of 15-minute *random* prices, a strategy's Sharpe swings by ±2.4 from luck alone. Every result
therefore has a **t-stat**, and the report shows the "luck bar": the t-stat the best of all the
tried strategies would reach with no edge at all. A strategy only counts as a real find when it
holds up in the vault with a vault t-stat of 2 or more. Otherwise it is marked "could be luck".

## Speed and big data

18 years of 1-minute bars is about 6 million bars per market. Loading takes some seconds per
file; after that use `--timeframe` (5min, 15min, 1h ...) for the research. With `numba`
installed (`pip install numba`, also in requirements-extra.txt) the backtester and the ICT zones are compiled, and
indicators are cached between variations. On the developer's test machine, three markets of
two years of 1-minute data (15min research) ran the full scout, expand and final chain in about
20 seconds without an AI. With a local AI each agent's turn takes as long as the model needs to
answer, so the AI is the slow part, not the backtests.

## How strategies look

The agents write strategies as JSON, never as code, so nothing they produce can run on your
computer. Example:

```json
{
  "name": "EMA trend with RSI filter",
  "idea": "Ride trends, but don't buy when overbought",
  "indicators": [
    {"id": "fast", "type": "ema", "period": 12},
    {"id": "slow", "type": "ema", "period": 48},
    {"id": "rsi", "type": "rsi", "period": 14}
  ],
  "entry_long": [{"left": "fast", "op": "crosses_above", "right": "slow"},
                 {"left": "rsi", "op": "<", "right": 70}],
  "exit_long":  [{"left": "fast", "op": "crosses_below", "right": "slow"}],
  "stop_loss_pct": 5
}
```

The classic indicators are `sma`, `ema`, `rsi`, `macd`, `bbands`, `atr`, `roc`, `stoch_k` and
`zscore` (in `village/indicators.py`); the ICT ones are in `village/ict.py`. Add one to a
`CATALOG` there and the agents can use it.

## Project layout

```
village/
  agents.py      the villagers and their prompts
  village.py     the round loop
  report.py      the Markdown and HTML reports
  backtest.py    the backtester and its metrics
  strategy.py    the strategy JSON format and its validation
  indicators.py  technical indicators
  ict.py         ICT building blocks (sweeps, FVGs, order blocks, kill zones, ...)
  features.py    the formula language for invented features (and its Pine translation)
  stages.py      scout-then-expand ladder
  pine.py        export to TradingView Pine Script
  live.py        the live village web page (characters, whiteboard)
  fast.py        compiled hot loops (numba)
  pdfs.py        reading PDFs and notes
  board.py       shared memory (SQLite)
  llm.py         Ollama / Claude / none
tests/           run with: python -m pytest
```

## A word of honesty

No backtest "cracks" a market, and nobody has shown that a single hidden algorithm delivers
price. What the village can do is turn each claim into a rule and measure it honestly. Most
strategies that look great on past data fail on new data, and the vault and the luck check exist
to show you that early. Treat a strategy that holds up as a candidate for
paper trading, not as a money machine. This is not financial advice.
