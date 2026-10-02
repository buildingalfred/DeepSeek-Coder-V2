# Agent Village

A village of AI agents that research trading strategies together. They read your PDFs, invent
strategies from indicators, backtest them on your price data, criticise each other, and get better
round after round. It runs on your own computer.

## Who lives in the village

| Villager | Job |
|---|---|
| **Lena the Librarian** | Reads every PDF, `.txt` and `.md` file in `papers/` and posts trading ideas to the board. |
| **Tom** (trend), **Rita** (mean reversion), **Bo** (breakout) | Quants. Each round, each one proposes a strategy in their own style, building on the board's ideas, the leaderboard and the critic's notes. |
| **The Backtester** | Not an AI. It runs every strategy on your data, with fees, and fills orders at the next bar's open so nobody can cheat by seeing the future. |
| **Tess the Tuner** | Not an AI. Each round she tries a dozen small variations of a leading strategy's numbers and keeps one only if it is clearly better. |
| **Carl the Critic** | Reviews each round: too few trades, deep drawdowns, losing to buy & hold, overfitting. |
| **Maya the Mayor** | Writes the final report and opens **the vault**. |

**The vault:** the agents only ever see the first 70% of your data (the *train* period). The last
30% stays sealed. The final report shows how each strategy did on that unseen data. A strategy
that only works on the train period was overfit, meaning it memorised noise. This is the honest
check that tells you whether the village found something real.

**The robust score:** the leaderboard does not just reward the most profit. It splits the train
period into slices and ranks strategies by how well they did in *every* slice (and every market),
minus a penalty when the slices disagree. A strategy that made all its money in one lucky year
ranks below one that earned steadily. The village also refuses to test the exact same rules
twice, and the report says how many strategies were tried in total. The more tries, the more
likely the winner is just lucky.

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
python -m village sample --out data/sample.csv        # write synthetic data
python -m village run --help                          # every option
```

Useful options for `run`: `--fee-bps 10` (fees per side; crypto and forex often cost more than
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
