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
| **Carl the Critic** | Reviews each round: too few trades, deep drawdowns, losing to buy & hold, overfitting. |
| **Maya the Mayor** | Writes the final report and opens **the vault**. |

**The vault:** the agents only ever see the first 70% of your data (the *train* period). The last
30% stays sealed. The final report shows how each strategy did on that unseen data. A strategy
that only works on the train period was overfit, meaning it memorised noise. This is the honest
check that tells you whether the village found something real.

Everything the village learns is stored in `village.db`. Run it again and it continues where it
stopped.

## Setup (once)

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/)). On Windows, tick
"Add Python to PATH" during install.

Open a terminal in this `agent_village` folder and run:

```
pip install -r requirements.txt
```

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

1. Put a price CSV in `data/`. It needs a date/time column and `open, high, low, close`
   (`volume` is optional). Exports from Yahoo Finance and TradingView usually
   work as they are.
2. Put PDFs, books or notes in `papers/`.
3. Run:
   ```
   python -m village run --data data/BTCUSD.csv --rounds 10
   ```

The report lands in `reports/`, and the best strategy that held up in the vault is saved to
`reports/best_strategy.json`.

### Other commands

```
python -m village board --data data/BTCUSD.csv        # leaderboard + latest critique
python -m village backtest my_strategy.json --data data/BTCUSD.csv
python -m village sample --out data/sample.csv        # write synthetic data
python -m village run --help                          # every option
```

Useful options for `run`: `--fee-bps 10` (fees per side; crypto and forex often cost more than
the default 5), `--quants trend,trend,reversion` (choose who lives in the village), and
`--train 0.6` (seal a bigger vault).

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

The indicators are `sma`, `ema`, `rsi`, `macd`, `bbands`, `atr`, `roc`, `stoch_k` and `zscore`.
They live in `village/indicators.py`. Add one to `CATALOG` there and the agents can use it.

## Project layout

```
village/
  agents.py      the villagers and their prompts
  village.py     the round loop and the report
  backtest.py    the backtester and its metrics
  strategy.py    the strategy JSON format and its validation
  indicators.py  technical indicators
  pdfs.py        reading PDFs and notes
  board.py       shared memory (SQLite)
  llm.py         Ollama / Claude / none
tests/           run with: python -m pytest
```

## A word of honesty

No backtest "cracks" a market. Most strategies that look great on past data fail on new data,
and the vault exists to show you that early. Treat a strategy that holds up as a candidate for
paper trading, not as a money machine. This is not financial advice.
