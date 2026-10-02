"""The village board: shared memory every agent reads from and writes to.

Stored in SQLite so the village remembers ideas, strategies and lessons between runs.
"""

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, round INTEGER, author TEXT, kind TEXT, content TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, created REAL, dataset TEXT, round INTEGER, agent TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY, round INTEGER, author TEXT, name TEXT, spec TEXT,
    train TEXT, test TEXT, score REAL, error TEXT, dataset TEXT, created REAL
);
"""


class Board:
    def __init__(self, path: str | Path = "village.db", readonly: bool = False):
        self.path = str(path)
        if readonly:
            self.db = sqlite3.connect(f"file:{Path(self.path).resolve().as_posix()}?mode=ro",
                                      uri=True, check_same_thread=False, timeout=5)
            self.db.row_factory = sqlite3.Row
            return
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            # Lets the live view read while the village writes.
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, much faster commits
        self.db.executescript(SCHEMA)
        # Columns added after the first release; older village.db files get them on open.
        for table, column in (("strategies", "markets"), ("notes", "dataset")):
            names = [r[1] for r in self.db.execute(f"PRAGMA table_info({table})")]
            if column not in names:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        self.db.commit()

    def close(self):
        self.db.close()

    # --- notes: ideas from the library, critiques, lessons -------------------------------
    def post(self, round_: int, author: str, kind: str, content: str, dataset: str = "") -> None:
        self.db.execute("INSERT INTO notes (round, author, kind, content, created, dataset)"
                        " VALUES (?,?,?,?,?,?)", (round_, author, kind, content, time.time(), dataset))
        self.db.commit()

    def notes(self, kind: str | None = None, limit: int = 20,
              dataset: str | None = None) -> list[sqlite3.Row]:
        where, args = [], []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if dataset is not None:
            where.append("dataset = ?")
            args.append(dataset)
        q = ("SELECT * FROM notes" + (" WHERE " + " AND ".join(where) if where else "")
             + " ORDER BY id DESC LIMIT ?")
        return list(reversed(self.db.execute(q, (*args, limit)).fetchall()))

    def has_source(self, source: str) -> bool:
        row = self.db.execute("SELECT 1 FROM notes WHERE kind = 'source' AND content = ?",
                              (source,)).fetchone()
        return row is not None

    # --- events: what each villager is doing (for the live view) ---------------------------
    def event(self, dataset: str, round_: int, agent: str, text: str) -> None:
        self.db.execute("INSERT INTO events (created, dataset, round, agent, text) VALUES (?,?,?,?,?)",
                        (time.time(), dataset, round_, agent, text))
        self.db.commit()

    # --- strategies ----------------------------------------------------------------------
    def add_strategy(self, round_, author, spec, train, test, score, error, dataset,
                     markets=None) -> int:
        cur = self.db.execute(
            "INSERT INTO strategies (round, author, name, spec, train, test, score, error, dataset,"
            " created, markets) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (round_, author, spec.get("name", "unnamed") if isinstance(spec, dict) else "invalid",
             json.dumps(spec), json.dumps(train), json.dumps(test), score, error, dataset,
             time.time(), json.dumps(markets)))
        self.db.commit()
        return cur.lastrowid

    def leaderboard(self, dataset: str, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM strategies WHERE dataset = ? AND error IS NULL ORDER BY score DESC, id"
            " LIMIT ?", (dataset, limit)).fetchall()
        return [_row(r) for r in rows]

    def by_author(self, dataset: str, author: str, limit: int = 5) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM strategies WHERE dataset = ? AND author = ? ORDER BY id DESC LIMIT ?",
            (dataset, author, limit)).fetchall()
        return [_row(r) for r in rows]

    def all_strategies(self, dataset: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM strategies WHERE dataset = ? AND error IS NULL",
                               (dataset,)).fetchall()
        return [_row(r) for r in rows]

    def trial_count(self, dataset: str) -> int:
        """Every strategy backtested on this dataset, including the tuner's discarded variations."""
        tested = self.db.execute("SELECT COUNT(*) FROM strategies WHERE dataset = ? AND error IS NULL",
                                 (dataset,)).fetchone()[0]
        extra = sum(int(n["content"]) for n in self.notes("trials", 10**9, dataset))
        return tested + extra

    def in_round(self, dataset: str, round_: int) -> list[dict]:
        rows = self.db.execute("SELECT * FROM strategies WHERE dataset = ? AND round = ? ORDER BY id",
                               (dataset, round_)).fetchall()
        return [_row(r) for r in rows]

    def last_round(self) -> int:
        row = self.db.execute(
            "SELECT MAX(r) FROM (SELECT MAX(round) r FROM notes UNION SELECT MAX(round) FROM strategies)"
        ).fetchone()
        return row[0] or 0


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in ("spec", "train", "test", "markets"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d
