"""The live village: a local web page that shows the villagers at work.

  python -m village watch            (or: python -m village run ... --watch)

It serves a page on http://localhost:8765 that reads village.db every two seconds: every
villager's house and what they are doing, the whiteboard, the leaderboard and the activity feed.
Nothing leaves your computer. The page only reads the database, so it can run while the village
works.
"""

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .board import Board

# first name -> (role, building, colour, soft colour, icon)
ROSTER = {
    "Maya": ("Mayor", "Town Hall", "#b0457a", "#f6dcea", "\U0001F3DB\uFE0F"),
    "Lena": ("Librarian", "Library", "#7c5cbf", "#e7defa", "\U0001F4DA"),
    "Nova": ("Inventor", "Inventor's Lab", "#3f9a4e", "#dcf1dc", "\U0001F4A1"),
    "Ada": ("Analyst", "Observatory", "#2f7fd8", "#dbe9fb", "\U0001F50D"),
    "Tess": ("Tuner", "Workshop", "#e0702f", "#fbe3d3", "\U0001F527"),
    "Ivy": ("Liquidity hunter", "Liquidity Desk", "#169c78", "#d5f1e8", "\U0001F3AF"),
    "Ian": ("Order-block hunter", "Block House", "#5146b8", "#e1defa", "\U0001F9F1"),
    "Iris": ("Time hunter", "Clock Tower", "#d65a92", "#fadbe9", "\u23F1\uFE0F"),
    "Tom": ("Trend quant", "Trend Desk", "#2a6fd6", "#dce8fb", "\U0001F4C8"),
    "Rita": ("Reversion quant", "Reversion Desk", "#d9473f", "#fadcd9", "\U0001F504"),
    "Bo": ("Breakout quant", "Breakout Desk", "#c98a00", "#f8ecc8", "\U0001F680"),
    "Carl": ("Critic", "Critic's Corner", "#6b6b6b", "#e8e8e6", "\U0001F9D0"),
}
CORE = ["Maya", "Lena", "Nova", "Ada", "Tess", "Carl"]
QUANTS = {"Tom", "Rita", "Bo", "Ivy", "Ian", "Iris"}


def _villager(first: str, a: dict | None = None, round_: int = 0) -> dict:
    role, building, color, soft, icon = ROSTER[first]
    a = a or {}
    return {"name": first, "full": a.get("full", first), "role": role, "building": building,
            "color": color, "soft": soft, "icon": icon, "text": a.get("text", ""),
            "time": a.get("time", 0), "this_round": bool(a) and a.get("round") == round_}


def state(db_path: str, dataset: str | None = None) -> dict:
    """Everything the page needs, read straight from the village's memory."""
    if not Path(db_path).exists():
        return {"ready": False, "message": "No village yet - start a run (the villagers are waiting)",
                "villagers": [_villager(n) for n in ROSTER if n in CORE or n in ("Ivy", "Ian", "Iris")]}
    board = Board(db_path, readonly=True)
    try:
        q = board.db.execute
        try:
            last = q("SELECT dataset, round, created FROM events ORDER BY id DESC LIMIT 1").fetchone()
        except Exception:  # older database without the events table
            last = None
        ds = dataset or (last["dataset"] if last else None)
        if ds is None:
            row = q("SELECT dataset FROM strategies ORDER BY id DESC LIMIT 1").fetchone()
            ds = row["dataset"] if row else ""
        events = [dict(r) for r in q("SELECT created, round, agent, text FROM events WHERE dataset = ? "
                                     "ORDER BY id DESC LIMIT 200", (ds,))] if last else []
        round_ = max((e["round"] or 0 for e in events), default=0)
        agents, brain = {}, ""
        for e in events:
            first = e["agent"].split(" ")[0]
            if first in ROSTER and first not in agents:
                agents[first] = {"full": e["agent"], "text": e["text"], "time": e["created"],
                                 "round": e["round"]}
            if e["agent"] == "village" and e["text"].startswith("brain:") and not brain:
                brain = e["text"][6:].strip()
        if last and not brain:
            row = q("SELECT text FROM events WHERE agent = 'village' AND text LIKE 'brain:%' "
                    "ORDER BY id DESC LIMIT 1").fetchone()
            brain = row["text"][6:].strip() if row else ""
        present = set(agents) | set(CORE)
        for e in q("SELECT DISTINCT author FROM strategies WHERE dataset = ?", (ds,)):
            first = (e["author"] or "").split(" ")[0]
            if first in ROSTER:
                present.add(first)
        if not present & QUANTS:
            present |= {"Ivy", "Ian", "Iris"}
        villagers = [_villager(n, agents.get(n), round_) for n in ROSTER if n in present]
        wb = [{"round": r["round"], "author": r["author"], "text": r["content"], "time": r["created"]}
              for r in board.notes("whiteboard", 80, ds)]
        ideas = [{"text": r["content"]} for r in board.notes("idea", 12)]
        leaders = [{"id": r["id"], "name": r["name"], "author": r["author"],
                    "robust": r["train"].get("robust_sharpe"), "t": r["train"].get("t_stat"),
                    "trades": r["train"].get("trades")} for r in board.leaderboard(ds, 10)]
        progress = [{"round": r[0], "best": r[1]} for r in q(
            "SELECT round, MAX(score) FROM strategies WHERE dataset = ? AND error IS NULL AND score > -99 "
            "GROUP BY round ORDER BY round", (ds,))]
        best = None
        for p in progress:
            best = p["best"] if best is None else max(best, p["best"])
            p["best_so_far"] = round(best, 3)
        tested = q("SELECT COUNT(*) FROM strategies WHERE dataset = ? AND error IS NULL",
                   (ds,)).fetchone()[0]
        datasets = [r[0] for r in q("SELECT dataset FROM events GROUP BY dataset ORDER BY MAX(id) DESC")] \
            if last else []
        return {"ready": True, "dataset": ds, "datasets": datasets[:12], "brain": brain,
                "round": round_, "last_event": last["created"] if last else 0,
                "now": time.time(), "villagers": villagers, "events": events[:60], "whiteboard": wb,
                "ideas": ideas, "leaders": leaders, "progress": progress, "tested": tested}
    finally:
        board.close()


def make_handler(db_path: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.startswith("/state"):
                ds = None
                if "?ds=" in self.path:
                    from urllib.parse import unquote
                    ds = unquote(self.path.split("?ds=", 1)[1]) or None
                try:
                    body = json.dumps(state(db_path, ds)).encode()
                except Exception as e:  # the village may be mid-write; try again next poll
                    body = json.dumps({"ready": False, "message": f"busy: {e}"}).encode()
                self._send(body, "application/json")
            elif self.path.startswith("/favicon"):
                self.send_response(204)
                self.end_headers()
            elif self.path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console for the villagers
            pass

    return Handler


def serve(db_path: str = "village.db", port: int = 8765, open_browser: bool = True,
          background: bool = False):
    """Start the live view. With background=True it runs in a thread and returns the URL."""
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler(db_path))
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError(f"no free port between {port} and {port + 19}")
    url = f"http://localhost:{server.server_address[1]}/"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    if background:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return url
    print(f"Live village at {url}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return url


PAGE = (Path(__file__).with_name("live.html")).read_text(encoding="utf-8")
