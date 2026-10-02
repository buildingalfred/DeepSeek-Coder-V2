"""The live village: a local web page that shows the villagers at work.

  python -m village watch            (or: python -m village run ... --watch)

It serves a page on http://localhost:8765 that reads village.db every two seconds: who is doing
what (speech bubbles), the whiteboard, the leaderboard and the best score so far. Nothing leaves
your computer. The page only reads the database, so it can run while the village works.
"""

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .board import Board

# first name -> (role, home x %, home y %, body colour, hair colour, prop emoji)
ROSTER = {
    "Lena": ("Librarian", 12, 60, "#7c5cbf", "#3b2a1a", "\U0001F4DA"),
    "Tom": ("Trend quant", 33, 76, "#2a78d6", "#1d1d1d", "\U0001F4C8"),
    "Rita": ("Reversion quant", 43, 80, "#e34948", "#7a3b12", "\U0001F504"),
    "Bo": ("Breakout quant", 53, 76, "#eda100", "#2b2b2b", "\U0001F680"),
    "Ivy": ("Liquidity hunter", 33, 76, "#1baf7a", "#6b2d0f", "\U0001F3AF"),
    "Ian": ("Order-block hunter", 43, 80, "#4a3aa7", "#111111", "\U0001F9F1"),
    "Iris": ("Time hunter", 53, 76, "#e87ba4", "#d9a441", "⏱️"),
    "Tess": ("Tuner", 71, 78, "#eb6834", "#5a3a22", "\U0001F527"),
    "Nova": ("Inventor", 88, 52, "#008300", "#ff9ad5", "\U0001F4A1"),
    "Ada": ("Analyst", 72, 46, "#3987e5", "#222222", "\U0001F50D"),
    "Carl": ("Critic", 22, 82, "#6b6b6b", "#c9c9c9", "\U0001F9D0"),
    "Maya": ("Mayor", 90, 82, "#9b2c6e", "#2a1a10", "\U0001F3A9"),
}
CORE = ["Lena", "Tess", "Nova", "Ada", "Carl", "Maya"]
WHITEBOARD_SPOT = (50, 40)


def _villager(first: str) -> dict:
    role, x, y, body, hair, prop = ROSTER[first]
    return {"name": first, "full": first, "role": role, "x": x, "y": y, "body": body, "hair": hair,
            "prop": prop, "text": "", "time": 0}


def state(db_path: str, dataset: str | None = None) -> dict:
    """Everything the page needs, read straight from the village's memory."""
    if not Path(db_path).exists():
        return {"ready": False, "message": "No village yet - start a run (the villagers are waiting)",
                "villagers": [_villager(n) for n in [*CORE, "Ivy", "Ian", "Iris"]],
                "whiteboard_spot": WHITEBOARD_SPOT}
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
                                     "ORDER BY id DESC LIMIT 60", (ds,))] if last else []
        agents = {}
        for e in events:
            first = e["agent"].split(" ")[0]
            if first in ROSTER and first not in agents:
                agents[first] = {"full": e["agent"], "text": e["text"], "time": e["created"]}
        present = set(agents) | set(CORE)
        for e in q("SELECT DISTINCT author FROM strategies WHERE dataset = ?", (ds,)):
            first = (e["author"] or "").split(" ")[0]
            if first in ROSTER:
                present.add(first)
        if not present & {"Tom", "Rita", "Bo", "Ivy", "Ian", "Iris"}:
            present |= {"Ivy", "Ian", "Iris"}
        # Two quant teams share the trading floor; shift them if both are present.
        villagers = []
        for first in ROSTER:
            if first not in present:
                continue
            role, x, y, body, hair, prop = ROSTER[first]
            if present & {"Tom", "Rita", "Bo"} and present & {"Ivy", "Ian", "Iris"}:
                # both quant teams: two rows on the trading floor
                x, y = {"Tom": (28, 64), "Rita": (38, 66), "Bo": (48, 64),
                        "Ivy": (33, 86), "Ian": (43, 88), "Iris": (53, 86)}.get(first, (x, y))
            a = agents.get(first, {})
            villagers.append({"name": first, "full": a.get("full", first), "role": role, "x": x,
                              "y": y, "body": body, "hair": hair, "prop": prop,
                              "text": a.get("text", ""), "time": a.get("time", 0)})
        wb = [{"round": r["round"], "author": r["author"], "text": r["content"], "time": r["created"]}
              for r in board.notes("whiteboard", 40, ds)]
        ideas = [{"text": r["content"]} for r in board.notes("idea", 8)]
        leaders = [{"id": r["id"], "name": r["name"], "author": r["author"],
                    "robust": r["train"].get("robust_sharpe"), "t": r["train"].get("t_stat"),
                    "trades": r["train"].get("trades")} for r in board.leaderboard(ds, 8)]
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
        return {"ready": True, "dataset": ds, "datasets": datasets[:12],
                "round": last["round"] if last else 0, "last_event": last["created"] if last else 0,
                "now": time.time(), "villagers": villagers, "events": events[:40], "whiteboard": wb,
                "ideas": ideas, "leaders": leaders, "progress": progress, "tested": tested,
                "whiteboard_spot": WHITEBOARD_SPOT}
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


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Village</title>
<style>
:root { --bg:#14161c; --panel:#1d2029; --panel2:#252935; --ink:#f3f1ea; --ink2:#b9b6ab;
  --muted:#8a877d; --line:#323745; --accent:#f5c451; --good:#4ade80; --bad:#fb923c; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { display:flex; flex-wrap:wrap; align-items:center; gap:10px 18px; padding:12px 16px;
  border-bottom:1px solid var(--line); }
header h1 { font-size:18px; margin:0; letter-spacing:.2px; }
.chip { background:var(--panel2); border:1px solid var(--line); border-radius:99px; padding:3px 10px;
  color:var(--ink2); font-size:12.5px; white-space:nowrap; }
.chip b { color:var(--ink); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--muted);
  margin-right:6px; vertical-align:middle; }
.dot.on { background:var(--good); box-shadow:0 0 0 0 rgba(74,222,128,.7); animation:ping 1.6s infinite; }
@keyframes ping { 0%{box-shadow:0 0 0 0 rgba(74,222,128,.6)} 100%{box-shadow:0 0 0 9px rgba(74,222,128,0)} }
select { background:var(--panel2); color:var(--ink); border:1px solid var(--line); border-radius:6px;
  padding:3px 6px; max-width:100%; }
main { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:14px; padding:14px 16px 28px; }
@media (max-width: 1000px) { main { grid-template-columns:minmax(0,1fr); } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
.panel h2 { font-size:14px; margin:0 0 8px; color:var(--ink2); font-weight:600;
  text-transform:uppercase; letter-spacing:.6px; }
/* ---- the scene ---- */
.scene { position:relative; width:100%; aspect-ratio: 16 / 8.6; border-radius:12px; overflow:hidden;
  border:1px solid var(--line); background:#9bd3f0; }
.scene svg.bg { position:absolute; inset:0; width:100%; height:100%; }
.villager { position:absolute; width:8.2%; transform:translate(-50%,-100%);
  transition:left 1.4s ease-in-out, top 1.4s ease-in-out; z-index:3; cursor:default; }
.villager .fig { width:100%; animation:breathe 3.2s ease-in-out infinite; transform-origin:50% 100%; }
.villager.active .fig { animation:hop .55s ease-in-out infinite alternate; }
.villager.walking .fig { animation:walk .3s ease-in-out infinite alternate; }
@keyframes breathe { 0%,100%{transform:scaleY(1)} 50%{transform:scaleY(1.025)} }
@keyframes hop { from{transform:translateY(0)} to{transform:translateY(-9%)} }
@keyframes walk { from{transform:rotate(-5deg)} to{transform:rotate(5deg)} }
.villager .tag { position:absolute; left:50%; top:100%; transform:translateX(-50%); margin-top:2px;
  background:rgba(20,22,28,.78); color:#fff; font-size:clamp(9px,1vw,12px); padding:1px 7px;
  border-radius:99px; white-space:nowrap; }
.bubble { position:absolute; left:50%; bottom:104%; transform:translateX(-50%) scale(.6);
  transform-origin:50% 100%; width:clamp(150px,19vw,250px); background:#fffdf5; color:#1d1b16;
  border-radius:12px; padding:7px 9px; font-size:clamp(10px,1vw,12.5px); line-height:1.3;
  box-shadow:0 4px 14px rgba(0,0,0,.25); opacity:0; pointer-events:none;
  transition:opacity .35s, transform .35s; z-index:5; }
.bubble::after { content:""; position:absolute; left:50%; top:100%; transform:translateX(-50%);
  border:8px solid transparent; border-top-color:#fffdf5; }
.bubble b { display:block; font-size:.85em; color:#6b665a; margin-bottom:2px; }
.villager.speaking .bubble, .villager.show .bubble, .villager:hover .bubble {
  opacity:1; transform:translateX(-50%) scale(1); }
.villager:hover, .villager.speaking { z-index:6; }
.villager.edge-l .bubble { left:0; transform:translateX(-10%) scale(1); }
.villager.edge-r .bubble { left:auto; right:0; transform:translateX(10%) scale(1); }
.villager.edge-l .bubble::after { left:22%; } .villager.edge-r .bubble::after { left:78%; }
.boardtext { position:absolute; left:36.5%; top:9%; width:27%; height:22%; overflow:hidden; z-index:2;
  font-family:"Segoe Print","Bradley Hand","Comic Sans MS",cursive; color:#22314f;
  font-size:clamp(8px,.95vw,12.5px); line-height:1.25; }
.boardtext div { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.boardtext .aha { color:#b4232a; font-weight:700; }
.banner { position:absolute; left:1.2%; bottom:2.5%; z-index:4;
  background:rgba(20,22,28,.72); color:#fff; padding:3px 12px; border-radius:99px;
  font-size:clamp(10px,1.1vw,13px); white-space:nowrap; }
/* ---- side panels ---- */
.notes { display:flex; flex-direction:column; gap:8px; max-height:520px; overflow:auto; padding-right:2px; }
.note { border-radius:6px; padding:7px 9px; color:#1d1b16; font-size:12.5px; line-height:1.35;
  box-shadow:0 2px 0 rgba(0,0,0,.25); position:relative; }
.note .who { font-weight:700; font-size:11.5px; opacity:.75; display:block; margin-bottom:1px; }
.note.aha { outline:2px solid #b4232a; }
.note.new { animation:pop .6s ease-out; }
@keyframes pop { from{transform:scale(.85); opacity:.2} to{transform:scale(1); opacity:1} }
table { width:100%; border-collapse:collapse; font-size:12.5px; }
td, th { padding:5px 6px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
tr:last-child td { border-bottom:0; }
.feed { max-height:300px; overflow:auto; font-size:12.5px; }
.feed div { padding:4px 0; border-bottom:1px solid var(--line); color:var(--ink2); }
.feed b { color:var(--ink); }
.grid2 { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; margin-top:14px; }
@media (max-width: 760px) { .grid2 { grid-template-columns:minmax(0,1fr); } }
.spark text { fill:var(--muted); font-size:10px; }
.empty { color:var(--muted); padding:8px 0; }
.side { display:flex; flex-direction:column; gap:14px; min-width:0; }
</style></head>
<body>
<header>
  <h1>&#127968; Agent Village</h1>
  <span class="chip"><span class="dot" id="live"></span><span id="status">connecting…</span></span>
  <span class="chip">round <b id="round">–</b></span>
  <span class="chip">strategies tested <b id="tested">–</b></span>
  <span class="chip">best robust score <b id="best">–</b></span>
  <select id="ds" title="Which board to watch"></select>
</header>
<main>
  <div>
    <div class="scene" id="scene">
      <svg class="bg" viewBox="0 0 1000 538" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#7cc4ec"/><stop offset="1" stop-color="#cdeefc"/></linearGradient>
          <linearGradient id="grass" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#8fd16a"/><stop offset="1" stop-color="#5fae47"/></linearGradient>
        </defs>
        <rect width="1000" height="538" fill="url(#sky)"/>
        <circle cx="905" cy="62" r="34" fill="#ffe27a"/>
        <g fill="#fff" opacity=".9"><ellipse cx="160" cy="70" rx="58" ry="17"/><ellipse cx="200" cy="58" rx="40" ry="17"/>
          <ellipse cx="700" cy="92" rx="52" ry="14"/><ellipse cx="735" cy="82" rx="34" ry="14"/></g>
        <path d="M0 250 Q 250 205 500 240 T 1000 228 V538 H0Z" fill="#a9db8a"/>
        <rect y="268" width="1000" height="270" fill="url(#grass)"/>
        <path d="M430 538 C 470 430, 540 400, 520 300 L 480 300 C 500 400, 400 440, 360 538Z" fill="#e8d3a2" opacity=".9"/>
        <!-- library -->
        <g transform="translate(20,150)"><rect x="0" y="60" width="150" height="120" fill="#c98f5a"/>
          <polygon points="-12,62 75,8 162,62" fill="#8e4b2a"/><rect x="58" y="120" width="34" height="60" fill="#5b3420"/>
          <rect x="16" y="82" width="30" height="26" fill="#ffeaa7"/><rect x="104" y="82" width="30" height="26" fill="#ffeaa7"/>
          <text x="75" y="52" text-anchor="middle" font-size="15" fill="#fff" font-weight="700">LIBRARY</text></g>
        <!-- whiteboard -->
        <g transform="translate(355,30)"><rect x="-6" y="-6" width="302" height="160" rx="8" fill="#6c5a45"/>
          <rect x="0" y="0" width="290" height="148" rx="4" fill="#fbfbf6"/>
          <rect x="40" y="148" width="10" height="100" fill="#6c5a45"/><rect x="240" y="148" width="10" height="100" fill="#6c5a45"/>
          <text x="145" y="-12" text-anchor="middle" font-size="14" fill="#2d3a55" font-weight="700">THE WHITEBOARD</text></g>
        <!-- trading floor -->
        <g transform="translate(300,300)"><rect x="0" y="40" width="300" height="12" rx="4" fill="#7b5b3a"/>
          <rect x="18" y="12" width="70" height="30" rx="3" fill="#24324f"/><rect x="115" y="12" width="70" height="30" rx="3" fill="#24324f"/>
          <rect x="212" y="12" width="70" height="30" rx="3" fill="#24324f"/>
          <polyline points="24,36 36,28 46,32 58,20 80,24" fill="none" stroke="#4ade80" stroke-width="3"/>
          <polyline points="121,22 140,34 156,26 178,36" fill="none" stroke="#fb7185" stroke-width="3"/>
          <polyline points="218,34 236,30 250,22 276,16" fill="none" stroke="#facc15" stroke-width="3"/>
          <text x="150" y="80" text-anchor="middle" font-size="13" fill="#24324f" font-weight="700">TRADING FLOOR</text></g>
        <!-- analyst observatory -->
        <g transform="translate(655,150)"><rect x="0" y="50" width="120" height="96" fill="#9fb6d6"/>
          <path d="M0 52 A60 52 0 0 1 120 52Z" fill="#5d7fb0"/><rect x="88" y="10" width="34" height="12" rx="4" fill="#3b4e70" transform="rotate(-25 100 16)"/>
          <text x="60" y="166" text-anchor="middle" font-size="13" fill="#24324f" font-weight="700">ANALYST</text></g>
        <!-- inventor lab -->
        <g transform="translate(815,140)"><rect x="0" y="50" width="150" height="120" fill="#b7e3c3"/>
          <polygon points="-10,52 75,10 160,52" fill="#3c8b55"/><circle cx="75" cy="94" r="20" fill="#fff7a8" stroke="#3c8b55" stroke-width="4"/>
          <text x="75" y="188" text-anchor="middle" font-size="13" fill="#24324f" font-weight="700">INVENTOR'S LAB</text></g>
        <!-- workshop -->
        <g transform="translate(640,330)"><rect x="0" y="30" width="130" height="80" fill="#e0a46a"/>
          <polygon points="-8,32 65,0 138,32" fill="#9a5b2b"/><rect x="48" y="62" width="34" height="48" fill="#5b3420"/>
          <text x="65" y="128" text-anchor="middle" font-size="13" fill="#24324f" font-weight="700">WORKSHOP</text></g>
        <!-- town hall -->
        <g transform="translate(845,345)"><rect x="0" y="40" width="140" height="90" fill="#efe3c9"/>
          <polygon points="-8,42 70,0 148,42" fill="#b8435f"/><rect x="14" y="58" width="12" height="70" fill="#d8c8a6"/>
          <rect x="60" y="58" width="12" height="70" fill="#d8c8a6"/><rect x="106" y="58" width="12" height="70" fill="#d8c8a6"/>
          <text x="70" y="150" text-anchor="middle" font-size="13" fill="#24324f" font-weight="700">TOWN HALL</text></g>
        <!-- critic bench -->
        <g transform="translate(150,440)"><rect x="0" y="0" width="120" height="12" rx="3" fill="#7b5b3a"/>
          <rect x="10" y="12" width="8" height="22" fill="#5b4128"/><rect x="102" y="12" width="8" height="22" fill="#5b4128"/></g>
        <g fill="#2f7d32"><circle cx="250" cy="250" r="26"/><circle cx="620" cy="255" r="22"/><circle cx="975" cy="300" r="20"/></g>
        <g fill="#6b4a2b"><rect x="246" y="270" width="8" height="22"/><rect x="616" y="272" width="8" height="20"/></g>
      </svg>
      <div class="banner" id="banner">waiting for the village…</div>
      <div class="boardtext" id="boardtext"></div>
      <div id="people"></div>
    </div>
    <div class="grid2">
      <div class="panel"><h2>Leaderboard</h2><div id="leaders"></div></div>
      <div class="panel"><h2>Best score by round</h2><div id="progress"></div>
        <h2 style="margin-top:12px">From the library</h2><div class="feed" id="ideas"></div></div>
    </div>
    <div class="panel" style="margin-top:14px"><h2>What just happened</h2><div class="feed" id="feed"></div></div>
  </div>
  <div class="side">
    <div class="panel"><h2>The whiteboard</h2><div class="notes" id="notes"></div></div>
  </div>
</main>
<script>
const NOTE_COLORS = {Lena:"#e6dbff",Tom:"#d6e8ff",Rita:"#ffd9d6",Bo:"#ffefc2",Ivy:"#d3f5e6",Ian:"#e0dcff",
  Iris:"#ffe0ee",Tess:"#ffe1cc",Nova:"#e2ffc9",Ada:"#d9ecff",Carl:"#ececec",Maya:"#f6d9e8"};
const people = {};
let lastBoardTime = 0, ds = "", firstLoad = true;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ago = t => { const s = Math.max(0, Date.now()/1000 - t); return s < 60 ? Math.round(s)+"s ago" :
  s < 3600 ? Math.round(s/60)+"m ago" : Math.round(s/3600)+"h ago"; };

function figure(v) {
  // a small round villager: legs, body, arms, head, hair, face, prop
  return `<svg class="fig" viewBox="0 0 100 130" aria-label="${esc(v.name)}">
    <ellipse cx="50" cy="126" rx="26" ry="4" fill="rgba(0,0,0,.18)"/>
    <rect x="36" y="100" width="10" height="24" rx="5" fill="#3a3a48"/><rect x="54" y="100" width="10" height="24" rx="5" fill="#3a3a48"/>
    <rect x="27" y="62" width="46" height="46" rx="18" fill="${v.body}"/>
    <rect x="16" y="66" width="12" height="30" rx="6" fill="${v.body}" transform="rotate(14 22 70)"/>
    <rect x="72" y="66" width="12" height="30" rx="6" fill="${v.body}" transform="rotate(-14 78 70)"/>
    <circle cx="50" cy="40" r="25" fill="#f2c9a0"/>
    <path d="M25 38 Q27 12 50 13 Q74 12 75 38 Q66 25 50 26 Q34 25 25 38Z" fill="${v.hair}"/>
    <circle cx="41" cy="42" r="3.2" fill="#222"/><circle cx="59" cy="42" r="3.2" fill="#222"/>
    <path d="M42 52 Q50 58 58 52" stroke="#7a3d2a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <circle cx="35" cy="50" r="4" fill="#f59a9a" opacity=".55"/><circle cx="65" cy="50" r="4" fill="#f59a9a" opacity=".55"/>
    <text x="80" y="92" font-size="26" text-anchor="middle">${v.prop}</text></svg>`;
}

function ensurePerson(v) {
  if (people[v.name]) return people[v.name];
  const el = document.createElement("div");
  el.className = "villager";
  el.innerHTML = figure(v) + `<div class="tag">${esc(v.name)} · ${esc(v.role)}</div><div class="bubble"></div>`;
  el.style.left = v.x + "%"; el.style.top = v.y + "%";
  if (v.x < 18) el.classList.add("edge-l"); if (v.x > 82) el.classList.add("edge-r");
  $("people").appendChild(el);
  return people[v.name] = {el, home: [v.x, v.y], last: 0, walking: 0};
}

function walkToBoard(p, spot) {
  if (p.walking) return;
  p.walking = 1; p.el.classList.add("walking");
  p.el.style.left = spot[0] + (Math.random()*16-8) + "%"; p.el.style.top = spot[1] + 8 + "%";
  setTimeout(() => { p.el.classList.remove("walking"); p.el.classList.add("show"); }, 1400);
  setTimeout(() => { p.el.classList.add("walking"); p.el.classList.remove("show");
    p.el.style.left = p.home[0] + "%"; p.el.style.top = p.home[1] + "%"; }, 4200);
  setTimeout(() => { p.el.classList.remove("walking"); p.walking = 0; }, 5600);
}

function render(s) {
  if (!s.ready) {
    $("status").textContent = s.message || "waiting";
    for (const v of s.villagers || []) ensurePerson(v);
    return;
  }
  const live = s.now - s.last_event < 20;
  $("live").className = "dot" + (live ? " on" : "");
  $("status").textContent = live ? "working" : (s.last_event ? "resting · last activity " + ago(s.last_event) : "idle");
  $("round").textContent = s.round || "–";
  $("tested").textContent = s.tested;
  const best = s.progress.length ? s.progress[s.progress.length-1].best_so_far : null;
  $("best").textContent = best ?? "–";
  $("banner").textContent = (s.dataset || "no board yet") + (s.round ? "  ·  round " + s.round : "");
  const sel = $("ds"), opts = s.datasets.map(d => `<option${d===s.dataset?" selected":""}>${esc(d)}</option>`).join("");
  if (sel.dataset.v !== opts) { sel.innerHTML = `<option value="">follow the action</option>` + opts; sel.dataset.v = opts; }

  const shown = new Set();
  const speaker = s.villagers.filter(v => s.now - v.time < 9).sort((a, b) => b.time - a.time)[0];
  for (const v of s.villagers) {
    const p = ensurePerson(v); shown.add(v.name);
    p.home = [v.x, v.y];
    if (!p.walking) { p.el.style.left = v.x + "%"; p.el.style.top = v.y + "%"; }
    const fresh = s.now - v.time < 9;
    p.el.classList.toggle("active", fresh && !p.walking);
    p.el.classList.toggle("speaking", !!speaker && speaker.name === v.name && !p.walking);
    if (v.text) p.el.querySelector(".bubble").innerHTML = `<b>${esc(v.full)} · ${ago(v.time)}</b>${esc(v.text.slice(0, 220))}`;
    p.el.title = v.text ? v.full + ": " + v.text : v.full;
  }
  for (const n in people) people[n].el.style.display = shown.has(n) ? "" : "none";

  // whiteboard: newest first on the side, latest few on the board in the scene
  const notes = s.whiteboard.slice().reverse();
  $("notes").innerHTML = notes.length ? notes.map(n => {
    const who = n.author.split(" ")[0], aha = /^AHA|ESSENTIAL/i.test(n.text);
    return `<div class="note${aha?" aha":""}${n.time > lastBoardTime && !firstLoad ? " new":""}" style="background:${NOTE_COLORS[who]||"#f4f1e6"}">
      <span class="who">${esc(who)} · round ${n.round}</span>${esc(n.text)}</div>`; }).join("")
    : `<div class="empty">Nothing on the whiteboard yet.</div>`;
  $("boardtext").innerHTML = notes.slice(0, 6).map(n => `<div class="${/^AHA/.test(n.text)?"aha":""}">• ${esc(n.text)}</div>`).join("");
  for (const n of s.whiteboard) {
    if (n.time > lastBoardTime && !firstLoad) {
      const p = people[n.author.split(" ")[0]];
      if (p) walkToBoard(p, s.whiteboard_spot);
    }
  }
  if (s.whiteboard.length) lastBoardTime = Math.max(lastBoardTime, ...s.whiteboard.map(n => n.time));

  $("leaders").innerHTML = s.leaders.length ? `<table><tr><th>#</th><th>Setup</th><th>By</th><th class="num">Robust</th><th class="num">t</th><th class="num">Trades</th></tr>` +
    s.leaders.map((l, i) => `<tr><td>${i+1}</td><td>${esc(l.name)}</td><td>${esc(l.author.split(" ")[0])}</td>
      <td class="num">${l.robust ?? "–"}</td><td class="num">${l.t ?? "–"}</td><td class="num">${l.trades ?? "–"}</td></tr>`).join("") + `</table>`
    : `<div class="empty">No strategies scored yet.</div>`;
  $("progress").innerHTML = spark(s.progress);
  $("ideas").innerHTML = s.ideas.length ? s.ideas.map(i => `<div>${esc(i.text)}</div>`).join("") : `<div class="empty">No ideas from the library yet. Put PDFs, Pine scripts or transcripts in papers/.</div>`;
  $("feed").innerHTML = s.events.map(e => `<div><b>${esc(e.agent.split(" ")[0])}</b> · ${ago(e.created)} · ${esc(e.text.slice(0, 260))}</div>`).join("") || `<div class="empty">Nothing yet.</div>`;
  firstLoad = false;
}

function spark(points) {
  if (points.length < 2) return `<div class="empty">Needs two rounds.</div>`;
  const W = 320, H = 90, L = 30, R = 8, T = 8, B = 18;
  const ys = points.map(p => p.best_so_far), lo = Math.min(...ys), hi = Math.max(...ys);
  const span = hi - lo || 1, X = i => L + i * (W - L - R) / (points.length - 1), Y = v => T + (hi - v) * (H - T - B) / span;
  const pts = points.map((p, i) => `${X(i).toFixed(1)},${Y(p.best_so_far).toFixed(1)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Best robust score by round">
    <line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="#323745"/>
    <polyline points="${pts}" fill="none" stroke="#f5c451" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="${X(points.length-1)}" cy="${Y(ys[ys.length-1])}" r="4" fill="#f5c451"/>
    <text x="${L-4}" y="${Y(hi)+4}" text-anchor="end">${hi.toFixed(2)}</text>
    <text x="${L-4}" y="${Y(lo)+4}" text-anchor="end">${lo.toFixed(2)}</text>
    <text x="${L}" y="${H-4}">round ${points[0].round}</text>
    <text x="${W-R}" y="${H-4}" text-anchor="end">round ${points[points.length-1].round}</text></svg>`;
}

async function tick() {
  try {
    const r = await fetch("/state" + (ds ? "?ds=" + encodeURIComponent(ds) : ""), {cache: "no-store"});
    render(await r.json());
  } catch (e) { $("status").textContent = "live view server stopped"; $("live").className = "dot"; }
}
$("ds").addEventListener("change", e => { ds = e.target.value; firstLoad = true; tick(); });
tick(); setInterval(tick, 2000);
</script>
</body></html>
"""
