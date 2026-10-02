"""Final reports: a Markdown file and a self-contained HTML page with equity charts."""

import html as _html
import json
import time

import numpy as np

from . import strategy as strat
from .backtest import luck_bar

NO_WINNER = ("No strategy in the top 10 held up in the vault yet. That is a real result: on this "
             "data, nothing the village tried has shown an edge it could keep on unseen prices. "
             "Run more rounds, add more data (longer history, more markets) or new ideas.")
DISCLAIMER = "Backtests are not promises. Paper-trade anything before risking money."


def _intro(trials: int) -> str:
    return (f"The villagers only saw the **train** period. The **test** (vault) period is later "
            f"data nobody optimised on: it is the honest check. The village has backtested "
            f"**{trials}** strategies on this data. With that many tries, some will look good by "
            f"pure luck: the best of {trials} random strategies would reach a train t-stat of "
            f"about {luck_bar(trials)} with no edge at all. Trust only strategies that also held "
            f"up in the vault with a vault t-stat of 2 or more (\"could be luck\" means the vault "
            f"was too short or too noisy to tell).")


def _cells(i: int, r: dict) -> list[str]:
    tr, te = r["train"], r["test"]
    return [str(i), r["name"], r["author"].split(" the ")[0], str(tr.get("robust_sharpe", "-")),
            str(tr.get("positive_periods", "-")), str(tr["sharpe"]), str(tr.get("t_stat", "-")),
            f"{tr['total_return_pct']}%", str(te["sharpe"]), str(te.get("t_stat", "-")),
            f"{te['total_return_pct']}%", str(te["trades"]), f"{te['buy_hold_pct']}%", r["verdict"]]


HEADERS = ["#", "Strategy", "By", "Robust score", "Good periods", "Train Sharpe", "Train t",
           "Train return", "Test Sharpe", "Test t", "Test return", "Test trades", "Test buy&hold",
           "Verdict"]
NUMERIC = range(3, 13)


def markdown(dataset, rows, best, story, trials, brain, fee_bps, analysis="", whiteboard="") -> str:
    lines = [f"# Village report: {dataset}", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · brain: {brain} · fees {fee_bps} bps/side", "",
             _intro(trials), "",
             "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    for i, r in enumerate(rows, 1):
        lines.append("| " + " | ".join(c.replace("|", "/") for c in _cells(i, r)) + " |")
    if not rows:
        lines.append("| - | no valid strategies yet |" + " |" * (len(HEADERS) - 2))
    if story:
        lines += ["", "## The Mayor's summary", "", story.strip()]
    if analysis:
        lines += ["", "## What carries the edge (Ada the Analyst)", "",
                  "Each condition of the leader was removed in turn to see how much the score "
                  "drops. Essential pieces are the real signal; pieces that add nothing are noise.",
                  "", analysis.split(" ", 1)[1] if analysis.startswith("#") else analysis]
    if whiteboard:
        lines += ["", "## The whiteboard (latest notes)", "", whiteboard]
    if best:
        lines += ["", f"## Best candidate: {best['name']} ({best['verdict']})", "",
                  strat.describe(best["spec"]), ""]
        if best.get("markets") and len(best["markets"]) > 1:
            lines += ["| Market | Train Sharpe | Train return | Test Sharpe | Test return |",
                      "|---|---|---|---|---|"]
            for name, m in best["markets"].items():
                lines.append(f"| {name} | {m['train']['sharpe']} | {m['train']['total_return_pct']}% "
                             f"| {m['test']['sharpe']} | {m['test']['total_return_pct']}% |")
            lines.append("")
        lines += ["```json", json.dumps(best["spec"], indent=2), "```"]
    elif rows:
        lines += ["", "## Best candidate", "", NO_WINNER]
    lines += ["", f"_{DISCLAIMER}_"]
    return "\n".join(lines) + "\n"


# ---- HTML ---------------------------------------------------------------------------------

CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#8a8984; --line:#e6e5e0; --vault:rgba(120,118,110,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --bh:#8a8984;
  --good:#1a7f37; --bad:#c2410c; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark; --bg:#1a1a19; --panel:#232322; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#8f8e86; --line:#3a3a37; --vault:rgba(255,255,255,.07);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --bh:#8f8e86; --good:#4ade80; --bad:#fb923c; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#1a1a19; --panel:#232322; --ink:#ffffff;
  --ink2:#c3c2b7; --muted:#8f8e86; --line:#3a3a37; --vault:rgba(255,255,255,.07);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --bh:#8f8e86; --good:#4ade80; --bad:#fb923c; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:36px 0 12px; }
.meta { color:var(--ink2); margin:0 0 16px; } p { max-width:75ch; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--ink2); font-weight:600; } tr:last-child td { border-bottom:0; }
td.num { font-variant-numeric:tabular-nums; text-align:right; } th.num { text-align:right; }
.pill { display:inline-block; padding:1px 8px; border-radius:99px; font-size:12px;
  border:1px solid currentColor; }
.pill.good { color:var(--good); } .pill.bad { color:var(--bad); } .pill.meh { color:var(--ink2); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px;
  margin:16px 0; }
.card h3 { margin:0 0 2px; font-size:16px; } .card .rules { color:var(--ink2); font-size:13px;
  margin:0 0 8px; overflow-wrap:anywhere; }
.legend { display:flex; flex-wrap:wrap; gap:14px; font-size:13px; color:var(--ink2); margin:4px 0; }
.legend i { display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle;
  margin-right:6px; }
.legend i.dash { background:repeating-linear-gradient(90deg,var(--bh) 0 4px,transparent 4px 7px)!important; }
.legend i.vault { height:10px; background:var(--vault)!important; border:1px solid var(--line); }
.chart { position:relative; } svg { display:block; width:100%; height:auto; }
svg text { fill:var(--muted); font-size:11px; }
.tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:6px 8px; font-size:12px; color:var(--ink); display:none;
  box-shadow:0 2px 8px rgba(0,0,0,.12); white-space:nowrap; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px;
  overflow-x:auto; font-size:12.5px; }
.story { white-space:pre-wrap; }
footer { color:var(--muted); font-size:13px; margin-top:40px; }
"""

JS = """
document.querySelectorAll('.chart').forEach(function (box) {
  var d = JSON.parse(box.dataset.series), svg = box.querySelector('svg'),
      tip = box.querySelector('.tip'), cross = svg.querySelector('.cross');
  var W = +svg.getAttribute('data-w'), L = +svg.getAttribute('data-l'), R = +svg.getAttribute('data-r');
  svg.addEventListener('mousemove', function (e) {
    var b = svg.getBoundingClientRect(), x = (e.clientX - b.left) * W / b.width;
    var i = Math.round((x - L) / (W - L - R) * (d.dates.length - 1));
    if (i < 0 || i >= d.dates.length) { tip.style.display = 'none'; return; }
    var px = L + i / (d.dates.length - 1) * (W - L - R);
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.style.display = '';
    var html = '<b>' + d.dates[i] + '</b>' + (i >= d.split ? ' (vault)' : '');
    d.lines.forEach(function (l) { html += '<br><span style="color:' + l.color + '">\\u25A0</span> '
      + l.name + ': ' + l.values[i].toFixed(2) + 'x'; });
    tip.innerHTML = html; tip.style.display = 'block';
    var left = (e.clientX - b.left) + 14;
    if (left + tip.offsetWidth > b.width) left = (e.clientX - b.left) - tip.offsetWidth - 14;
    tip.style.left = left + 'px'; tip.style.top = '8px';
  });
  svg.addEventListener('mouseleave', function () { tip.style.display = 'none'; cross.style.display = 'none'; });
});
"""

COLORS = ["var(--s1)", "var(--s2)", "var(--s3)"]
TIP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a8984"]


def _e(x) -> str:
    return _html.escape(str(x))


def _pill(v: str) -> str:
    cls = "good" if v == "held up" else "meh" if v.startswith("held up") else ("bad" if v in ("failed in vault", "never worked") else "meh")
    return f'<span class="pill {cls}">{_e(v)}</span>'


def _chart(curves: dict, train_frac: float, points: int = 500) -> str:
    """Inline SVG of equity curves (growth of 1), log scale, with the vault period shaded."""
    names = list(curves)
    index = curves[names[0]].index
    same_dates = all(curves[k].index.equals(index) for k in names)
    n = min(len(c) for c in curves.values())
    count = min(points, n)
    take = np.linspace(0, n - 1, count).astype(int)
    split = int(count * train_frac)

    def label(i):
        # Markets with different dates share the chart by position in their own history.
        return str(index[take[i]])[:10] if same_dates else f"{round(100 * i / (count - 1))}%"
    W, H, L, R, T, B = 760, 260, 44, 12, 12, 26
    series = []
    for name in names:
        arr = curves[name].to_numpy(float)
        pos = np.linspace(0, len(arr) - 1, count).astype(int)
        vals = np.maximum(np.nan_to_num(arr[pos], nan=1.0), 1e-6)
        series.append((name, vals))
    lo = min(v.min() for _, v in series)
    hi = max(v.max() for _, v in series)
    lo, hi = np.log(min(lo, 1.0) * 0.95), np.log(max(hi, 1.0) * 1.05)

    def X(i):
        return L + i / max(len(take) - 1, 1) * (W - L - R)

    def Y(v):
        return T + (hi - np.log(v)) / (hi - lo) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" data-w="{W}" data-l="{L}" data-r="{R}" role="img" '
             f'aria-label="Equity curves">']
    vx = X(split)
    parts.append(f'<rect x="{vx:.1f}" y="{T}" width="{W - R - vx:.1f}" height="{H - T - B}" '
                 'style="fill:var(--vault)"/>')
    parts.append(f'<text x="{vx + 6:.1f}" y="{T + 13}">vault (unseen)</text>')
    for tick in _ticks(np.exp(lo), np.exp(hi)):
        y = Y(tick)
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" style="stroke:var(--line)"'
                     + (' stroke-width="1.5"' if tick == 1 else "") + "/>")
        parts.append(f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:g}x</text>')
    for frac in (0, 0.25, 0.5, 0.75, 1):
        i = int(frac * (len(take) - 1))
        anchor = "start" if frac == 0 else ("end" if frac == 1 else "middle")
        parts.append(f'<text x="{X(i):.1f}" y="{H - 8}" text-anchor="{anchor}">{_e(label(i))}</text>')
    tip_lines = []
    for k, (name, vals) in enumerate(series):
        is_bh = name == "buy & hold"
        color = "var(--bh)" if is_bh else COLORS[k % len(COLORS)]
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        dash = ' stroke-dasharray="4 3"' if is_bh else ""
        parts.append(f'<polyline points="{pts}" fill="none" style="stroke:{color}" stroke-width="2" '
                     f'stroke-linejoin="round"{dash}/>')
        tip_lines.append({"name": name, "color": TIP_COLORS[3 if is_bh else k % 3],
                          "values": [round(float(v), 4) for v in vals]})
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{H - B}" '
                 'stroke-width="1" style="stroke:var(--muted);display:none"/>')
    parts.append("</svg>")
    data = {"dates": [label(i) for i in range(count)], "split": split, "lines": tip_lines}
    legend = "".join(
        f'<span><i class="{"dash" if n == "buy & hold" else ""}" style="background:'
        f'{"var(--bh)" if n == "buy & hold" else COLORS[k % 3]}"></i>{_e(n)}</span>'
        for k, n in enumerate(names))
    legend += '<span><i class="vault"></i>vault period</span>'
    return (f'<div class="legend">{legend}</div><div class="chart" data-series="{_e(json.dumps(data))}">'
            + "".join(parts) + '<div class="tip"></div></div>')


def _ticks(lo: float, hi: float) -> list[float]:
    cands = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    ticks = [t for t in cands if lo <= t <= hi]
    return ticks or [1]


def html(dataset, rows, best, story, trials, brain, fee_bps, charts, train_frac,
         analysis="", whiteboard="") -> str:
    intro = _intro(trials).replace("**", "")
    head = "".join(f'<th class="{"num" if i in NUMERIC else ""}">{_e(h)}</th>'
                   for i, h in enumerate(HEADERS))
    body = ""
    for i, r in enumerate(rows, 1):
        cells = _cells(i, r)
        tds = "".join(f'<td class="{"num" if j in NUMERIC else ""}">{_e(c)}</td>'
                      for j, c in enumerate(cells[:-1]))
        body += f"<tr>{tds}<td>{_pill(cells[-1])}</td></tr>"
    if not rows:
        body = f'<tr><td colspan="{len(HEADERS)}">No valid strategies yet.</td></tr>'
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Village Report</title><style>{CSS}</style></head><body><main>",
           f"<h1>Village report: {_e(dataset)}</h1>",
           f"<p class='meta'>{time.strftime('%Y-%m-%d %H:%M')} · brain: {_e(brain)} · "
           f"fees {fee_bps} bps/side</p>", f"<p>{_e(intro)}</p>",
           "<h2>Leaderboard</h2><p class='meta'>Ranked by robust score: the average Sharpe across "
           "slices of the train period (and across markets), minus a penalty when they disagree. "
           "\"Good periods\" counts the slices where the strategy made money.</p>",
           f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
           "</table></div>"]
    if story:
        out.append(f"<h2>The Mayor's summary</h2><div class='card story'>{_e(story.strip())}</div>")
    if analysis:
        text = analysis.split(" ", 1)[1] if analysis.startswith("#") else analysis
        out.append("<h2>What carries the edge</h2><p class='meta'>Ada the Analyst removed each "
                   "condition of the leading strategy in turn. If the score collapses, that piece "
                   "is the real signal. If nothing changes, it is decoration.</p>"
                   f"<div class='card story'>{_e(text)}</div>")
    if whiteboard:
        out.append("<h2>The whiteboard</h2><p class='meta'>Hunches, findings and dead ends the "
                   "team wrote down while working (latest last).</p>"
                   f"<div class='card story'>{_e(whiteboard)}</div>")
    if charts:
        out.append("<h2>Equity curves</h2><p class='meta'>Growth of 1 unit of money (log scale). "
                   "Hover for values. The shaded part is the vault: what happened on data the "
                   "villagers never saw.</p>")
        for r, curves in charts:
            out.append(f"<div class='card'><h3>{_e(r['name'])} {_pill(r['verdict'])}</h3>"
                       f"<p class='rules'>{_e(strat.describe(r['spec']))}</p>"
                       f"{_chart(curves, train_frac)}</div>")
    if best:
        out.append(f"<h2>Best candidate: {_e(best['name'])} {_pill(best['verdict'])}</h2>")
        if best.get("markets") and len(best["markets"]) > 1:
            trs = "".join(
                f"<tr><td>{_e(n)}</td><td class='num'>{m['train']['sharpe']}</td>"
                f"<td class='num'>{m['train']['total_return_pct']}%</td>"
                f"<td class='num'>{m['test']['sharpe']}</td>"
                f"<td class='num'>{m['test']['total_return_pct']}%</td></tr>"
                for n, m in best["markets"].items())
            out.append("<div class='scroll'><table><thead><tr><th>Market</th>"
                       "<th class='num'>Train Sharpe</th><th class='num'>Train return</th>"
                       "<th class='num'>Test Sharpe</th><th class='num'>Test return</th></tr>"
                       f"</thead><tbody>{trs}</tbody></table></div>")
        out.append(f"<pre>{_e(json.dumps(best['spec'], indent=2))}</pre>")
    elif rows:
        out.append(f"<h2>Best candidate</h2><div class='card'>{_e(NO_WINNER)}</div>")
    out.append(f"<footer>{_e(DISCLAIMER)} Not financial advice.</footer></main>"
               f"<script>{JS}</script></body></html>")
    return "\n".join(out)
