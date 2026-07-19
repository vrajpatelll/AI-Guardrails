"""
Read-only HTTP dashboard over the guardrail decision log.

Reads the same JSONL file `guardrail/logger.py` writes (`logs/decisions.jsonl`
by default, override with GUARDRAIL_LOG_FILE) and renders a single
self-contained HTML page: verdict-rate stat tiles, a per-day ALLOW/REDACT/
BLOCK timeline, a category-hit breakdown, and an audit table of the most
recent BLOCK decisions with their reasons.

No JS build step, no chart library, no external network call — the page is
one inline SVG + vanilla-JS file so it keeps working on a laptop with no
Wi-Fi, matching the "zero network dependency" demo requirement in
development-plan.md Day 6.

Run it with:
    uv run uvicorn guardrail.dashboard:app --reload --port 8050
"""

from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from guardrail import playground
from guardrail.logger import DEFAULT_LOG_FILE

REPO_ROOT = Path(__file__).resolve().parent.parent

CATEGORY_ORDER = ["pii", "secrets", "harmful_content", "prompt_injection"]
CATEGORY_LABELS = {
    "pii": "PII",
    "secrets": "Secrets",
    "harmful_content": "Harmful content",
    "prompt_injection": "Prompt injection",
}
# Fixed categorical order (palette.md slots 1/2/3/4) — never cycled.
CATEGORY_COLORS = {
    "pii": ("#2a78d6", "#3987e5"),
    "secrets": ("#008300", "#008300"),
    "harmful_content": ("#e87ba4", "#d55181"),
    "prompt_injection": ("#eda100", "#c98500"),
}

ACTION_ORDER = ["ALLOW", "REDACT", "BLOCK"]
# Status palette (palette.md) — these are verdict *states*, not identities.
ACTION_COLORS = {
    "ALLOW": ("#0ca30c", "good"),
    "REDACT": ("#fab219", "warning"),
    "BLOCK": ("#d03b3b", "critical"),
}

TIMELINE_DAYS = 14
RECENT_BLOCKS_LIMIT = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _log_path() -> Path:
    raw = os.environ.get("GUARDRAIL_LOG_FILE", DEFAULT_LOG_FILE)
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_timestamp(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def load_records(path: Path) -> list[dict[str, Any]]:
    """Parse the JSONL decision log. Malformed lines are skipped, not fatal."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "guardrail" not in rec:
                continue
            rec["_ts"] = _parse_timestamp(rec.get("timestamp", ""))
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    action_counts = Counter(r["guardrail"].get("action", "ALLOW") for r in records)
    cache_hits = sum(1 for r in records if r["guardrail"].get("cache_hit"))
    latencies = sorted(r["guardrail"].get("latency_ms", 0) for r in records)

    def pct(n: int) -> float:
        return round(100 * n / total, 1) if total else 0.0

    def p95(values: list[int]) -> int:
        if not values:
            return 0
        idx = min(len(values) - 1, int(len(values) * 0.95))
        return values[idx]

    category_counts: Counter[str] = Counter()
    for r in records:
        for cat, detail in r["guardrail"].get("categories", {}).items():
            if detail.get("match"):
                category_counts[cat] += 1

    # Per-day ALLOW/REDACT/BLOCK counts for the last TIMELINE_DAYS days.
    today = datetime.now().date()
    day_keys = [today - timedelta(days=i) for i in range(TIMELINE_DAYS - 1, -1, -1)]
    timeline = {d: Counter() for d in day_keys}
    for r in records:
        ts = r.get("_ts")
        if ts is None:
            continue
        day = ts.date()
        if day in timeline:
            timeline[day][r["guardrail"].get("action", "ALLOW")] += 1

    recent_blocks = sorted(
        (r for r in records if r["guardrail"].get("action") == "BLOCK" and r.get("_ts")),
        key=lambda r: r["_ts"],
        reverse=True,
    )[:RECENT_BLOCKS_LIMIT]

    return {
        "total": total,
        "action_counts": action_counts,
        "action_pct": {a: pct(action_counts.get(a, 0)) for a in ACTION_ORDER},
        "cache_hit_pct": pct(cache_hits),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": p95(latencies),
        "category_counts": category_counts,
        "timeline": timeline,
        "recent_blocks": recent_blocks,
    }


# ---------------------------------------------------------------------------
# SVG mark helpers (rounded data-end, square baseline — marks-and-anatomy.md)
# ---------------------------------------------------------------------------

def _rounded_top_path(x: float, w: float, y: float, h: float, r: float = 4) -> str:
    """Column: rounded top corners, square baseline. Degrades gracefully when h < r."""
    r = min(r, h / 2, w / 2) if h > 0 and w > 0 else 0
    if r <= 0:
        return f"M{x},{y + h} L{x},{y} L{x + w},{y} L{x + w},{y + h} Z"
    return (
        f"M{x},{y + h} L{x},{y + r} "
        f"Q{x},{y} {x + r},{y} "
        f"L{x + w - r},{y} Q{x + w},{y} {x + w},{y + r} "
        f"L{x + w},{y + h} Z"
    )


def _rounded_right_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Bar: rounded right (data) end, square at the baseline (left)."""
    r = min(r, h / 2, w / 2) if h > 0 and w > 0 else 0
    if r <= 0:
        return f"M{x},{y} L{x + w},{y} L{x + w},{y + h} L{x},{y + h} Z"
    return (
        f"M{x},{y} L{x + w - r},{y} "
        f"Q{x + w},{y} {x + w},{y + r} "
        f"L{x + w},{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} "
        f"L{x},{y + h} Z"
    )


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

def render_timeline_chart(timeline: dict[Any, Counter]) -> str:
    days = list(timeline.keys())
    n = len(days)
    chart_w, chart_h = 720, 220
    pad_l, pad_b, pad_t = 40, 24, 12
    plot_w = chart_w - pad_l - 12
    plot_h = chart_h - pad_b - pad_t

    max_total = max((sum(c.values()) for c in timeline.values()), default=0)
    y_max = max_total if max_total > 0 else 1
    # round the axis ceiling up to a clean-ish step
    step = max(1, round(y_max / 4))
    y_max = step * 4 if step * 4 >= y_max else step * 5

    slot_w = plot_w / n
    bar_w = min(24, slot_w * 0.6)

    gridlines = []
    for i in range(5):
        gy = pad_t + plot_h - (plot_h * i / 4)
        val = round(y_max * i / 4)
        gridlines.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{chart_w - 12}" y2="{gy:.1f}" '
            f'class="gridline"/>'
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" class="axis-label" text-anchor="end">{val}</text>'
        )

    bars = []
    for i, day in enumerate(days):
        counts = timeline[day]
        cx = pad_l + slot_w * i + (slot_w - bar_w) / 2
        y_cursor = pad_t + plot_h
        segments_present = [a for a in ACTION_ORDER if counts.get(a, 0) > 0]
        for idx, action in enumerate(ACTION_ORDER):
            v = counts.get(action, 0)
            if v <= 0:
                continue
            seg_h = plot_h * v / y_max
            y_top = y_cursor - seg_h
            is_top_segment = action == (segments_present[-1] if segments_present else None)
            color, _ = ACTION_COLORS[action]
            path = (
                _rounded_top_path(cx, bar_w, y_top, seg_h, r=4)
                if is_top_segment
                else _rounded_top_path(cx, bar_w, y_top, seg_h, r=0)
            )
            label = day.strftime("%b %-d")
            bars.append(
                f'<path d="{path}" fill="{color}" stroke="var(--surface-1)" '
                f'stroke-width="2" paint-order="stroke" class="mark" '
                f'data-tip="{label}: {action} {v}"/>'
            )
            y_cursor = y_top
        # x-axis tick every ~2 days to avoid label collision
        if n <= 14 or i % 2 == 0:
            tx = cx + bar_w / 2
            bars.append(
                f'<text x="{tx:.1f}" y="{chart_h - 4}" class="axis-label" '
                f'text-anchor="middle">{day.strftime("%-m/%-d")}</text>'
            )

    legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{ACTION_COLORS[a][0]}"></span>{a.title()}</span>'
        for a in ACTION_ORDER
    )

    svg = (
        f'<svg viewBox="0 0 {chart_w} {chart_h}" class="chart-svg" role="img" '
        f'aria-label="Decisions per day, by action">'
        f"{''.join(gridlines)}{''.join(bars)}"
        f"</svg>"
    )
    return f'<div class="legend">{legend}</div>{svg}'


def render_category_chart(category_counts: Counter[str]) -> str:
    chart_w, chart_h = 720, 40 + 36 * len(CATEGORY_ORDER)
    pad_l, pad_r = 140, 60
    plot_w = chart_w - pad_l - pad_r
    row_h = 36
    bar_h = 20

    max_v = max((category_counts.get(c, 0) for c in CATEGORY_ORDER), default=0) or 1

    rows = []
    for i, cat in enumerate(CATEGORY_ORDER):
        v = category_counts.get(cat, 0)
        y = 24 + i * row_h
        w = plot_w * v / max_v if max_v else 0
        color, _ = CATEGORY_COLORS[cat]
        path = _rounded_right_path(pad_l, y, max(w, 1 if v else 0), bar_h, r=4)
        rows.append(
            f'<text x="{pad_l - 10}" y="{y + bar_h / 2 + 4:.1f}" class="axis-label" '
            f'text-anchor="end">{CATEGORY_LABELS[cat]}</text>'
        )
        if v > 0:
            rows.append(
                f'<path d="{path}" fill="{color}" stroke="var(--surface-1)" '
                f'stroke-width="2" paint-order="stroke" class="mark" '
                f'data-tip="{CATEGORY_LABELS[cat]}: {v} hit{"s" if v != 1 else ""}"/>'
            )
            rows.append(
                f'<text x="{pad_l + w + 8:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
                f'class="data-label">{v}</text>'
            )
        else:
            rows.append(
                f'<line x1="{pad_l}" y1="{y + bar_h / 2:.1f}" x2="{pad_l + 1}" '
                f'y2="{y + bar_h / 2:.1f}" class="baseline"/>'
                f'<text x="{pad_l + 8}" y="{y + bar_h / 2 + 4:.1f}" class="axis-label">0</text>'
            )

    svg = (
        f'<svg viewBox="0 0 {chart_w} {chart_h}" class="chart-svg" role="img" '
        f'aria-label="Detection category hit counts">'
        f'<line x1="{pad_l}" y1="16" x2="{pad_l}" y2="{chart_h - 8}" class="baseline"/>'
        f"{''.join(rows)}"
        f"</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

def _stat_tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="stat-tile"><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{escape(value)}</div>{sub_html}</div>'
    )


def _blocks_table(recent_blocks: list[dict[str, Any]]) -> str:
    if not recent_blocks:
        return '<p class="empty-note">No BLOCK decisions logged yet.</p>'

    rows = []
    for r in recent_blocks:
        g = r["guardrail"]
        ts = r["_ts"].strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "—"
        hits = g.get("hits", [])
        hit_labels = ", ".join(CATEGORY_LABELS.get(h, h) for h in hits) or "—"
        reason = "—"
        for cat in hits:
            detail = g.get("categories", {}).get(cat, {})
            if detail.get("reason"):
                reason = detail["reason"]
                break
        rows.append(
            "<tr>"
            f"<td>{escape(ts)}</td>"
            f'<td><span class="pill pill-{escape(g.get("direction", ""))}">{escape(g.get("direction", ""))}</span></td>'
            f"<td>{escape(hit_labels)}</td>"
            f"<td>{escape(reason)}</td>"
            f"<td class=\"num\">{g.get('latency_ms', 0)} ms</td>"
            "</tr>"
        )

    return (
        '<div class="table-wrap"><table class="audit-table">'
        "<thead><tr><th>Time</th><th>Direction</th><th>Categories</th>"
        "<th>Reason</th><th>Latency</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


NAV_ITEMS = [("/", "Dashboard"), ("/try", "Try it")]


def _nav_html(active_path: str) -> str:
    links = []
    for href, label in NAV_ITEMS:
        cls = "nav-link nav-link-active" if href == active_path else "nav-link"
        links.append(f'<a class="{cls}" href="{href}">{label}</a>')
    return f'<nav class="nav">{"".join(links)}</nav>'


SHELL_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_meta}
<title>{title}</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --good: #0ca30c;
    --warning: #fab219;
    --critical: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px;
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 28px; }}
  .nav {{ display: flex; gap: 4px; margin-bottom: 20px; }}
  .nav-link {{ font-size: 13px; text-decoration: none; color: var(--text-secondary); padding: 6px 12px; border-radius: 999px; }}
  .nav-link:hover {{ background: var(--gridline); }}
  .nav-link-active {{ background: var(--text-primary); color: var(--surface-1); }}
  .stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 28px; }}
  .stat-tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .stat-label {{ font-size: 12px; color: var(--text-secondary); }}
  .stat-value {{ font-size: 28px; font-weight: 600; margin-top: 4px; }}
  .stat-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 24px; }}
  .card h2 {{ font-size: 14px; margin: 0 0 14px; }}
  .chart-svg {{ width: 100%; height: auto; overflow: visible; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .baseline {{ stroke: var(--baseline); stroke-width: 1; }}
  .axis-label {{ fill: var(--text-muted); font-size: 10px; }}
  .data-label {{ fill: var(--text-primary); font-size: 11px; font-variant-numeric: tabular-nums; }}
  .mark {{ cursor: pointer; }}
  .legend {{ display: flex; gap: 16px; margin-bottom: 10px; font-size: 12px; color: var(--text-secondary); }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .table-wrap {{ overflow-x: auto; }}
  table.audit-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.audit-table th {{ text-align: left; color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; padding: 8px 10px; border-bottom: 1px solid var(--gridline); }}
  table.audit-table td {{ padding: 8px 10px; border-bottom: 1px solid var(--gridline); vertical-align: top; }}
  table.audit-table td.num {{ font-variant-numeric: tabular-nums; color: var(--text-secondary); }}
  .pill {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; background: var(--gridline); color: var(--text-secondary); }}
  .empty-note {{ color: var(--text-muted); font-size: 13px; }}
  #tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1); font-size: 12px; padding: 5px 9px; border-radius: 6px; opacity: 0; transform: translate(-50%, -130%); transition: opacity 0.08s; z-index: 10; white-space: nowrap; }}
  #tooltip.visible {{ opacity: 1; }}
  form {{ display: flex; flex-direction: column; gap: 10px; }}
  textarea {{ width: 100%; resize: vertical; font: inherit; font-size: 14px; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--page); color: var(--text-primary); }}
  textarea:focus {{ outline: 2px solid var(--text-secondary); outline-offset: -1px; }}
  button {{ align-self: flex-start; font: inherit; font-size: 13px; font-weight: 600; padding: 9px 18px; border-radius: 8px; border: none; background: var(--text-primary); color: var(--surface-1); cursor: pointer; }}
  button:hover {{ opacity: 0.88; }}
  .banner {{ border-radius: 10px; padding: 14px 16px; font-size: 13px; margin-bottom: 24px; border: 1px solid var(--border); }}
  .banner-error {{ background: color-mix(in srgb, var(--critical) 12%, var(--surface-1)); }}
  .banner-warn {{ background: color-mix(in srgb, var(--warning) 14%, var(--surface-1)); }}
  .banner code {{ font-size: 12px; }}
  .verdict-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }}
  .verdict-header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
  .action-badge {{ font-size: 11px; font-weight: 700; letter-spacing: 0.03em; color: #fff; padding: 3px 10px; border-radius: 999px; }}
  .verdict-title {{ font-size: 13px; font-weight: 600; }}
  .verdict-meta {{ font-size: 12px; color: var(--text-muted); margin-left: auto; }}
  .chip-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .cat-chip {{ font-size: 12px; padding: 4px 10px; border-radius: 999px; background: var(--gridline); color: var(--text-secondary); }}
  .cat-chip-hit {{ background: color-mix(in srgb, var(--critical) 16%, var(--surface-1)); color: var(--text-primary); }}
  .cat-chip-hit b {{ font-variant-numeric: tabular-nums; }}
  .cat-chip-skip {{ opacity: 0.55; }}
  .sanitized-block {{ margin-top: 12px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; font-size: 13px; background: var(--page); border-radius: 8px; padding: 10px 12px; margin: 4px 0 0; }}
</style>
</head>
<body>
  {nav}
  <h1>{heading}</h1>
  <p class="subtitle">{subtitle}</p>
  {body}
  <div id="tooltip"></div>
  <script>
    const tip = document.getElementById('tooltip');
    document.querySelectorAll('.mark[data-tip]').forEach(el => {{
      el.addEventListener('mousemove', e => {{
        tip.textContent = el.getAttribute('data-tip');
        tip.style.left = e.clientX + 'px';
        tip.style.top = e.clientY + 'px';
        tip.classList.add('visible');
      }});
      el.addEventListener('mouseleave', () => tip.classList.remove('visible'));
    }});
  </script>
</body>
</html>
"""


def _shell(title: str, heading: str, subtitle: str, body: str, active_path: str, auto_refresh: bool = False) -> str:
    return SHELL_TEMPLATE.format(
        title=title,
        refresh_meta='<meta http-equiv="refresh" content="20">' if auto_refresh else "",
        nav=_nav_html(active_path),
        heading=heading,
        subtitle=subtitle,
        body=body,
    )


def render_page(records: list[dict[str, Any]], log_path: Path) -> str:
    agg = aggregate(records)

    stat_tiles = "".join([
        _stat_tile("Total decisions", f"{agg['total']:,}"),
        _stat_tile("Block rate", f"{agg['action_pct']['BLOCK']}%",
                    f"{agg['action_counts'].get('BLOCK', 0)} blocked"),
        _stat_tile("Redact rate", f"{agg['action_pct']['REDACT']}%",
                    f"{agg['action_counts'].get('REDACT', 0)} redacted"),
        _stat_tile("Cache hit rate", f"{agg['cache_hit_pct']}%"),
        _stat_tile("Avg latency", f"{agg['avg_latency_ms']} ms", f"p95 {agg['p95_latency_ms']} ms"),
    ])

    if agg["total"] == 0:
        empty = (
            '<p class="empty-note">No decisions logged yet. Run '
            "<code>uv run python scripts/generate_sample_data.py</code> to seed sample "
            "data, or send traffic through the guardrail middleware.</p>"
        )
        timeline_chart = empty
        category_chart = ""
    else:
        timeline_chart = render_timeline_chart(agg["timeline"])
        category_chart = render_category_chart(agg["category_counts"])

    body = f"""
    <div class="stats-row">
      {stat_tiles}
    </div>

    <div class="card">
      <h2>Decisions per day (last {TIMELINE_DAYS} days)</h2>
      {timeline_chart}
    </div>

    <div class="card">
      <h2>Detection category hits</h2>
      {category_chart}
    </div>

    <div class="card">
      <h2>Recent BLOCK decisions</h2>
      {_blocks_table(agg["recent_blocks"])}
    </div>
    """

    return _shell(
        title="Guardrail Dashboard",
        heading="Guardrail Dashboard",
        subtitle=(
            f"{agg['total']} decisions logged &middot; source: <code>{escape(str(log_path))}</code> "
            "&middot; auto-refreshes every 20s"
        ),
        body=body,
        active_path="/",
        auto_refresh=True,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager, once per process: Tier 1 (Presidio/spaCy) + Tier 2 (Qwen2.5-0.5B)
    # both load here, ~15-20s. Blocking the startup coroutine is fine - no
    # requests are served until this returns.
    playground.init_middleware()
    yield
    playground.shutdown_middleware()


app = FastAPI(title="Guardrail Dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    path = _log_path()
    records = load_records(path)
    return render_page(records, path)


@app.get("/try", response_class=HTMLResponse)
def try_get() -> str:
    return _shell(
        title="Try it — Guardrail",
        heading="Try it",
        subtitle="Send a prompt through the guardrail and, if it's not blocked, on to the LLM.",
        body=playground.render_try_body(),
        active_path="/try",
    )


@app.post("/try", response_class=HTMLResponse)
def try_post(text: str = Form(...)) -> str:
    hits = playground.check_injection_keywords(text)
    if hits:
        body = playground.render_approval_body(text, hits)
    else:
        outcome = playground.run_prompt(text)
        body = playground.render_try_body(submitted_text=text, outcome=outcome)
    return _shell(
        title="Try it — Guardrail",
        heading="Try it",
        subtitle="Send a prompt through the guardrail and, if it's not blocked, on to the LLM.",
        body=body,
        active_path="/try",
    )


@app.post("/try/approve", response_class=HTMLResponse)
def try_approve(text: str = Form(...)) -> str:
    hits = playground.check_injection_keywords(text)
    outcome = playground.run_prompt(text)
    body = playground.render_try_body(
        submitted_text=text,
        outcome=outcome,
        approved_keywords=sorted({h.category.removeprefix("DENY_LIST:") for h in hits}),
    )
    return _shell(
        title="Try it — Guardrail",
        heading="Try it",
        subtitle="Send a prompt through the guardrail and, if it's not blocked, on to the LLM.",
        body=body,
        active_path="/try",
    )


@app.post("/try/reject", response_class=HTMLResponse)
def try_reject(text: str = Form(...)) -> str:
    return _shell(
        title="Try it — Guardrail",
        heading="Try it",
        subtitle="Send a prompt through the guardrail and, if it's not blocked, on to the LLM.",
        body=playground.render_rejected_body(text),
        active_path="/try",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("guardrail.dashboard:app", host="127.0.0.1", port=8050, reload=True)
