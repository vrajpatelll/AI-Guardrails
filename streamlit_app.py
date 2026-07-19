"""
Streamlit demo UI: three tabs — Chat (send a prompt through the guardrail +
LLM), Review (accept/reject prompts a keyword pre-check flagged as possible
prompt injection), and Dashboard (charts over logs/decisions.jsonl).

This file is presentation only. It reuses:
  - guardrail/playground.py for the actual pipeline calls (middleware init,
    keyword pre-check, run_prompt) — the same logic the FastAPI "Try it"
    page uses, so the two frontends never drift out of sync.
  - guardrail/dashboard.py for log loading/aggregation — the same numbers
    the FastAPI dashboard shows.

Run with:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from guardrail import playground
from guardrail.dashboard import (
    ACTION_COLORS,
    ACTION_ORDER,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    TIMELINE_DAYS,
    _log_path,
    aggregate,
    load_records,
)
from guardrail.playground import ScanOutcome
from guardrail.schema import GuardrailResponse

st.set_page_config(page_title="Guardrail Demo", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading guardrail models (Tier 1 + Tier 2) — first run takes ~15-20s…")
def _init_middleware() -> None:
    playground.init_middleware()


_init_middleware()


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def _run_and_record(text: str, approved_keywords: list[str] | None = None) -> None:
    with st.spinner("Running guardrail pipeline…"):
        outcome = playground.run_prompt(text)

    prefix = ""
    if approved_keywords:
        prefix = f"*Human-approved despite matching keyword(s): {', '.join(approved_keywords)}.*\n\n"

    if outcome.combined_action == "ERROR":
        content = (
            prefix
            + "⚠️ The guardrail scan did not block this request, but the LLM gateway "
            f"call itself failed: `{outcome.llm_error}`. That's usually a network/"
            "reachability issue, not a guardrail bug."
        )
    elif outcome.blocked:
        content = prefix + "🚫 **Blocked** — this request was not sent to the LLM."
    elif outcome.llm_text:
        content = prefix + outcome.llm_text
    else:
        content = (
            prefix
            + "_Request was not blocked, but no response text could be read from the "
            "LLM backend._"
        )

    st.session_state.messages.append({"role": "assistant", "content": content, "outcome": outcome})


def _render_one_verdict(title: str, verdict: GuardrailResponse) -> None:
    sr = verdict.sanitization_result
    action = sr.action.value
    icon = {"ALLOW": "✅", "REDACT": "🟡", "BLOCK": "🔴"}.get(action, "⚪")
    meta = sr.sanitization_metadata
    st.markdown(
        f"**{title}:** {icon} `{action}`  ·  {sr.latency_ms}ms  ·  "
        f"{'cache HIT' if meta.cache_hit else 'cache MISS'}  ·  policy v{meta.policy_version}"
    )

    chips = []
    reasons = []
    for cat_name, cr in sr.filter_results.items():
        is_output = cat_name.startswith("output_")
        label = CATEGORY_LABELS.get(cat_name.removeprefix("output_"), cat_name)
        if is_output:
            label += " (out)"
        if cr.match_state.value == "MATCH_FOUND":
            top = cr.detections[0] if cr.detections else None
            detail = f" {top.confidence:.2f}" if top else ""
            chips.append(f"🔴 **{label}**{detail}")
            if top and cr.tier == 2 and top.rule:
                reasons.append(f"- **{label}:** {top.rule}")
        elif cr.execution_state.value == "NOT_EVALUATED":
            chips.append(f"⚪ {label} (skipped)")
        else:
            chips.append(f"🟢 {label}")
    st.markdown("&nbsp;&nbsp;".join(chips))
    if reasons:
        st.caption("\n".join(reasons))
    if sr.sanitized_text:
        st.markdown("**Sanitized text:**")
        st.code(sr.sanitized_text, language=None)


def _render_verdict_details(outcome: ScanOutcome) -> None:
    if outcome.combined_action == "ERROR":
        return
    with st.expander(f"Guardrail verdict — {outcome.combined_action} · {outcome.latency_ms:.0f}ms"):
        if outcome.input_verdict is not None:
            _render_one_verdict("Blocked on input" if outcome.blocked else "Input scan", outcome.input_verdict)
        if outcome.output_verdict is not None:
            _render_one_verdict("Output scan", outcome.output_verdict)


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_queue", [])  # list of {id, text, keywords, msg_index}
    st.session_state.setdefault("next_pending_id", 0)


def render_chat_tab() -> None:
    st.caption(
        "Runs the real guardrail pipeline (Tier 1 + Tier 2, verdict cache, hot-reloaded "
        "policy.yaml) against your message, then forwards it to the LLM gateway if it "
        "isn't blocked. Messages matching known prompt-injection phrasing are held for "
        "human review — decide them in the 🔍 Review tab."
    )

    if playground.setup_error():
        st.error(
            f"Guardrail middleware failed to start: `{playground.setup_error()}`. Check "
            "your `.env` (`LLM_GATEWAY_API_KEY`, `GUARDRAIL_TOKEN`) and restart the app."
        )
        return

    _init_session_state()

    if st.session_state.pending_queue:
        st.info(
            f"⏳ {len(st.session_state.pending_queue)} message(s) awaiting human review "
            "— see the 🔍 Review tab."
        )

    pending_indices = {p["msg_index"] for p in st.session_state.pending_queue}
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if i in pending_indices:
                st.caption("⏳ Awaiting human review — see the Review tab.")
            if msg.get("outcome") is not None:
                _render_verdict_details(msg["outcome"])

    prompt = st.chat_input("Type a message…")
    if prompt:
        msg_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        hits = playground.check_injection_keywords(prompt)
        if hits:
            keywords = sorted({h.category.removeprefix("DENY_LIST:") for h in hits})
            pending_id = st.session_state.next_pending_id
            st.session_state.next_pending_id += 1
            st.session_state.pending_queue.append({
                "id": pending_id,
                "text": prompt,
                "keywords": keywords,
                "msg_index": msg_index,
            })
        else:
            _run_and_record(prompt)
        st.rerun()


# ---------------------------------------------------------------------------
# Review tab
# ---------------------------------------------------------------------------

def render_review_tab() -> None:
    st.caption(
        "Messages that matched a known prompt-injection phrase (checked before the "
        "guardrail model or the LLM ever see them) wait here until a human explicitly "
        "approves or rejects them. Approving runs the full guardrail pipeline and sends "
        "it on; rejecting discards it. Either way the result appears back in the Chat tab."
    )

    _init_session_state()

    queue = st.session_state.pending_queue
    if not queue:
        st.success("Nothing awaiting review.")
        return

    for item in list(queue):
        with st.container(border=True):
            st.markdown(f"**Matched keyword(s):** `{', '.join(item['keywords'])}`")
            st.code(item["text"], language=None)
            col1, col2 = st.columns(2)
            if col1.button("✅ Approve & send", key=f"approve_{item['id']}", width="stretch"):
                _run_and_record(item["text"], approved_keywords=item["keywords"])
                st.session_state.pending_queue = [
                    p for p in st.session_state.pending_queue if p["id"] != item["id"]
                ]
                st.rerun()
            if col2.button("🚫 Reject", key=f"reject_{item['id']}", width="stretch"):
                st.session_state.messages.append(
                    {"role": "assistant", "content": "_Rejected — not sent to the LLM._"}
                )
                st.session_state.pending_queue = [
                    p for p in st.session_state.pending_queue if p["id"] != item["id"]
                ]
                st.rerun()


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

def render_dashboard_tab() -> None:
    top = st.columns([1, 5])
    if top[0].button("🔄 Refresh"):
        st.rerun()

    path = _log_path()
    records = load_records(path)
    agg = aggregate(records)
    top[1].caption(f"{agg['total']} decisions logged · source: `{path}`")

    if agg["total"] == 0:
        st.info(
            "No decisions logged yet. Run `uv run python scripts/generate_sample_data.py` "
            "to seed sample data, or send a few messages in the Chat tab."
        )
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total decisions", f"{agg['total']:,}")
    c2.metric("Block rate", f"{agg['action_pct']['BLOCK']}%", f"{agg['action_counts'].get('BLOCK', 0)} blocked")
    c3.metric("Redact rate", f"{agg['action_pct']['REDACT']}%", f"{agg['action_counts'].get('REDACT', 0)} redacted")
    c4.metric("Cache hit rate", f"{agg['cache_hit_pct']}%")
    c5.metric("Avg latency", f"{agg['avg_latency_ms']} ms", f"p95 {agg['p95_latency_ms']} ms")

    st.subheader(f"Decisions per day (last {TIMELINE_DAYS} days)")
    timeline_df = pd.DataFrame(
        {
            day.strftime("%Y-%m-%d"): {a: counts.get(a, 0) for a in ACTION_ORDER}
            for day, counts in agg["timeline"].items()
        }
    ).T
    timeline_df.index.name = "Date"
    st.bar_chart(timeline_df, color=[ACTION_COLORS[a][0] for a in ACTION_ORDER])

    st.subheader("Detection category hits")
    cat_df = pd.DataFrame(
        {
            "Category": [CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
            "Hits": [agg["category_counts"].get(c, 0) for c in CATEGORY_ORDER],
        }
    ).set_index("Category")
    st.bar_chart(cat_df, horizontal=True)

    st.subheader("Recent BLOCK decisions")
    if not agg["recent_blocks"]:
        st.caption("No BLOCK decisions logged yet.")
        return

    rows = []
    for r in agg["recent_blocks"]:
        g = r["guardrail"]
        hits = g.get("hits", [])
        reason = "—"
        for cat in hits:
            detail = g.get("categories", {}).get(cat, {})
            if detail.get("reason"):
                reason = detail["reason"]
                break
        rows.append({
            "Time": r["_ts"].strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "—",
            "Direction": g.get("direction", ""),
            "Categories": ", ".join(CATEGORY_LABELS.get(h, h) for h in hits) or "—",
            "Reason": reason,
            "Latency (ms)": g.get("latency_ms", 0),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------------

@st.dialog("About this project", width="large")
def _show_about_dialog() -> None:
    st.markdown(
        """
### What this is
A **two-tier LLM guardrail**: an SDK middleware that sits between your app and an
LLM gateway, scanning every request *and* response before it goes anywhere.
Plus a **separate guardrail for agentic tool calls** — the arguments an LLM
wants to pass to `execute_db_query`, `fetch_external_url`, etc. — since that's
a different attack surface (structured JSON, not prose).

### How a message is scanned
Two tiers run **concurrently**, not as a sequential cascade:
- 🔵 **Tier 1 (deterministic)** — Presidio for PII (email, phone, SSN, credit
  card, IP…), a regex catalog for secrets (AWS/GitHub/Slack/API keys, PEM
  private keys), and custom `deny_patterns` from `policy.yaml` for org-specific
  terms. Confidence 1.0, no model involved.
- 🟣 **Tier 2 (semantic)** — a local Qwen2.5-0.5B model judges harmful content.
  Prompt injection is **not** trusted to this small model alone — a
  deterministic keyword/regex gate checks first and routes a match to a human
  for **Approve / Reject** (see the Review tab) before Tier 2 or the LLM ever
  see it.
- 🔓 **Base64 detection** — before scanning, the message is checked for a
  base64-encoded payload; if found, it's decoded and the decoded text is
  scanned for PII or harmful content too, closing a guardrail bypass where
  someone base64-encodes a secret or a jailbreak prompt to slip it past
  plain-text detectors.

Verdicts combine to the strictest outcome: **BLOCK > REDACT > ALLOW**. Every
decision is cached (`hash(text + policy_version)`), and editing `policy.yaml`
hot-reloads instantly — no restart, and the cache invalidates automatically
since the policy version is part of the cache key.

### The tabs on this page
- **💬 Chat** — type a message, it runs through the real pipeline (Tier 1 +
  Tier 2, cache, hot-reloaded policy) and on to the LLM if it isn't blocked.
- **🔍 Review** — messages the keyword gate flagged as possible prompt
  injection, waiting on a human decision.
- **📊 Dashboard** — stat tiles, a per-day ALLOW/REDACT/BLOCK timeline, a
  category-hit breakdown, and an audit table of recent BLOCK decisions —
  reading the same `logs/decisions.jsonl` every scan writes to.

### Also in this repo
- `guardrail/tools/` — the agentic tool-calling guardrail (SQL injection,
  SSRF/domain-allowlist, shell-injection, PII-in-parameters), with its own
  ALLOW / BLOCK / HUMAN_APPROVAL routing.
- `examples/full_demo.py` — a guided CLI walkthrough of every feature above.
- `README.md` — full setup and architecture docs.
        """
    )
    if st.button("Close"):
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

title_col, help_col = st.columns([20, 1])
with title_col:
    st.title("🛡️ AI Guardrails — Demo")
with help_col:
    st.write("")  # vertical alignment nudge next to the title
    if st.button("❓", help="What is this project?"):
        _show_about_dialog()

chat_tab, review_tab, dashboard_tab = st.tabs(["💬 Chat", "🔍 Review", "📊 Dashboard"])
with chat_tab:
    render_chat_tab()
with review_tab:
    render_review_tab()
with dashboard_tab:
    render_dashboard_tab()
