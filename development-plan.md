# LLM Guardrail — Development Plan

Scope recap: two-tier (Presidio + Qwen3Guard), policy-as-code, verdict cache,
SDK middleware first (proxy is v2 roadmap talking point). One-week timeline.

---

## 1. Development Plan (day-by-day)

### Day 1 — Foundations
**Goal:** plumbing works end-to-end with zero real detection logic.
- Repo setup, project structure, `.env`/config handling
- Define the request/response schema (from the output-schema doc) as Pydantic models
- Build the SDK middleware shell: wraps an LLM client, intercepts `.chat()`/`.complete()`, currently just passes through
- Write the YAML policy loader (parse `policy.yaml` → config object)
- Stub the verdict object so every layer downstream has something to fill in
- Define the integrator-facing connection config (LLM provider, their API key/endpoint, guardrail auth token) — see Section 4 for the full breakdown of what setup requires
- **Checkpoint:** a call through the wrapped client reaches the real LLM and returns a response, with a no-op guardrail response wrapped around it.

### Day 2 — Tier 1 (deterministic)
- Integrate Presidio: PII recognizers (email, phone, credit card, common ID formats)
- Add secrets/API-key regex patterns (a small curated set is fine — don't chase completeness)
- Build the normalizer (zero-width char / homoglyph stripping) and run it first in the pipeline
- Wire Tier 1 output into the response schema (`filterResults.pii`, `filterResults.secrets`, with `matchedSpan`/`rule`/`confidence`)
- **Checkpoint:** feed 10-15 handcrafted test strings (clean + PII + secrets) and confirm correct detection with spans.

### Day 3 — Verdict cache + policy engine
- Implement the normalize → hash → cache lookup (in-memory dict or `lru_cache` is enough; Redis if time allows)
- **Cache key must include `policy_version`, not just the text hash** — key = `hash(normalized_text + policy_version)`. Implement a policy reload watcher that bumps `policy_version` on every `policy.yaml` change, so old cache entries stop matching automatically the moment the policy changes. Without this, editing the policy mid-demo and re-testing the same prompt will silently return the old, stale verdict.
- Wire cache hit/miss and `policyVersion` into `sanitizationMetadata`
- Implement the policy engine: per-category enabled/disabled, action (block/redact/allow), threshold reading from YAML
- Implement fail-open/fail-closed behavior on a timeout budget (even a simple `asyncio.wait_for` with a fallback works)
- **Checkpoint:** repeated identical input returns a cached verdict in near-zero time; disabling a category in YAML actually skips it; **re-running the same prompt immediately after editing `policy.yaml` returns a freshly computed verdict, not the cached one** — test this explicitly, it's the exact scenario that would embarrass you live in front of judges if it's wrong.

### Day 4 — Tier 2 (Qwen3Guard)
- Get Qwen3Guard (0.6B or 4B — pick based on available compute, see Resources) running locally or via a hosted inference endpoint
- Write the policy-conditioned prompt template (your categories as plain text, per the model's expected format)
- **Wire Tier 1 and Tier 2 to run concurrently, not as a sequential cascade.** Tier 1 (PII/secrets) and Tier 2 (harmful content/jailbreak) check *unrelated* categories — a prompt with zero PII can still be a jailbreak, so Tier 1 being clean must never skip Tier 2. Both detectors run for every category enabled in policy; the verdict combiner waits on all of them and merges results. The only thing that legitimately skips a detector is that category being turned off in `policy.yaml` — never the other tier's result.
- Running them concurrently (rather than one gating the other) is still where your latency win comes from: wall-clock cost is `max(tier1, tier2)`, not the sum, and it's correct instead of just fast.
- Parse Tier 2 output into the same `filterResults` schema (`harmfulContent`, `promptInjectionJailbreak`)
- **Checkpoint:** a jailbreak-style prompt with *no PII in it* still gets flagged by Tier 2 — this is the specific case the old sequential design would have silently let through. A benign prompt with a disabled `harmful_content` category shows `executionState: NOT_EVALUATED` for that category only, for the correct reason (disabled), not because Tier 1 was clean.

### Day 5 — Output-path guardrail + end-to-end wiring
- Apply the same parallel Tier 1 + Tier 2 detection to the LLM *response*, not just the input
- **Build the redaction builder — this was the missing step.** When any category's action is `redact`, the guardrail must construct the actual sanitized string (walk `detections[]`, replace each `matchedSpan` with its redaction token, e.g. an email becomes `[REDACTED_EMAIL]`) and put it in a `sanitizedText` field. On the input side, `sanitizedText` — not the original — is what gets forwarded to the LLM. On the output side, it's what gets returned to the client. If any category says `block`, block wins outright and redaction is skipped for that request.
- Full request → guardrail(input, redact-or-forward) → LLM → guardrail(output, redact-or-return) → client loop working through the SDK wrapper
- Add async structured logging (non-blocking) writing block/allow/redact decisions to a local log store
- **Checkpoint:** run 5-10 end-to-end scenarios (clean, PII-in-input, PII-in-output, jailbreak attempt, toxic content) through the full wrapped client.

### Day 6 — Demo surface + polish
- Build a minimal demo: a small script or lightweight UI showing side-by-side "without guardrail" vs "with guardrail" calls, plus the JSON verdict
- Show the cache-hit speed difference visually (this is your strongest live demo moment)
- Show a log/audit view listing recent blocked requests with their reasons
- Clean up error handling so nothing crashes mid-demo (wrap external calls, sane defaults)
- **Checkpoint:** a stranger could run your demo script and understand the pitch in 2 minutes without narration.

### Day 7 — Buffer, testing, pitch
- Morning: bug bash — run every category through edge cases (empty string, huge input, non-English text, unicode tricks), plus three specific regression checks: (1) a PII-free jailbreak prompt still gets caught by Tier 2, (2) a redact-policy PII input actually forwards sanitized text to the LLM, not the raw text, (3) editing `policy.yaml` and re-running an identical prompt returns a new verdict, not a cached stale one
- Afternoon: finalize slides — lead with policy-as-code + latency story, show the architecture diagram, be upfront about the v2 roadmap (streaming interception, a genuine within-category cost cascade — e.g. a cheap keyword pre-filter before invoking Qwen3Guard specifically for harmful-content, not the cross-category skip logic that was corrected this week — and HTTP proxy mode)
- Ship the setup checklist (Section 4) as your README's "Getting Started" — this doubles as your live proof of "plug and play"
- Rehearse the live demo at least twice, including a rehearsed fallback if live demo breaks (recorded backup or screenshots)

---

## 2. Resources Required

### People (map to however many are on your team — solo is workable given the scoped cuts, but roles are listed separately so you can split if you have teammates)
- **Detection/ML integration** — Presidio setup, Qwen3Guard integration, prompt/policy tuning
- **Backend/platform** — SDK middleware, schema, cache, policy engine, logging
- **Demo/frontend** — the comparison demo, log viewer, slides
If solo: do Days 1-3 as backend-only, Day 4 as ML-only, Days 5-7 blend all three — the day-by-day plan above is already sequenced to work for one person.

### Compute
- **CPU is enough for Tier 1** — Presidio and the normalizer run fine on any laptop.
- **Tier 2 (Qwen3Guard) needs a GPU for reasonable latency**, but you have three viable no-budget-or-low-budget paths for a hackathon:
  - Free-tier GPU notebook (Google Colab / Kaggle) hosting the model, called via a small FastAPI wrapper exposed through a tunnel (e.g. ngrok) — fastest to set up.
  - A hosted inference endpoint if your team has API credits (Hugging Face Inference Endpoints, or a cloud GPU VM if someone has credits).
  - Fall back to the **0.6B quantized variant on CPU** if GPU access is unreliable during the event — slower (hundreds of ms) but keeps the demo self-contained and immune to tunnel/network flakiness, which matters more than raw speed on stage.
- **Local dev machine** — enough RAM to run Presidio + a lightweight backend comfortably; the heavy model can live remotely per above.

### Libraries / frameworks
- `presidio-analyzer`, `presidio-anonymizer` (PII)
- `transformers` (+ `torch`) for Qwen3Guard, or its hosted API if using an inference endpoint
- `fastapi` + `uvicorn` for the middleware/service layer
- `pydantic` for schema validation
- `pyyaml` for policy-as-code parsing
- `redis` (optional) or plain in-memory dict/`lru_cache` for the verdict cache
- `openai`/`anthropic` SDKs (whichever LLM you're wrapping for the demo)
- `pytest` for the Day 7 bug bash, even minimal coverage helps

### Test data
- A small handcrafted set covering: clean prompts, PII-laden prompts, secrets/API keys, known jailbreak phrasing patterns, toxic/harmful content examples, and unicode-obfuscated variants of the above
- Optional: pull a handful of examples from public prompt-injection/jailbreak benchmark datasets if you want more adversarial variety than you can hand-write — just check licensing before including any in a public repo/demo

### Infra/tooling
- GitHub repo (or equivalent) with a simple CI step (lint + test) if time allows — not essential for a hackathon judge, but reduces last-minute breakage
- `.env`-based config so API keys/endpoints aren't hardcoded
- Docker (see deployment section) — even if you don't deploy remotely, a `Dockerfile` makes the judges' "can we run this ourselves" question trivial to answer

---

## 3. Deployment Strategies

### For the hackathon demo itself
Priority is **reliability over scale** — you're demoing to a handful of judges, not serving production traffic.

- **Option A — Local, containerized (recommended default).** `docker-compose` with two services: the guardrail middleware/API and (if not using a remote GPU) the Tier 2 model server. Runs on the demo laptop, zero network dependency, zero risk of a venue Wi-Fi issue killing your demo. This should be your primary path.
- **Option B — Cloud Run / Render / Railway free tier for the API + a remote GPU notebook for Tier 2.** More "real" looking (a public URL judges can hit), but adds two points of failure (deployment platform + tunnel to the GPU notebook). Use this only as a secondary/bonus, not your primary demo path, and rehearse it as thoroughly as Option A.
- **Fallback plan regardless of A or B:** have a recorded demo video or screenshot walkthrough ready. Live demos fail; judges respect a team that has a backup.

### Rollout order within the demo build
1. Get it running locally, unstyled, ugly — correctness first.
2. Containerize once it works, so "how do we run this" is a solved problem before Day 6.
3. Only then invest in the polished demo surface (Day 6) — don't polish UI before the pipeline is trustworthy.

### Beyond the hackathon (the deployment-model question from earlier, now concrete)
This maps back to the self-hosted-vs-SaaS decision flagged at the start of planning — worth one slide, not more, since it's roadmap:
- **Self-hosted library/service** the customer runs in their own infra — the proxy/SDK sits next to their app, data never leaves their network. This is your strongest differentiator vs. Model Armor (which is a Google Cloud call) and fits the "plug and play into existing infra" pitch directly — positions well for enterprise/compliance-sensitive buyers.
- **Hosted SaaS API** you operate — easier for customers to adopt (no infra to manage on their side) but you now own uptime, scaling, and multi-tenant data handling, and you lose the "your data never leaves your infra" selling point.
- For the hackathon pitch: present self-hosted as the primary direction (it's the more defensible story given your x-factor), and mention hosted SaaS as a possible future tier for teams that don't want to self-manage.

### What NOT to build this week
- No Kubernetes, no autoscaling, no multi-region — these are answers to problems you don't have yet and will only eat time you need for Days 4-7.
- No real CI/CD pipeline beyond a basic lint/test step — a working `Dockerfile` and a clean README are what judges and future-you actually need.

---

## 4. Required Input From the Integrator (Setup-Side)

This answers "what does someone have to hand us before this works" —
and keeping this list short is itself part of the plug-and-play pitch. Split
into what's actually mandatory vs. what's optional customization with
sensible defaults, so the demo can show a near-zero-config path first.

### Mandatory (can't run without it)
| Input | Why it's needed | Where it lives |
|---|---|---|
| LLM provider + existing API key/endpoint | The guardrail forwards the request on to the real LLM after checking it — it's a pass-through, not a replacement | Environment variable / secrets file, never hardcoded, never logged |
| Integration mode choice | SDK wrapper (`import` + wrap client) vs. proxy `base_url` swap — determines which few lines of their code change | One-time setup decision, stated in README |
| A guardrail auth token | Since the proxy now sits in the request path, it needs its own credential so it isn't an open relay anyone can call | Generated at setup, passed as a header on every call |

If the integrator accepts every default below, this is the *entire* list — that's the "plug and play" claim made concrete and demoable.

### Optional — policy customization (defaults ship, but most real users will want to tune)
- Which categories are enabled/disabled (`pii`, `secrets`, `harmful_content`, `prompt_injection`, custom)
- Action per category: `block` / `redact` / `allow`
- Confidence threshold per category
- Custom category definitions written as plain text — this is the direct payoff of a policy-conditioned guard model: no code, no fine-tuning, no retraining, just describe a new category in the YAML and it's live
- Org-specific deny-list keywords or regex (internal codenames, custom internal ID formats, etc.)

### Optional — operational/infra choices (defaults ship)
- Latency budget in ms before the fallback path triggers
- Fail-open vs. fail-closed behavior on timeout or detector failure
- Verdict cache TTL
- Where Tier 2 runs — a hosted endpoint you provide by default, or their own self-hosted Qwen3Guard endpoint URL if they want data to never leave their infra (this is the enterprise-facing option flagged in the Deployment section)
- Audit log destination — local file by default, or a webhook/DB connection string if they want centralized logging

### What the integrator should explicitly NOT have to provide
- No training data or fine-tuning inputs — the whole pitch rests on this
- No per-category model management — one guard model, policy text drives categories
- No custom schema mapping — request/response shape mirrors their existing LLM client call, so there's no translation layer for them to write

### How this shows up in the demo
A `config.yaml.example` with every optional field commented out and defaulted, plus a 3-line "quickstart" (install, drop in your LLM key + guardrail token, wrap your client) is worth more to judges than a slide describing plug-and-play — it's the same claim, proven instead of asserted.
