# Plan: Gemini Pass 2 backend (branch experiment, merge-if-good)

## 0. Goal & strategy

Add a **Gemini backend for Cuts v3 Pass 2 only**, behind a config flag, on a
throwaway branch off `main`. Pass 1 and everything else stay on Claude. If a
head-to-head A/B beats or matches the Sonnet baseline on quality (at a big cost
win), merge the branch — landing the capability **off by default** so `main`
behavior is unchanged until we flip an env flag.

Branch:
```
git checkout main && git pull
git checkout -b gemini-pass2
```

### Why this is NOT a drop-in (measured 2026-07-13)

A drop-in swap (route `ic.complete("pass2", ...)` to Flash-Lite) fails two ways:

| Mode | Result | Root cause |
|---|---|---|
| Structured (`response_schema`) | **0 cuts on every batch**, ~4 output tokens/call | model satisfies the schema trivially with `cuts: []` (the field isn't `required`, has no `minItems`); Claude-flavored prompt doesn't force engagement |
| Free JSON (no schema) | emits rich cuts but **drops required fields** (`label`) even after a re-ask | no enforced contract |
| Any structured attempt | Gemini **rejects our schema** outright | `Tuple[...]` fields → JSON `prefixItems`, which Gemini's converter refuses |

So the branch has to solve four concrete things: (1) a **Gemini-shaped schema**,
(2) **forced engagement** (`cuts` required + `minItems:1` + prompt), (3) a
**Gemini-native structured call path** with re-ask parity, (4) **context
caching** so the cost win is real. Correctness first (P1–P3); cost second (P4).

### Non-goals
- Do **not** touch Pass 1 (stays `claude-sonnet-5`).
- Do **not** touch the editing brain / arranger (already provider-neutral via `get_llm()`).
- Do **not** remove or alter the Anthropic ingest path — it stays the default.
- Do **not** change the `Completion` contract or any `pass2.py` / `ingest.py`
  call sites beyond a gated prompt suffix.

---

## 1. Current state (grounding — read these first)

- **`backend/app/services/llm/client.py`** — `complete(stage, system, blocks, schema, *, extra_blocks, cache, max_tokens, extra_check) -> Completion`.
  Anthropic-only: `_sdk_client()`, `_block_to_anthropic`, streamed tool-forced
  call (schema wrapped as a single TOOL `emit_result`, `tool_choice` forced),
  one re-ask on schema/semantic failure, else `IngestFailure`. Returns
  `Completion(data: dict, usage: dict, attempts: int)`.
  - Stage→model: `_STAGE_MODEL_ATTR = {"pass1": "ingest_pass1_model", "pass2": "ingest_pass2_model"}`, resolved by `_model_for(stage)`.
  - Reusable helpers to keep: `_validate` (unwrap/unstringify leniency),
    `_unwrap_single_key`, `_unstringify_json_fields`, `Completion`, `IngestFailure`.
- **`backend/app/services/llm/gemini_client.py`** — a neutral `LLMClient` for the
  brain. Has block→`types.Part` translation (`_parts_for_content`) worth reusing,
  and `_sdk()` for the `google-genai` client. It is **not** structured/schema-
  enforced, so it can't be used directly for ingest, but its translation code is
  a good starting point.
- **`backend/app/services/l3/pass2.py`**
  - `run_pass2_batch(file_rows, pass1_output, batch_frames, images_b64)` builds
    `cached_blocks = build_pass1_blocks(file_rows) + [text_block(render_pass1_output(pass1_output, batch_refs))]`
    and `image_blocks`, then calls
    `ic.complete("pass2", _SYSTEM, cached_blocks, Pass2BatchOutput, extra_blocks=image_blocks, max_tokens=32000, extra_check=lambda o: _pass2_semantic_checks(o, pass1_output, lattices, batch_refs))`.
  - `_SYSTEM` — the Pass 2 brief (already mostly provider-neutral: "reference
    every cut by source_ref verbatim", "do NOT echo word_span", etc.).
  - `Pass2BatchOutput { cuts: List[CutJudgment] }`; `CutJudgment` has the Tuple
    fields that break Gemini: `word_span: Tuple[int,int]|None`,
    `framing.subject_box/crop_*: Tuple[4 floats]|None`,
    `caption_zones: List[Tuple[4 floats]]`.
  - Locators are **code-derived** post-call (`backfill_locators`), so the model
    only needs correct `source_ref` — not `word_span`/`atom_ids`.
- **`backend/app/services/l3/ingest.py`** — builds batches, runs
  `pass2.run_pass2_batch` concurrently, `store.accumulate_pass2_usage`,
  `backfill_locators`, `to_pass2_cuts`, `apply_junk_suspects`, `apply_take_groups`.
- **`backend/app/config.py`** — has `gemini_api_key`, `gemini_model` (default
  `gemini-2.5-pro`), `ingest_pass1_model`, `ingest_pass2_model`
  (both `claude-sonnet-5`). `google-genai` SDK is already installed and imports.

Baseline for the A/B (podcast trail 3, project `57b689b3-39db-4cb4-8385-9e87a996fe9a`):
run `df99a8c5-b054-4907-a696-74589ea36ead` — **94 cuts** (speech 66 / video 28),
speech on_camera 38/28, outlook beats 33 (28 clean 1-on/1-off), channel
{said 66, done 4, shown 24}, all 94 cuts carry subject_box + known shot_size +
people. Use these as the quality bar.

---

## 2. Phase 1 — Gemini structured backend (new module)

Create **`backend/app/services/llm/ingest_gemini.py`**. It mirrors `complete()`'s
contract but talks Gemini. Keep it a *sibling* of `client.py` (do not modify the
Anthropic code path).

### 2.1 Schema sanitizer — `gemini_schema(schema: type[BaseModel]) -> dict`

Start from `schema.model_json_schema()` and rewrite recursively:

- **Tuples** (`prefixItems`): replace the node with
  `{"type": "array", "items": {"type": "number"}, "minItems": N, "maxItems": N}`
  where `N = len(prefixItems)`. Keeping `minItems=maxItems=N` preserves the
  4-length `subject_box`/crop contract (pydantic coerces list→tuple on validate;
  a wrong length then fails validation → re-ask, which is correct behavior).
- **Optionals** (`anyOf` containing `{"type":"null"}`): collapse to the non-null
  branch and set `"nullable": true`.
- **Strip keys Gemini's converter rejects**: `title`, `default`,
  `additionalProperties`, `$schema`, `const`, `format` (unless a supported one).
  Keep: `type`, `properties`, `items`, `required`, `enum`, `nullable`,
  `description`, `$ref`, `$defs`, `minItems`, `maxItems`.
  (Confirmed in the probe: Gemini **does** resolve `$ref`/`$defs`, so leave them.)
- **Force engagement (the fix for empty output):**
  - Top-level: add `"required": ["cuts"]` and set `cuts.minItems = 1`.
    (Every Pass 2 batch is shown ≥1 ref, so a non-empty result is always
    correct; this is what stops the trivial `cuts: []`.)
  - On `CutJudgment`, ensure `label` and `summary` are in `required` (they are in
    pydantic; confirm they survive the rewrite) — and consider adding
    `"minLength": 1` to both so an empty string is rejected.
- **Enum tightening (optional but recommended):** emit `enum` for `shot_size`
  (from `SHOT_SIZES`) and `channel` (`said`/`done`/`shown`) so structured
  decoding can't drift; the aliases in `pass2.py` already fold synonyms if not.

Unit-test this sanitizer against `Pass2BatchOutput` (P6) — assert no `prefixItems`
remain, `cuts` is required with `minItems:1`, and the dict is accepted by
`google.genai.types.GenerateContentConfig(response_schema=...)` without raising.

### 2.2 The call — `complete_gemini(...) -> Completion`

Signature identical to `complete()` minus `cache` semantics:
```python
def complete_gemini(system, blocks, schema, *, extra_blocks=None,
                    max_tokens=32000, extra_check=None,
                    model=None, thinking=None, cached_content=None) -> Completion: ...
```
Steps:
1. `client, types = _sdk()` (reuse `gemini_client._sdk`).
2. Build `contents = [types.Content(role="user", parts=_parts(blocks + (extra_blocks or [])))]`
   where `_parts` is the text/image translation from `gemini_client._parts_for_content`
   (text → `Part(text=...)`, image → `Part.from_bytes(...)`).
3. `config = types.GenerateContentConfig(`
   - `system_instruction=system`,
   - `max_output_tokens=max_tokens`,
   - `temperature=0` (determinism / reproducible A/B),
   - `response_mime_type="application/json"`,
   - `response_schema=gemini_schema(schema)`,
   - `thinking_config=types.ThinkingConfig(...)` — **set a real budget** (e.g.
     `thinking_level="low"` or a `thinking_budget` of a few thousand tokens for
     3.x); the near-zero output in the drop-in test correlated with no thinking.
     Make the level a config knob (`ingest_pass2_thinking`).
   - `cached_content=cached_content` when P4 caching is on.
   `)`
4. `resp = client.models.generate_content(model=model or settings.ingest_pass2_model, contents=contents, config=config)`
5. Parse: `raw = json.loads(resp.text)`. **Normalize** a bare list →
   `{"cuts": raw}` (Gemini sometimes drops the wrapper key).
6. Validate with the existing leniency: `parsed, err = client._validate(schema, raw)`
   then run `extra_check(parsed)` exactly like `complete()`.
7. **One re-ask** on failure: append a `types.Content(role="user", parts=[Part(text=f"Your previous JSON failed validation: {err}\nRe-emit ONLY a valid JSON object {{\"cuts\": [...]}} for the schema; include every required field, one cut per shown source_ref.")])` and re-generate. Second failure → `raise IngestFailure("pass2", ...)`.
8. **Usage mapping** → the same keys `store.accumulate_pass2_usage` expects:
   - `input_tokens = usage_metadata.prompt_token_count`
   - `output_tokens = candidates_token_count + (thoughts_token_count or 0)`
   - `cache_read_input_tokens = usage_metadata.cached_content_token_count or 0`
   - `cache_creation_input_tokens = 0`
9. Return `Completion(data=parsed.model_dump(), usage=..., attempts=1|2)`.

**Fallback path (keep in back pocket, don't build unless P5 shows sparse output):**
Gemini function-calling with `tools=[FunctionDeclaration(name="emit_result", parameters=gemini_schema)]` + `tool_config=FunctionCallingConfig(mode="ANY", allowed_function_names=["emit_result"])`, reading `function_call.args`. This most closely mimics Claude's forced tool-use. Note it as the escalation if `response_schema` still under-produces.

---

## 3. Phase 2 — provider routing (single switch, no call-site churn)

In **`config.py`** add:
```python
# Cuts v3 Pass 2 backend. "anthropic" (default, unchanged) | "gemini".
ingest_pass2_provider: str = "anthropic"
# Gemini thinking effort for Pass 2 ("low" | "medium" | "high" or a budget int).
ingest_pass2_thinking: str = "low"
```
Keep `ingest_pass2_model` as the model id for whichever provider is selected
(set it to e.g. `gemini-3.1-flash-lite` in the branch's `.env` when testing).

In **`client.py` `complete()`**, add a routing guard at the top (only for the
pass2 stage; everything else untouched):
```python
if stage == "pass2" and get_settings().ingest_pass2_provider == "gemini":
    from app.services.llm.ingest_gemini import complete_gemini
    return complete_gemini(system, blocks, schema, extra_blocks=extra_blocks,
                           max_tokens=max_tokens, extra_check=extra_check,
                           thinking=get_settings().ingest_pass2_thinking,
                           cached_content=_pass2_cache_handle())  # P4; None until then
```
`run_pass2_batch` / `ingest.py` are unchanged — they still call `ic.complete("pass2", ...)`.

---

## 4. Phase 3 — prompt adaptation (gated, so Claude quality is untouched)

`_SYSTEM` is mostly neutral, but the Anthropic path relied on *forced tool-use*
to guarantee a non-empty, fully-populated response. Gemini needs that stated
explicitly. **Do not edit `_SYSTEM` in place** (that would perturb the proven
Claude path). Instead, in `pass2.run_pass2_batch`, when the pass2 provider is
`gemini`, append a reinforcement suffix:

```python
_GEMINI_REINFORCE = (
    "\n\nOUTPUT CONTRACT (STRICT): Return a JSON object {\"cuts\": [ ... ]}. "
    "Emit EXACTLY ONE cut object per source_ref you were shown — never an empty "
    "list, never skip a ref. Every cut MUST include a non-empty label and "
    "summary, plus framing (subject_box + shot_size) and look. Fill every "
    "required field from the pixels; use the 'unsure' category rather than "
    "omitting a field."
)
system = _SYSTEM + (_GEMINI_REINFORCE if settings.ingest_pass2_provider == "gemini" else "")
```

Audit `_SYSTEM` for any residual tool-centric phrasing (e.g. "call the tool")
and, if present, neutralize it in the suffix rather than the base string.

---

## 5. Phase 4 — Gemini context caching (cost realization; do AFTER P1–P3 pass)

Without caching, each batch re-sends the ~200–270K-token shared prefix uncached,
erasing most of the price advantage. On Anthropic this is `cache_control`
ephemeral; on Gemini use **explicit `CachedContent`**.

- The **stable, project-wide** part of the prefix is `build_pass1_blocks(file_rows)`
  (transcripts + atom tables — the bulk). The `render_pass1_output(..., batch_refs)`
  is **trimmed per batch**, so it is *not* identical across batches and must stay
  uncached. Plan:
  1. Once per ingest run, create a `CachedContent` from `system + build_pass1_blocks(all_file_rows)` with a TTL covering the run (e.g. 15 min).
  2. Per batch, pass `cached_content=<name>` and put only the trimmed pass-1
     render + this batch's images in `contents`.
- Respect Gemini's **minimum cacheable token** threshold (model-dependent, ~1–4K);
  our prefix is far above it.
- A small per-run cache manager (create → reuse handle across batches → delete on
  finish) keeps this clean. Wire the handle through `run_pass2_batch` (new
  optional arg) or a contextvar set by `ingest.py` around the batch loop.
- Map `cached_content_token_count` into `cache_read_input_tokens` for honest cost
  accounting.

Mark P4 **optional for the first quality gate** — measure raw cost first, then
add caching and re-measure.

---

## 6. Phase 5 — A/B verification harness (committed on the branch)

Create **`backend/scripts/pass2_provider_ab.py`** (keep it — it's the merge gate,
not a throwaway). It should:
- Take a project id and run the ingest twice: once with `ingest_pass2_provider=anthropic`,
  once with `=gemini` (toggle via `get_settings()` monkeypatch or env), tagging
  each resulting `ingest_run_id`.
- Accumulate the **Gemini Pass 2 token usage** (in/out/cached) inside the run and
  print cost at both Gemini and Sonnet rates.
- Compare `cut_records` of the two runs on:
  - total cuts and by-kind split
  - `channel` distribution
  - speech `on_camera` True/False split
  - outlook beats + clean 1-on/1-off ratio
  - % cuts with `subject_box`, with known `shot_size` (≠ "unsure"), with `characteristics`
  - take/outlook role counts
  - schema-fallback count, re-ask count, IngestFailure count
- **Clean up after itself**: delete the throwaway ingest runs it creates (and
  their `cut_records`), and verify the project's latest `ready` run is restored,
  so the frontend/brain never picks up an experimental run. (The drop-in test
  hit exactly this: 0-cut "ready" runs can hijack "latest".)
- Run it on **≥3 projects of different types** (podcast/outlook, a reel, a
  b-roll/food reel), not just podcast trail 3.

### Merge gate (all must hold on every test project)
- 0 `IngestFailure`; schema-fallback count = 0 (sanitized schema accepted); re-ask rate low (≤ ~10% of batches).
- Total cuts within **±10%** of the Sonnet run; by-kind split comparable.
- `subject_box` + known `shot_size` on essentially all cuts (parity with Sonnet's 94/94).
- `channel` and `on_camera`/clean-beat ratios comparable (podcast baseline: 28/33 clean).
- `characteristics`/people populated (identity map depends on it).
- Realized cost clearly below Sonnet (target ≥ ~5× cheaper with caching on).

If gates fail on sparse output → raise `ingest_pass2_thinking`, then try the
function-calling fallback (SS2.2) before giving up.

---

## 7. Phase 6 — merge & rollout

- If gates pass: merge `gemini-pass2` into `main` with `ingest_pass2_provider`
  **defaulting to `anthropic`** (capability lands dark). Flip to `gemini` via env
  per-environment once confident.
- Rollback is a single env flip back to `anthropic`.
- Follow-up (separate task): consider moving Pass 1 similarly only if Pass 2
  proves out and a strong-reasoner Gemini (2.5 Pro / 3.x Pro) A/Bs favorably —
  out of scope here.

---

## 8. Files touched (summary)

| File | Change |
|---|---|
| `backend/app/config.py` | add `ingest_pass2_provider`, `ingest_pass2_thinking` |
| `backend/app/services/llm/ingest_gemini.py` | **NEW** — `gemini_schema()`, `complete_gemini()`, usage mapping, one-re-ask parity, P4 cache manager |
| `backend/app/services/llm/client.py` | add pass2 provider-routing guard at top of `complete()`; reuse `_validate`/`Completion`/`IngestFailure` (no other change) |
| `backend/app/services/l3/pass2.py` | gated `_GEMINI_REINFORCE` suffix in `run_pass2_batch`; (P4) thread a cache handle |
| `backend/app/services/l3/ingest.py` | (P4 only) create/reuse/delete the per-run `CachedContent` around the batch loop |
| `backend/scripts/pass2_provider_ab.py` | **NEW** — committed A/B harness + self-cleanup |
| `backend/requirements*.txt` | pin a `google-genai` version supporting dict `response_schema` + caching + `thinking_config` |
| unit tests | sanitizer test (no `prefixItems`, `cuts` required+minItems, Gemini accepts it); usage-mapping test |

## 9. Risks & mitigations

- **Still-sparse output under `response_schema`** → `cuts` required + `minItems:1`
  + prompt reinforcement + real thinking budget; escalate to function-calling
  `mode=ANY` fallback.
- **Tuple length** not enforced by a plain number array → set `minItems=maxItems=N`
  in the sanitizer; wrong length fails pydantic → re-ask (acceptable).
- **Caching prefix isn't per-batch identical** (render is trimmed) → cache only
  the stable `build_pass1_blocks` bulk; send trimmed render uncached.
- **`google-genai` API drift** (thinking/caching arg names vary by version) →
  pin the version; wrap thinking/caching config construction so a missing arg
  degrades gracefully (log + continue without it).
- **Experimental runs polluting a project** → the A/B harness must delete its own
  runs and restore the latest `ready` run (learned the hard way in the drop-in test).
- **Determinism** for a fair A/B → `temperature=0`.
