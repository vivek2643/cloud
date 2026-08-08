# Wire the new vcut scene_specifics into the brain

## Problem

The vcut Pass-2 pipeline (`vcut_pass2_video_specifics.plan.md`) produces rich,
video-grounded shot descriptions per cut and persists them to
`cut_records.scene_specifics`. That data reaches the brain's data map as a raw
blob, but it **never renders into the brain's prompt** — so today the brain sees
*none* of it.

The break is a single-line shape mismatch. The only renderer that turns
`scene_specifics` into a brain-prompt tag is `_specific_tag`:

```763:766:backend/app/services/l3/footage_map.py
    specific = ((m.get("scene_specifics") or {}).get("specific") or "").strip()
    if not specific:
        return ""
```

It reads the **old** shape `{"specific": "...", "label": "..."}` written by the
retired `l3/scene_specificity.py` Pass-B. The **new** vcut shape has no
`specific` key, so this returns `""` and the beat line renders nothing.

Net effect: the brain currently gets *less* shot description than under the old
architecture (only the Pass-1 `summary`), despite the new pipeline generating
much more. The fix is purely a renderer change — no model spend, no re-ingest.

## The new shape (ground truth)

`scene_specifics` is now a **flat dict keyed by question-bank ids**
(`backend/app/services/vcut/questions.py`), assembled in
`backend/app/services/vcut/pass2.py` (`out[c.key] = c.value`) and composed onto
resolved cuts in `backend/app/services/vcut/resolve.py`
(`_composed_specifics`). Possible keys:

- Descriptive: `subject`, `action`, `moment_type`, `setting`, `on_screen_text`,
  `notable_object`, `count`, `motion_quality`, `continuity_cue`, `setting`
- Edit-decision: `shot_size` (wide/medium/close/extreme_close),
  `camera_move` (static/pan/tilt/push/handheld), `motion_direction`,
  `subject_entry_exit`, `headroom_lookroom`, `usable` (strong/ok/weak),
  `energy_emotion`, `hook_potential`, `tags` (array), plus custom-probe keys
  (planner-minted, flat `key: value`).
- Merged-cut only: `moments`: `[{t_ms, summary, <flat specifics per absorbed flag>}]`
  — one entry per moment inside a loose cut (single-flag cuts have no `moments`).

Any given cut only carries the subset the question planner (`qplan.py`) selected
for its moment(s), so **every field is optional**. Old pre-migration rows still
carry `{specific, label}`; both shapes must render.

## Scope

- **One file, one function** (plus tests): rewrite `_specific_tag` in
  `backend/app/services/l3/footage_map.py`.
- No plumbing changes: `cutrecord_map._to_cut_dict` (line 563) and
  `footage_map` moment dict (line 367) already carry the blob through, and
  `_moment_line` (line 996) already splices `spec_tag = _specific_tag(m)` into
  the beat line. Fixing the function is enough.
- **Descriptive only.** Per the design decision, the specifics feed the brain
  *facts*; the brain does the judgment. Render `usable`/`hook_potential` as
  neutral descriptive tokens (facts the brain may weigh), never as a
  select/discard decision applied here.
- No re-ingest, no DB migration, no model calls.

## Implementation

### 1. Rewrite `_specific_tag(m)` to render the new shape

Read `spec = m.get("scene_specifics") or {}`. Branch:

**a. Legacy shape (`spec.get("specific")` present):** keep exactly today's
behavior — `spec:"<short_gist(specific)>"`. This preserves any un-migrated rows.

**b. New shape (no `specific` key, dict non-empty):** compose a compact,
ordered, token-bounded `spec:"..."` from a fixed priority list of keys. Suggested
order and format (skip any absent/empty field; never emit a bare label with no
value):

1. `subject` + `action` → the lead phrase, e.g. `barista pours latte`
   (join with a space; if only one present, use it alone).
2. `shot_size` → e.g. `medium` (raw token).
3. `camera_move` → only when not `static`/`unknown` (a static shot adds no
   signal), e.g. `push`.
4. `on_screen_text` → `text:'<short_gist>'` (single-quoted, gisted). Skip when
   empty. (Note: this may duplicate the code-derived `screen_text` tag already on
   the line — de-dupe: if `m.get("screen_text")` already carries the same text,
   omit here.)
5. `motion_direction` → only when not `none` (reframe/match-cut hint).
6. `headroom_lookroom` → only when not `good` (only surface when notable).
7. `usable` / `hook_potential` → descriptive tokens, e.g. `usable:weak`,
   `hook:high`, only when present. Keep terse.
8. `tags` → up to 3, comma-joined, e.g. `tags:pour,closeup,coffee`.

Render as `spec:"subject action · medium · push · text:'…' · usable:weak"`
(mid-dots between groups, matching the existing tag idiom). Keep the whole tag
under a sane length (reuse `_short_gist` on free-text fields; cap tag count).

**c. Merged loose cut (`spec.get("moments")` is a non-empty list):** the cut
contains multiple moments — render a compact mini shot-list so the brain sees the
beats *inside* one cut. Format:

```
spec:"<representative subject/action> · medium" moments:[
  +1.2s barista grabs cup;
  +3.4s pours latte, close;
  +5.1s slides across counter]
```

i.e. the representative fields first (from the flat top-level keys, same as case
b), then a bracketed list of `+<Δt from cut in>s <summary or subject action>`
per entry in `moments` (use each moment's own `summary`, falling back to
`subject action`). Cap the list length (e.g. first 5 moments; append `…+N more`
if longer). Compute `+Δt` as `moment.t_ms - m["in_ms"]` in seconds.

**d. Empty (`{}`):** return `""` (unchanged — a common, valid state before
enrichment reaches a cut).

Keep a single small helper (e.g. `_render_new_specifics(spec)`) so cases b/c
share the top-level field composition.

### 2. Respect compact vs resident mode

`_moment_line(m, compact=...)` renders full detail in resident mode and truncates
in compact/paged mode. Mirror that:

- Resident mode: full `spec:"..."` + `moments:[...]` list.
- Compact mode: only the lead group (`subject action · shot_size`), drop the
  `moments` list and the secondary tokens, and rely on `inspect_moment`
  (Step 3) for full detail. Pass `compact` down into `_specific_tag`
  (add the kwarg; update the call site at line 975).

### 3. Surface full specifics via `inspect_moment` (optional but recommended)

The brain drills into a beat via the `inspect_moment` tool (see `l3/tools.py` /
`l3/converse.py`). Add the full, untruncated `scene_specifics` (pretty
key→value list, including all `moments`) to that tool's output so the brain can
pull the complete shot log on demand for a specific cut without bloating every
line of the resident map. This is where `count`, `notable_object`,
`continuity_cue`, `setting`, custom-probe answers, etc. live — carried on demand
rather than inline.

### 4. De-dupe against existing code-derived tags

The beat line already carries code-derived `cam:` (from `m["camera"]`) and
`text:` (from `m["screen_text"]`). Avoid double-printing:

- If `_specific_tag` would emit `camera_move` equal to the existing `cam:` token,
  omit it in spec (or vice-versa — pick one owner; prefer keeping `cam:` as the
  code-derived truth and dropping `camera_move` from spec when they agree).
- Same for `on_screen_text` vs `text:`.

Keep the spec tag focused on what the *vision model* added beyond code signals
(subject/action/shot_size/motion_direction/headroom/usable/hook/tags).

## Testing

Add cases to the footage_map test suite (find the existing test for
`_moment_line` / `_specific_tag`; likely `backend/scripts/test_*footage*` or
`test_l3_*`). Assert `_specific_tag` output for:

1. **Legacy row** `{"specific": "...", "label": "..."}` → unchanged `spec:"..."`.
2. **New single-flag** `{"subject": "barista", "action": "pours latte",
   "shot_size": "medium", "camera_move": "push"}` → contains
   `barista pours latte`, `medium`, `push`.
3. **New merged** with `moments:[{t_ms, summary}, ...]` → renders the mini
   shot-list with `+Δt` offsets, capped at N.
4. **Empty `{}`** → `""`.
5. **Compact mode** → lead group only, no `moments` list.
6. **De-dupe** → `camera_move` omitted when it equals the code `cam:` tag.
7. Full pipeline sanity: `assemble_map` over a fixture cut with new specifics
   contains the rendered `spec:` on the beat line (integration-level).

All fixtures are in-memory dicts — zero real spend, no DB, no model.

## Rollout

1. Rewrite `_specific_tag` (+ helper) — Step 1.
2. Thread `compact` — Step 2.
3. `inspect_moment` full specifics — Step 3.
4. De-dupe — Step 4.
5. Tests — all green + pyflakes clean.
6. Manual: open one enriched project's brain map (or dump `assemble_map` for a
   known-enriched `file_id`) and eyeball that `spec:"..."` now carries the real
   shot description alongside the `said_text` transcript the brain already gets.

No re-ingest, no migration, no spend. Existing `scene_specifics` data becomes
visible to the brain the moment this ships.

## Out of scope (explicitly not doing here)

- Making specifics *judge* (select/discard). The brain does judgment; this plan
  only makes the description reach it.
- Changing the vcut Pass-2 question bank or specifics content.
- Frontend cut-card rendering (that already reads the blob directly and is a
  separate surface from the brain).
