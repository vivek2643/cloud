# Brain continuity-awareness — implementation plan

Make the blind editor ("Edso") aware of the continuity of the joins **it actually
creates on the timeline** — not just the source-neighbor continuity baked into
each cut at ingest. Two features, both **read-side / brain-facing only** (no
re-ingest, no schema change):

- **Feature 1 — LIVE join-continuity score.** A pure signal over the current
  resolved timeline that classifies every adjacent seam as SEAMLESS / SAME-SHOT
  JUMP / CUT, surfaced un-gated at decision time and review time.
- **Feature 2 — Sacred continuity runs.** Bind the existing v15 continuity-run
  grouping as a strong preference in guidance, and surface a run's ordering
  prominently enough that the brain can place a run as one ordered unit.

Features **(3) disprefer/hide same-clip non-contiguous joins** and
**(4) deterministic compile-time backstop** are **OUT OF SCOPE** — see
[§7 Deferred](#7-deferred-3--4).

---

## 1. Problem statement

The fragmented edit (thread `57a517fd`, project `8621c012`, run `24894c8a`) came
from the brain juxtaposing same-person / same-clip beats **wildly out of source
order** with no check on the joins it was creating. Program order vs. source
time on the spine:

| prog | file | src start | content |
|---|---|---|---|
| a002 | 93e94ec3 | **408.1 s** | "Big players trying to automate video editing" |
| a003 | 93e94ec3 | **209.5 s** | "LLM decides editing commands" |
| a004 | 93e94ec3 | **14.4 s** | "Start with a demo" |
| a005 | 93e94ec3 | 31.7 s | "user selects multiple files" |
| a006 | 93e94ec3 | 44.6 s | "user types a prompt" |

Within one continuous recording (`93e94ec3`) the brain jumped `408 → 209 → 14 s`
— three backward same-clip cuts in a row. Each is a **same-shot jump cut**: same
room, same person, same framing, only the source time leaps. Nothing in the
brain's senses scored those *joins*, and the one warning that comes closest
(`same speaker back-to-back`) was suppressed by the done-gate before the brain
could act (see §2.3). Diagnosis confirmed this was a **genuine brain choice**,
not a heal/reindex regression — so the fix is to give the brain the missing
sense + a binding preference, not to change the pipeline.

### The core gap: source-descriptive vs. join-aware

Every continuity signal we ship today describes a cut's **ORIGINAL source
neighbors** — who it welded to *in the raw clip*. None of them describe the pair
of cuts the brain actually places **adjacent on the timeline**. That is the
missing sense.

---

## 2. Current-state findings (file:line)

### 2.1 What continuity data exists, and how it's rendered

**Per-cut source-neighbor continuity (persisted at ingest).** Each cut carries a
`continuity` block `{cut_no, of, prev_contiguous, next_contiguous,
seam_reason_prev, seam_reason_next}`, produced once at ingest per
`cuts_v3_continuity.plan.md` and passed straight through untouched:

- `backend/app/services/l3/cutrecord_map.py:549` — `"continuity": row.get("continuity") or {}` in `_to_cut_dict` (also `junk`/`junk_reason` at `:547-548`). `CUTRECORD_MAP_VERSION = 5` at `cutrecord_map.py:59`.
- `backend/app/services/l3/footage_map.py:353` — `build_clip_tree` copies `"continuity": cut.get("continuity") or {}` onto each moment.
- Rendered by `footage_map._continuity_tag` (`footage_map.py:1082`) → `· ↔cut:3/57⋯`, using `_weld_mark` (`footage_map.py:1040`): `↔` = weldable continuation toward that neighbor, `⋯` = hard break, `''` = no neighbor.
- Emitted onto the beat line at `footage_map._moment_line:1148` (`cut_tag = _continuity_tag(m)`), described to the brain in the system prompt at `converse.py:116-119`.

**Continuity RUNS (v15, computed at tree-build, clip-local).** A maximal chain of
same-clip beats that are back-to-back in source time within `_RUN_GAP_MS`:

- `backend/app/services/l3/footage_map.py:199` — `_assign_runs(moments)`; `_RUN_GAP_MS = 500` at `footage_map.py:127`; called from `build_clip_tree:393`.
- Members (≥2) get `m["run_id"]` (`"r0"`, `"r1"`, …), `m["run_pos"]`, `m["run_len"]` — `footage_map.py:234-237`. **All three are stored in the cached tree** (they mutate the moment dicts before the tree is returned), so they need no recompute to render.
- Rendered by `_moment_line:1145-1147`: only `· run:rN` — the **run id, but NOT `run_pos`/`run_len`**, so the brain sees *that* beats share a run but not their order within it. `TREE_VERSION = 22` at `footage_map.py:121` (v15 note at `:86-92`).

**Proof it is source-descriptive, not join-aware.** `_continuity_tag`,
`_weld_mark`, and `_assign_runs` all read `continuity` / source `in_ms/out_ms`
from the **cut record** — they describe the raw clip. Nothing in
`footage_map.py`, `feel.py`, or `observe.py` compares the source spans of two
segments that end up **adjacent on the working timeline**.

### 2.2 How the resolved timeline exposes source file id + src in/out

The join score needs, per spine segment, its source file and source in/out. Both
are already present in two places:

- **Timeline segments** (`document["timeline"]`): each seg carries `file_id`, `in_ms`, `out_ms` (source ms), `ref`, `seg_id`, `axis`, `content`. Confirmed by usage in `feel.simulate` (`feel.py:107,119`) and `read_state` (`observe.py:508,518,536`).
- **Resolved spine layers** (`document["resolved"]["video_layers"]`, `kind=="spine"`): `VideoLayer` dataclass carries `source_file_id`, `src_in_ms`, `src_out_ms`, `prog_start_ms`, `prog_end_ms` — `backend/app/services/l3/layers.py:258-283`. Produced by `layers.resolve` at the end of `observe.resolve_doc` (`observe.py:319-321`).

Because the info lives on the plain timeline segs, **the classifier can be a pure
function of `document["timeline"]`** — no resolve required, and it composes with
the existing `feel.simulate` path that already walks the timeline.

### 2.3 Proof the jump warning is gated

`feel._same_speaker_runs` (`feel.py:171-187`) already finds ≥2 adjacent
same-speaker speech cuts, and `observe.diagnose` turns each into a
`warn` finding `"same speaker back-to-back (jump-cut risk)"`
(`observe.py:1176-1178`). But the done-gate swallows it:

- `tools._verify_before_finish` (`tools.py:624`) runs staged checks. Stage 3
  ("specific flags") is the **only** stage that surfaces `diagnose` findings —
  and it is guarded at `tools.py:693`:

```693:694:backend/app/services/l3/tools.py
    if reviewed or state["reviewed"]:
        return None
```

  `reviewed` is set true the moment the brain called `diagnose`/`validate`/`review`
  this turn (`tools.py:667`). So once the brain self-reviews (which a competent
  turn always does), **every Stage-3 finding — including the same-speaker /
  same-shot jump warning — is dropped before the brain is ever forced to act.**
  Stage 2's "fit to craft" nudge (`tools.py:683-691`) still fires, but it is
  blind prose ("judge it cold"), not the concrete anchored verdict.

`diagnose`, `read_state`, and `review` themselves are **not** gated — they return
whatever they compute whenever the brain calls them. The gate problem is
strictly the *finish-time forcing function*: a genuine jump verdict must be able
to **block finishing**, not just sit in an advisory list the gate skips.

---

## 3. Feature 1 — LIVE join-continuity score

### 3.1 Where it's computed (a pure helper)

Add the classifier to `feel.py`, next to `_same_speaker_runs`, because
`feel.simulate` already walks the timeline once and is the pure, DB-free "how
does this edit feel" layer that `read_state`/`diagnose` both consume.

**Step A — carry source spans on `CutFeel`.** Extend the dataclass
(`feel.py:32-44`) with two fields:

```python
@dataclass
class CutFeel:
    ...
    file_id: str
    src_in_ms: int      # NEW: source in (seg["in_ms"])
    src_out_ms: int     # NEW: source out (seg["out_ms"])
    ...
```

Populate them in `simulate` (`feel.py:116-127`) from `seg.get("in_ms")` /
`seg.get("out_ms")` (already read at `feel.py:107`).

**Step B — the classifier.** Pure function over the ordered `CutFeel` list:

```python
# A same-clip forward gap within this tolerance reads as one continuous shot
# (matches _RUN_GAP_MS, the source-run grouping threshold).
_JOIN_CONTIG_MS = 500

def join_continuity(cuts: List[CutFeel]) -> List[dict]:
    """Classify the join FROM each cut to the one before it (pos>=2).
    Pure; reads only file_id + src_in/out already on each CutFeel."""
    out: List[dict] = []
    for i in range(1, len(cuts)):
        prev, cur = cuts[i - 1], cuts[i]
        if not prev.file_id or not cur.file_id or prev.file_id != cur.file_id:
            kind = "cut"                      # different clip -> normal cut
        else:
            gap = cur.src_in_ms - prev.src_out_ms
            if -_JOIN_CONTIG_MS <= gap <= _JOIN_CONTIG_MS:
                kind = "seamless"             # same clip, source-contiguous
            else:
                kind = "jump"                 # same clip, non-contiguous / out-of-order
        out.append({
            "from_pos": prev.pos, "to_pos": cur.pos, "kind": kind,
            "same_clip": prev.file_id == cur.file_id,
            "src_gap_ms": (cur.src_in_ms - prev.src_out_ms) if prev.file_id == cur.file_id else None,
            "backward": prev.file_id == cur.file_id and cur.src_in_ms < prev.src_out_ms,
        })
    return out
```

Classification (matches the user's spec):
- **same clip, source-contiguous** (`|gap| ≤ _JOIN_CONTIG_MS`, incl. small overlap) → `seamless`.
- **same clip, source non-contiguous** (`gap` beyond tolerance, in either direction; `backward` when `cur.src_in < prev.src_out`) → `jump` — the thing to flag.
- **different clip** → `cut` (normal, not flagged).

`_JOIN_CONTIG_MS = 500` intentionally equals `footage_map._RUN_GAP_MS` so "these
two beats are one source run" and "this join is seamless" use the same
threshold. (An out-of-order pair inside one v15 run — see Feature 2 — is exactly
a `jump` here.)

**Step C — a run-order helper for Feature 2** (co-located, same file): given the
timeline refs + `meta_by_ref` run tags, detect a run whose members were placed
out of source order (see §4.2).

### 3.2 How it's surfaced UN-GATED

**Decision time — `read_state`** (`observe.py:477`). In the per-cut loop
(`observe.py:517-573`) attach a `join` tag describing the seam **into** each cut
(omit for `pos==1`). Compute once before the loop:
`joins = {j["to_pos"]: j for j in feel.join_continuity(report.cuts)}`
(the `report` is already built at `observe.py:486`). Then per cut:

```python
j = joins.get(i + 1)
if j and j["kind"] != "cut":
    cut["join"] = j["kind"]            # "seamless" | "jump"
    if j["kind"] == "jump":
        cut["join_note"] = (
            "same-clip jump: source "
            + ("goes backward" if j["backward"] else f"skips {j['src_gap_ms']//1000}s")
        )
```

Also append a compact jump summary to the feel narration (§3.3) so a plain
`read_state` shows it without per-cut inspection.

**Decision time — `diagnose`** (`observe.py:1166`). Add, alongside the existing
`_same_speaker_runs` block (`observe.py:1176-1178`):

```python
for j in feel.join_continuity(report.cuts):
    if j["kind"] == "jump":
        findings.append({
            "severity": "warn",
            "anchor": f"cuts {j['from_pos']}-{j['to_pos']}",
            "message": "same-shot jump (same clip, source non-contiguous)",
        })
```

**Review time — `review`** (`observe.py:1448`). Add a continuity flag with a
**new category `"continuity"`** (so it can be routed past the done-gate, see
§3.4) after the existing craft flags (`observe.py:1497-1503`):

```python
report = feel.simulate(timeline, ctx.meta_by_ref)
flags.extend(_tag_category(
    [{"severity": "warn", "anchor": f"cuts {j['from_pos']}-{j['to_pos']}",
      "message": "same-shot jump (same clip, source non-contiguous)"}
     for j in feel.join_continuity(report.cuts) if j["kind"] == "jump"],
    "continuity"))
```

### 3.3 The exact rendered form the brain sees

Consistent with the existing tag vocabulary (`↔` weld / `⋯` break / `cut:N/of`):

- **Per-seam in `read_state.cuts[i]`**: structured field `join: "seamless" | "jump"` plus `join_note` for a jump. When `read_state` is folded into prose elsewhere, render as a seam glyph reusing the established marks: `↷jump` for a same-shot jump, `↔seamless` for a seamless continuation, nothing for a normal cross-clip cut.
- **In `feel.narrate()`** (`feel.py:58-79`), add a clause mirroring the same-speaker one (`feel.py:75-77`):

```python
jumps = [j for j in join_continuity(self.cuts) if j["kind"] == "jump"]
if jumps:
    spans = ", ".join(f"{j['from_pos']}\u2192{j['to_pos']}" for j in jumps)
    parts.append(f"same-shot jump cuts at {spans} (same clip, out of source order)")
```

- **In `diagnose`/`review` flags**: `[warn] cuts 3-4: same-shot jump (same clip, source non-contiguous)`.

Describe the new `join`/`↷jump` marks to the brain in `converse._LOOP_SYSTEM`'s
"READING A BEAT LINE"/senses section (`converse.py:113-128, 157-159`): one line
that `read_state`/`diagnose` now score the **joins you create** —
`↷jump` = you placed two pieces of the SAME clip out of source order (a same-shot
jump); `↔seamless` = same clip, continuous.

### 3.4 The un-gating fix to `tools._verify_before_finish`

Add a dedicated **continuity stage that runs BEFORE the `reviewed` early-return**
and is **not** skippable by having called `diagnose`/`review`. Insert between
Stage 2 (craft, `tools.py:683-691`) and the Stage-3 early-return
(`tools.py:693`):

```python
# Stage 2.5 -- CONTINUITY (hard-ish): a genuine same-shot jump must reach the
# brain even after it self-reviewed. Surfaced at most once (state-tracked), so
# the loop still terminates; unlike Stage 3 it is NOT gated on `reviewed`.
jumps = [f for f in findings if "same-shot jump" in (f.get("message") or "")]
if jumps and not state["continuity_surfaced"]:
    state["continuity_surfaced"] = True
    body = "\n".join(f"- {f.get('anchor')}: {f.get('message')}" for f in jumps[:8])
    return ("AUTOMATIC CHECK -- continuity: you placed pieces of the SAME clip "
            "out of source order (same-shot jump cuts). Reorder them into source "
            "order, bridge with a different clip, or -- if this juxtaposition is "
            "deliberate -- say in ONE line why, then finish:\n" + body)
```

- `findings` is already computed at `tools.py:659` (`observe.diagnose(...)`), which now includes the same-shot-jump warnings from §3.2 — no extra call.
- Add `"continuity_surfaced": False` to the `verify` state dict initialised at `tools.py:723-724`.
- Because this returns **before** `tools.py:693`, the `reviewed` gate can no longer swallow a real jump verdict. It surfaces exactly once, so termination is preserved (same one-shot discipline as every other stage).
- Mirror this stage in the FINISHING description in `converse._LOOP_SYSTEM` (`converse.py:160-176`): add a continuity item ahead of "(3) SPECIFIC FLAGS" so the brain knows a same-shot jump can block finishing.

This satisfies "un-gated at BOTH decision time and review time": decision time via
`read_state`/`diagnose` (never gated), review time via the new `continuity`-category
review flag **and** the new finish stage that fires regardless of `reviewed`.

---

## 4. Feature 2 — Sacred continuity runs

### 4.1 Guidance paragraph (binding default)

Continuity runs already exist (`_assign_runs`, §2.1) but are only a soft "hint"
(v15 note, `footage_map.py:90-92`). Bind them. Add a new numbered section to
`backend/app/services/l3/guidance_doc.md` (injected verbatim as a binding
default via `converse._guidance_block`, `converse.py:356-360`). Insert after
§4 "Select for the video's purpose" (`guidance_doc.md:77-93`), matching the
doc's second-person, principle-not-cookbook voice:

```markdown
## 5. Keep a continuous shot continuous
Beats that share a `run:` tag are one uninterrupted stretch of a single clip —
the camera never stopped between them. Treat that run as one shot: keep its
members together and in their source order (the `run:rN[i/of]` position), and
prefer placing a whole run as a single continuous stretch rather than scattering
its pieces. Do not shuffle a run's beats out of order or interleave another clip
between them unless you have a real reason — cutting from one piece of a shot to
an EARLIER or much-later piece of the SAME shot is a jump cut (same room, same
person, only the source time leaps), the most jarring join there is. Splitting a
run, or reordering it, is a deliberate move you can make when it serves the story
— but never the accidental byproduct of picking beats by meaning and ignoring
where they came from. When you do need a beat from mid-run, bridge the seam with
a DIFFERENT clip rather than a backward jump within the same one.
```

(Renumber the current §5 "Working with music" → §6, §6 "Color" → §7, and update
the podcast worked-example's "principle 2" reference if needed. These are the
only cross-references in the doc.)

**`_LOOP_SYSTEM` "simplest edit" check.** The clause at `converse.py:79-81`
("make the SIMPLEST edit … don't add layers, cutaways, effects, or pacing moves
the ask didn't call for") is about **not adding** unrequested complexity; keeping
a run intact and in order is the *simpler, fewer-cuts* choice, so there is **no
conflict** — if anything they reinforce each other. The `PRECEDENCE` line
(`converse.py:90-92`) already establishes guidance defaults bind over the brain's
own judgment, so no `_LOOP_SYSTEM` edit is required for Feature 2. (Optionally add
a half-line to the `cut:N/of` senses description at `converse.py:116-119` noting
runs are sacred; not required since the guidance block carries it.)

### 4.2 Surface the run grouping so the brain can act on it

Today the brain sees `· run:rN` but not the member ordering. Two small
**render-only** additions:

**(a) Show run position on the beat line.** In `footage_map._moment_line`
(`footage_map.py:1145-1147`) render `run_pos`/`run_len` (already on the moment,
§2.1):

```python
run = ""
if m.get("run_id"):
    pos, ln = m.get("run_pos"), m.get("run_len")
    run = (f" · run:{m['run_id']}[{pos + 1}/{ln}]" if pos is not None and ln
           else f" · run:{m['run_id']}")
```

So a run reads `· run:r0[1/4] … r0[2/4] … r0[3/4] … r0[4/4]` down the (source-
ordered) beat index — the brain can see both membership and order.

**(b) Present a run as a unit in `affordances`.** `affordances` (`observe.py:1522`)
currently exposes per-cut options + a flat `cutaway_pool` (`observe.py:1557-1565`)
with no run grouping. Add a `runs` section built from the map struct
(`ctx.map_struct.get("clips")`), one entry per run: `{run_id, file, member_refs
(in source order), on_line: bool}`. This lets the brain place a whole run in one
ordered stretch, and see when a run is only partially on the line.

**Versioning for Feature 2.** `run_pos`/`run_len` have been persisted since v15,
and the beat-index **text is rendered fresh** in `assemble_map`
(`footage_map.py:1334`) from the cached tree — it is **not** cached — so changing
`_moment_line` needs **no `TREE_VERSION` bump and no re-ingest**. Bump
`TREE_VERSION` to `23` **only** as a belt-and-suspenders measure if we want to
guarantee any legacy cached tree row predating v15 gets rebuilt with the run
fields; note it is not strictly required for correctness.

### 4.3 Where runs are computed / persisted / rendered (citations)

- Computed: `footage_map._assign_runs` (`footage_map.py:199-237`), threshold `_RUN_GAP_MS = 500` (`footage_map.py:127`), called from `build_clip_tree` (`footage_map.py:393`). Design origin: `cuts_v3_continuity.plan.md` (continuity substrate) + v15 note (`footage_map.py:86-92`).
- Persisted: inside the per-file cached `footage_trees` row (`footage_map.py:422-432`), keyed by `_tree_version` (`footage_map.py:435-436`).
- Rendered: `footage_map._moment_line:1145-1147` (`· run:rN`).

---

## 5. Exact change list per file

| File | Change | Version bump? | Re-ingest? |
|---|---|---|---|
| `backend/app/services/l3/feel.py` | Add `src_in_ms`/`src_out_ms` to `CutFeel` (`:32-44`) and populate in `simulate` (`:116-127`); add `_JOIN_CONTIG_MS` + `join_continuity(cuts)` helper; add a same-shot-jump clause to `narrate()` (`:58-79`). | none | no |
| `backend/app/services/l3/observe.py` | `read_state`: attach `join`/`join_note` per cut (`:517-573`). `diagnose`: emit `same-shot jump` warnings (`:1176`). `review`: add `continuity`-category jump flags (`:1497-1503`). `affordances`: add a `runs` section (`:1557-1565`). | none | no |
| `backend/app/services/l3/tools.py` | Add Stage 2.5 continuity check before the `reviewed` return (`:693`); add `continuity_surfaced` to `verify` state (`:723-724`). Optionally note it in the `diagnose` tool blurb (`:101`). | none | no |
| `backend/app/services/l3/footage_map.py` | Render `run_pos`/`run_len` in `_moment_line` (`:1145-1147`). Optional defensive `TREE_VERSION` → 23 (`:121`). | optional (TREE_VERSION only if chosen) | no |
| `backend/app/services/l3/guidance_doc.md` | New §5 "Keep a continuous shot continuous"; renumber music/color; check §2 cross-ref. | n/a | no |
| `backend/app/services/l3/converse.py` | Describe the new `↷jump`/`join` marks in the senses section (`:113-128`); add a continuity item to the FINISHING list (`:160-176`). No `_LOOP_SYSTEM` "simplest edit" change needed. | n/a | no |

**Re-ingest: NOT required.** Every input the two features read —
`file_id`, source `in_ms/out_ms` on timeline segs, and `run_id/run_pos/run_len`
on cached moments — is **already persisted**. `CUTRECORD_MAP_VERSION`
(`cutrecord_map.py:59`) does **not** change; `TREE_VERSION` only changes if we
opt into the defensive rebuild in §4.2. These are purely read-side / brain-facing
changes.

---

## 6. Test plan

**Unit — join classifier (`feel.join_continuity`)**, new `tests/l3/test_feel_join.py`:
1. **same-clip contiguous → seamless**: two segs, same `file_id`, `prev.out=10_000`, `cur.in=10_100` (gap 100 ≤ 500) → `kind=="seamless"`.
2. **same-clip non-contiguous forward → jump**: `prev.out=10_000`, `cur.in=45_000` → `kind=="jump"`, `backward is False`, `src_gap_ms==35_000`.
3. **same-clip backward → jump**: `prev.out=408_000`, `cur.in=209_000` → `kind=="jump"`, `backward is True` (the exact `57a517fd` pattern).
4. **cross-clip → cut**: different `file_id` → `kind=="cut"`, `same_clip is False`, never flagged.
5. **boundary/tolerance**: gap exactly `±_JOIN_CONTIG_MS` → `seamless`; one ms beyond → `jump`.
6. **out-of-order run detection**: a 3-beat run placed `[r0/3, r0/1, r0/2]` yields ≥1 `jump` seam.

**Unit — done-gate survives self-review** (`tests/l3/test_tools_verify.py`):
7. Build a working doc with a same-shot jump; set `steps=["diagnose"]` (so `reviewed=True`) and `verify["continuity_surfaced"]=False`; assert `_verify_before_finish(...)` returns the **continuity** feedback string (i.e. it is NOT suppressed by the `reviewed` gate at `tools.py:693`). Then call again with `continuity_surfaced=True` and assert it no longer re-surfaces (termination).

**Unit — surfacing** (`tests/l3/test_observe_continuity.py`):
8. `read_state`: a jump seam sets `cuts[i]["join"]=="jump"` with a `join_note`; a seamless seam sets `"seamless"`; a cross-clip seam sets no `join`.
9. `diagnose`: returns a `warn` finding whose message contains `"same-shot jump"` for the out-of-order fixture.
10. `review`: returns a flag with `category=="continuity"` for the same fixture.

**Render — footage_map** (`tests/l3/test_footage_map_render.py`):
11. A clip whose moments form a run renders `· run:r0[1/N]…[N/N]` in source order on the beat lines; a lone moment renders no `run:` tag.
12. (If `TREE_VERSION` bumped) assert `_tree_version(sig)` reflects the new number so stale rows rebuild.

**Guidance smoke** (extend existing converse test): assert the assembled system
prompt contains "continuous shot" (the new §5 made it through
`_load_guidance`'s comment-strip at `converse.py:351`).

---

## 7. Deferred (3 & 4)

Intentionally **out of scope** for this plan; noted so the implementer knows
they're coming and doesn't half-build them:

- **(3) Disprefer / hide same-clip non-contiguous joins.** A stronger *preference/
  penalty* than Feature 2's guidance — e.g. down-ranking a placement that creates
  a same-shot jump, or visually hiding such joins (a cutaway/reframe auto-bridge).
  Feature 1 only *scores and surfaces* the join; it does not bias selection or
  mask the seam. Deferred.
- **(4) Deterministic compile-time backstop.** A resolve-time guard (near
  `observe.resolve_doc`/`arrange.heal_adjacent_cuts`, `observe.py:288-303`) that
  detects or repairs same-shot jumps mechanically (reorder within a run, or refuse
  to emit a backward same-clip seam) independent of the brain. Feature 1 keeps the
  decision with the brain; this backstop would enforce it in code. Deferred.
