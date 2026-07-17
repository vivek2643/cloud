# V4 Cuts as the Primitive — remove atoms from the video path

## 0. One-line intent

Make the **V4 cut the atomic unit for video**: computed directly from signals,
carrying its own span, passed to the brain as-is. Pass 2 only *labels* it — it
never re-splits it. Speech is unchanged (word-based cuts → Pass 1 grouping).
Atoms stay only as a **speech/word substrate** and as the **V3 fallback**, both
gated by `settings.cuts_segmenter`.

## 1. Why (the evidence, not a hunch)

Every V4 failure this session is one root cause — **free-form V4 spans forced
onto a grounded atom grid**:

| Failure seen | Real cause |
|---|---|
| `atom_id N in both video_group[i] and [j]` | one atom straddles a V4 boundary |
| `video_group[N] has no atom_ids` | a V4 cut is smaller than any atom |
| `overlap [23571-25280] vs [24020-25040]` | Pass 2 **split** a V4 group along atoms; the piece got an atom-bbox span that collides with the whole-group V4 span |

`_finalize_cuts` (~60 lines) exists *only* to reconcile these. The split
capability (`pass2.py` ~520-524: "split a video group into multiple cuts along
its existing `atom_ids`") is the only reason a video cut carries atoms at all.
Remove the pretense → remove the reconciliation glue → remove the whole class.

**Atoms are a good substrate (lossless tiling, word alignment), a bad
primitive (carved at signal events + scrap, never at "usable moment").** V4
cuts are purpose-built as usable moments. So: video primitive = V4 cut; speech
substrate = words/atoms (kept).

## 2. Guardrails

- **Switchable.** `settings.cuts_segmenter="v3"` keeps atoms + Pass-1 video
  grouping + Pass-2 split **byte-identical to today**. `"v4"` is the new
  atom-free video path. Every branch below is gated; nothing on the V3 path
  changes behavior.
- **Nothing generic is lost.** No podcast/domain assumptions anywhere.
- **Speech untouched.** Speech cuts, backchannel splits, `word_span`, and Pass-1
  speech grouping stay exactly as they are.
- **Scrap is discarded by design.** V4 covers all *usable* non-speech; leftover
  scrap contributes no cut. `post` already treats a zero-cut file as legal.

## 3. Data-model change

### 3.1 Introduce an explicit video-cut primitive out of Pass 1

Today the V4 cut is disguised as `VideoTentativeGroup(atom_ids=[...])`. Replace
that (V4 path only) with a first-class carrier that holds the **span**, not
atoms.

- `pass1.VideoTentativeGroup` gains an optional field so the model/schema is
  unchanged on V3 but V4 can carry a span with **no atoms**:
  ```python
  class VideoTentativeGroup(BaseModel):
      file_id: str
      atom_ids: List[int] = []          # V3: grounded atoms. V4: EMPTY.
      src_in_ms: int | None = None      # V4: the cut's own span (ground truth)
      src_out_ms: int | None = None
  ```
  On V4, `ingest` builds these with `atom_ids=[]` and the real span filled.
  `v4_meta_by_ref` continues to carry salience/density/shape (unchanged).

### 3.2 Pass 2 cut: atoms become optional and video never splits

- `pass2` `CutJudgment` / `Cut`: `atom_ids` already `List[int] | None`. On V4,
  a video cut's `atom_ids` is **always `None`**.
- `pass2._backfill_*`: on V4, skip the "fill atom_ids from the group" and the
  "split along atoms" paths entirely for video. A video `source_ref` maps
  1:1 to exactly one output cut.
- Delete (V4 only) the video validators that enforce the atom contract:
  `_split_groups_partition_atoms`, `_no_duplicate_atoms`,
  "boundary-inside-atom", "every video cut owns ≥1 atom". They validate a
  contract V4 no longer has.
- **Prompt change (V4):** drop the "you MAY split a video group along its
  atom_ids" clause. New rule: *"Each video unit is a finished cut — emit exactly
  one labeled cut per `video_group[i]`, never split or merge it. Your job on
  video is to describe it (summary/label/shape/characteristics) and decide
  keep-vs-junk for the WHOLE cut, not to re-cut it."* Speech-split guidance is
  untouched.
- **Junk-the-whole-cut is allowed (resolved).** The labeler has seen the frames;
  the segmenter has not. So Pass 2 may mark an entire V4 video cut as junk (drop
  it) — that's the "dispose" half of the funnel. The only thing it cannot do is
  change the cut's boundaries. Junking is per-cut (all-or-nothing), never a
  partial re-cut. `post` already treats a file whose cuts all got junked as a
  legal zero-cut file, so no extra handling is needed.

## 4. Span & frame resolution (delete the atom-bbox fallback for video)

- `post.assemble_cut_records` (line ~803-817): on V4 a video cut's span is
  **always** `v4_meta_by_ref[source_ref]`. Because Pass 2 no longer splits,
  every video `source_ref` is present in `v4_meta_by_ref` → the atom-bbox
  `else` branch (816-817) is never hit for video. Keep it only as the V3 path.
- `image_plan._atom_group_span`: unchanged for V3. On V4, frame spans already
  come from `v4_meta` (line ~193-201) — verify no video code path falls to
  `_atom_group_span` when the flag is v4.
- `identity/faces`: already uses `v4_span_override`. No change (it never needed
  atom_ids, only the span).

## 5. Continuity & any other atom-keyed video logic

- `post._clip_continuity` / seam detection: currently keyed off atom boundaries.
  On V4, key seams off **V4 cut boundaries** (adjacent cuts' `src_out`/`src_in`)
  — which is strictly better (real cut edges, not grounded micro-edges). Gate
  the V4 variant; keep V3 atom-based path.
- `cuts_v3_read` / `ingest_store`: audit the 1-3 atom references; ensure the
  reader tolerates video cuts with `atom_ids = []`/`None` (display span from
  `src_in/out`, which is already persisted on the cut_record).

## 6. The segmenter itself gets simpler

`v4_segment.segment_video`:

- **Stop emitting `atom_ids`.** `VideoCut` drops `atom_ids` (or leaves it empty
  and unused).
- **Delete `_finalize_cuts`'s atom half**: no atom-ownership assignment, no
  atom-less merge loop. Keep only the two invariants that are intrinsically
  correct for spans:
  1. **Disjoint + clamped to working span** (already added) — cuts never
     overlap each other or speech.
  2. **`min_ms` floor** — a sub-floor sliver merges into its nearest neighbor
     (this is the *good* reason the merge existed; keep it, but merge on
     **duration**, not on atom-emptiness).
- Net: `_finalize_cuts` shrinks to ~15 lines of pure geometry. The atom bugs
  can't recur because atoms aren't in the loop.

## 7. What stays exactly the same

- **Speech path unchanged.** Transcript (words + speaker labels) is produced in
  L1 *before* Pass 1; the Pass-1 LLM call groups those words into speech cuts
  (`SpeechCut.word_span`), then deterministic repair runs
  (`enforce_lattice_partition` + `_split_at_speaker_changes`, backchannel-aware).
  All of this — grouping, backchannel splits, `word_span`,
  `resolve_speech_span_ms` — is untouched. On V4 the Pass-1 LLM effectively only
  does speech grouping (video no longer goes through it).
- **`lattice.atoms` MUST still be built (do not strip it).** The speech path
  depends on atoms as its substrate: `_split_speech_at_atom_gaps` uses "is there
  an atom in this inter-word gap?" to decide whether the model may weld two
  sentences across a pause (short pause = weld allowed; atom in the gap = hard
  split, that gap is non-speech territory). Removing atoms from the *video cut*
  does NOT remove atoms from the lattice — we only stop threading `atom_ids`
  into video cuts. Speech gap/weld behavior stays byte-identical.
- `lattice.build` still produces atoms (words need them; cheap). We just don't
  *thread atoms into video cuts* on V4.
- Salience/shape/density, the asymmetric energy ladder, `compute_pace_envelope`,
  `read_state`/`review`/Program Map, guidance doc — all unchanged.
- The entire V3 path (flag = `v3`).

## 8. Downstream contract summary

| Module | V3 (flag=v3) | V4 (flag=v4) |
|---|---|---|
| `v4_segment` | n/a | emit cuts with span, **no atom_ids**; `_finalize_cuts` = geometry+min_ms only |
| `pass1` video | LLM groups atoms | groups carry span, `atom_ids=[]` |
| `pass2` video | may split along atoms | **label + keep/junk only, 1 cut per ref, no split**; video atom validators disabled |
| `post` video span | atom bbox | V4 span (`v4_meta`) always |
| `image_plan` video | atom span | V4 span (already) |
| continuity | atom boundaries | V4 cut boundaries |
| speech (all) | words/atoms | **unchanged** |

## 9. Testing

- `test_v4_segment.py`: drop the atom-ownership/duplicate/empty-atom tests
  (contract removed); replace with: disjoint spans, clamped-to-working-span,
  sub-`min_ms` sliver merges into neighbor, no cut crosses speech.
- `test_pass2.py`: add a V4 case — N video groups in → exactly N labeled video
  cuts out, each `atom_ids is None`, no split, no atom validation error.
- `test_post` / `test_ingest`: V4 fixture with a video cut smaller than any
  atom and one straddling an atom boundary — both must now pass cleanly
  (these are the exact shapes that fail today).
- Keep all V3 tests green (regression guard for the flag).

## 10. Rollout

1. Land behind the flag; run V3 tests (must be identical) + new V4 tests.
2. Smoke-ingest the 3 currently-failing/unfinished projects
   (`a596ea5f`, `a294f9da`, `57b689b3`) — these fail today purely on atom
   friction and should now pass.
3. Re-ingest all 13 under V4 (the 10 already-good ones should be unchanged or
   better; the failing ones should complete).
4. Once stable, V3 can be retired in a later cleanup (not part of this change).

## 11. Resolved decisions

- **Pass 2 may junk a whole V4 video cut** — confirmed. Segmenter proposes
  candidate usable moments; the labeler (which has seen the frames) disposes by
  keeping or junking the whole cut. It can never re-cut boundaries. This keeps
  "extract the usable, discard the scrap" as a clean two-stage funnel without
  reintroducing atoms. Wired via the existing junk path (`junk_confidence` /
  drop), tested in §9.
