# Cuts v3 — Speech-to-Speech Beat Bridge (via a reusable seam-significance primitive)

## Goal

Let a spoken beat stay **one continuous cut** when it continues across a brief
wordless moment — a pause, or a demonstrated action between two lines
("Watch this … *[swing]* … and that's the topspin"). Today code force-splits
every speech cut at any atom-owned gap, so that always becomes three tiles
(speech · action · speech). We want it to be **one absorbed beat** when the
footage between the lines is a continuous take, and stay split when there's a
genuine break.

## Design principles (unchanged from deterministic-keep)

- **LLM owns MEANING (proposes)**: which spoken beats belong together as one
  continuous moment (a single podcast answer; a line that resumes after a
  demonstrated action). It signals this two ways, both categorical (no numbers):
  1. implicitly, by grouping words on both sides of a wordless moment into ONE
     `speech_cut` range, and
  2. explicitly, by tagging consecutive cuts it judges to be one beat with a
     shared **`beat_id`** (the reliable path — the model reasons logically about
     what belongs together, e.g. "one answer stays together").
- **Code owns the SEAM (disposes)**: given the gap, decide *weld* (absorb into
  one continuous beat) or *hard* (keep split), deterministically and
  clip-relative, with no tuned constants. The model's `beat_id` is only ever a
  request; code merges **only across a weldable seam** (no shot change /
  transition / speaker change; gap not longer than the bridged speech). This is
  why we don't rely on the LLM emitting a number or a precise span — LLM
  proposes, deterministic seam guard disposes.
- **Non-speech is NOT the LLM's call.** We deliberately keep the LLM out of
  "do these two actions belong together" (weakly grounded from stills). A
  bridge is always **anchored by speech on both ends**; the middle is swept in
  by code. Standalone actions elsewhere are unaffected.

## Why explicit `beat_id`, not implicit spans alone

First real Reel-trail run: model (a) (implicit one-span) fired **0 times** — the
LLM emitted two separate `speech_cut`s at every sandwich, so nothing bridged,
even where the seam was clearly weldable (2 such pairs found). Relying on the
LLM to shape its span a certain way is exactly the fragility we're avoiding.
So the LLM gets an explicit, easy categorical handle (`beat_id`), and the
deterministic seam guard is what actually decides the merge.

## The shared primitive: seam significance (`app/services/l3/seam.py`)

One deterministic function answers a question with two callers:

> Given two kept spans on the SAME clip and the gap between them — is the seam
> **weldable** (play as one continuous unit) or **hard** (keep separate)?

A seam is **HARD** (a genuine break) when:
1. **cross-clip** (different footage) — always hard,
2. **speaker change** across the seam (two speakers ≠ one continuous beat),
3. a **shot/scene boundary** or **wipe/degenerate transition** inside the gap
   (read off the atoms' own boundary reasons — `R_SHOT`/`R_WIPE`/`R_DEGENERATE`;
   an `R_ACTION` energy-regime edge is NOT a break — it's continuous footage
   with motion),
4. a **flagged production break** inside the gap (a pass-1 junk suspect — cue /
   reset / dead-air — overlapping the gap's atoms),
5. **magnitude backstop**: the gap is **longer than the speech it bridges**
   (`gap_ms > left_speech_ms + right_speech_ms`). Structural 1:1 — absorb only
   when there's at least as much speech as connective tissue. NOT a tuned knob.

Otherwise → **weldable** (one continuous take).

Callers:
- **now**: ingest beat-bridge (pass 1 enforce), below.
- **future hook (documented, not wired)**: timeline weld — when the editor
  drops two cuts adjacent, the SAME function decides weld vs hard cut
  (cross-clip → always hard).

## Ingest wiring (`pass1.enforce_lattice_partition`)

Today: every LLM `speech_cut` is split at every atom-owned inter-word gap
(`_span_pieces`), and the post-condition forbids a speech cut from containing
any atom.

Change:
- For each **LLM** `speech_cut`, split only at **hard** seams (new
  `_seam_split` over `_gap_seam`). Weldable internal gaps are **absorbed**: the
  beat keeps its full word range, and the atoms in that gap are collected as
  **absorbed**.
- **BEAT MERGE (`_merge_beats`)**: after coverage fill, fuse consecutive
  same-file speech cuts that share a `beat_id`, but ONLY across a weldable seam
  (`_gap_seam` → `seam.classify_seam`; a bare no-atom pause welds when same
  speaker). Word-adjacent only (`prev.end+1 == next.start`). Welded gaps'
  atoms are absorbed too.
- **Coverage-fill / recovered** speech cuts keep splitting at every gap
  (`_span_pieces`) and carry no `beat_id`, so they never merge — the LLM never
  claimed those are a beat.
- **Absorbed atoms are removed from the video pool**: dropped from any
  `video_tentative_group` (remaining atoms re-split for contiguity) and skipped
  by the video coverage fill. They produce no video cut — they're covered by
  (and play inside) the speech beat. Deterministic-keep is preserved: nothing
  is lost, it just belongs to the spoken beat now.
- **Post-condition relaxed**: `_no_speech_cut_swallows_atoms` now checks a
  speech span only against **grouped** atoms (members of some
  video_tentative_group), never absorbed ones. The real invariant — no overlap
  between a speech cut and a video cut — still holds (video cuts come only from
  grouped atoms), and `post._validate_no_overlap` remains the final guard.

`resolve_speech_span_ms` already yields the full span for an absorbed beat: the
absorbed atoms sit between the first and last word, so they're neither
"following" nor "preceding" and the inward clamp doesn't touch them.

## Prompt (light nudge only)

Add one sentence to the `speech_cuts` bullet: a spoken beat MAY continue across
a brief wordless moment (a held pause, a demonstrated action) as ONE beat when
it's a single continuous thought — code keeps the footage continuous when the
take is unbroken and splits it at a real cut. No numbers, no thresholds.

## What does NOT change

- No DB migration, no new `cut_records` column: an absorbed beat is just a
  `kind="speech"` cut with a wider span → renders as one tile already.
- No frontend change.
- Take/outlook logic, junk, energy dial: untouched.

## Tests

- `scripts/test_seam.py` (new): each hard rule + the weldable case + the
  magnitude backstop, as pure `classify_seam` unit tests.
- `scripts/test_pass1.py`: a weldable gap is absorbed into one beat (its atoms
  leave the video pool); a hard gap (shot cut / speaker change / over-long gap)
  still splits into speech · video · speech. Recovered coverage-fill still
  splits at every gap.

## Verify

Backend test suite green, then re-ingest Reel trail and confirm: a
speech→action→speech moment that's one continuous take shows as a single beat;
genuine breaks still split; nothing lost; cues still culled.

## OUTCOME / DECISION (after two Reel-trail runs)

- Built, unit-tested, and shipped: `seam.py`, `_gap_seam`, `_seam_split`
  (model-a within-span absorb), `_merge_beats` (LLM-gated by `beat_id`), the
  relaxed grouped-only swallow post-condition. All suites green; run stays
  `ready`, cues still culled, nothing lost.
- **The LLM would not propose beats**: 0/26 `beat_id` tags across two runs,
  even with an explicit prompt, and it splits a continuous sentence at a long
  dramatic pause. So the bridge is **dormant in practice**.
- **Deterministic auto-merge was investigated and REJECTED as unsafe.** Two
  real Reel-trail seams proved no deterministic (non-semantic) signal can tell a
  good merge from a bad one:
  - good continuation "…inform you ⟶ there's a catch": gap `is_action=False`
    (silent 2.8s pause),
  - false start "…matters ⟶ I'm sorry": gap `is_action=True` (energy ~0.9).
  The case that MUST NOT merge has *more* action than the one that could — only
  MEANING (junk/semantics) separates them. Under the user's rules
  (deterministic + never wrong-merge/prefer-split + no junk-detection band-aid),
  an automatic merge is impossible, so it was not built.
- **Resting state (user-chosen):** keep the conservative LLM-gated bridge. It
  merges ONLY when the model explicitly groups a beat AND the seam is
  continuous; it prefers split everywhere else and never wrong-merges. Humans
  still combine cuts via timeline selection-weld, where they have the context
  code lacks. The mechanism is ready if a dedicated beat-grouping call is added
  later.
