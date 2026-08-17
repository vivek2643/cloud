# Brain mirror — read-side plan (frame-grounded continuity, transcript truth, always-present mirror)

**Scope.** Read-side consumers of the per-frame perceptual descriptor. **Phase 1
(the L1 descriptor stage) is already shipped to `main`** (`frame_descriptors`
table + stage, commit `5428c5b`) and is intentionally EXCLUDED here. This plan
codes directly against that shipped interface. READ-ONLY handoff — implement from
another chat.

**Goal.** Make the brain's self-view a truthful, always-present mirror of the
edit it actually built — per-seam visual continuity judged from real frame
content, speech presence judged from the transcript, and a small flag vocabulary
grounded in real editing principles — while removing four band-aids the red-team
audit flagged.

**Two user decisions baked in:**
1. **Keep pHash** (do not swap to learned embeddings).
2. **Reground the flag vocabulary in editing principles** (§5 below).

---

## 0. The four band-aids being retired (and their generic replacements)

| # | Band-aid (current/earlier-draft behavior) | Generic replacement (this plan) |
|---|---|---|
| **A** | **Fixed global Hamming threshold** `_PHASH_SAME_BITS ≤ 8` decides "same shot" | **Content-relative, graded verdict** — distance is scored against a per-project adaptive scale into `seamless / same-shot-jump / cut`, no single magic cutoff (§2.2) |
| **B** | **`file_id` hard gate** — different file ⇒ always `cut`; same file ⇒ heuristic | **Pixels decide.** file_id is dropped as an arbiter; source-contiguity kept only as a cheap fast-path for genuinely adjacent same-source spans (§2.2) |
| **C** | **Blanket auto-mute** of incidental speech under picture cuts | **Gated mute** (speech vs music/ambience) with **words ALWAYS surfaced** — never silently muted (§3) |
| **D** | **Full assembled transcript always-on** in `read_state` | **Bounded distilled transcript** in the always-present mirror + an **on-demand full-text** sense (§3.2, §4) |

Plus the surfacing change: the mirror is **pushed every step** and the one-shot
finish-gate is removed (§4).

---

## 1. Prerequisites (operational, from the L1 push — do these first)

The read-side **fails open** (no descriptor ⇒ no false alarm), so it is safe to
land before these, but continuity stays "dark" until:

1. **Apply migration `055_frame_descriptors.sql`** to the shared/prod DB.
2. **Backfill re-ingest** existing files so `frame_descriptors` populates (the new
   stage is guarded by `processing_jobs` idempotency, so this only runs the one
   new stage per file — a targeted backfill, not a full re-analysis).

**Shipped L1 interface this plan consumes** (`backend/app/services/l1/frame_descriptors.py`):
- `load_descriptors(conn, file_ids) -> Dict[str, List[Tuple[int, int]]]` — bulk
  `{file_id: [(t_ms, phash_int), …]}` sorted by `t_ms`; missing files absent.
- `load_phash_at(conn, file_id, t_ms, *, nearest_ms=600) -> Optional[int]` —
  nearest sampled frame's hash; `None` ⇒ fail open.
- `nearest_phash(samples, t_ms, *, nearest_ms=600) -> Optional[int]`,
  `hamming64(a, b) -> int`.
- Table `public.frame_descriptors`: per-file jsonb `phashes` `[{t_ms, h}]`, `h` =
  16-hex 64-bit DCT pHash, sampled at **2 fps** (`hop_ms=500`).
- Empirical calibration from the L1 unit tests: **identical ≈ 0 bits, visually
  similar ≈ 6 bits, clearly different ≈ 36 bits.**

---

## 2. Phase 2 — Seam continuity from frame content (replaces band-aids A + B)

### 2.1 Current-state (file:line)

- `feel.join_continuity(cuts)` (`feel.py:208-240`) classifies each seam from
  `file_id` + `src_in_ms`/`src_out_ms` **only**:
  - different clip ⇒ `cut` (`:226-227`) — **band-aid B**;
  - same clip, `|gap| ≤ _JOIN_CONTIG_MS` ⇒ `seamless` (`:230-231`);
  - same clip, gap beyond tol ⇒ `jump` (`:232-233`), `backward` at `:238` —
    **false-positive source** (one screen-recording take holds logo/talking-head/
    slides/demo, so every content change trips a false jump).
- `_JOIN_CONTIG_MS = 500` (`feel.py:35`). `CutFeel.src_in_ms/src_out_ms`
  (`feel.py:48-49`) are populated in `simulate` (`feel.py:133-145`) — the data
  path already carries source spans; it lacks only the *picture*.
- Consumers (must keep working): `FeelReport.narrate` (`feel.py:90-93`),
  `observe.read_state`, `observe.diagnose`, `observe.review`,
  `tools._verify_before_finish` Stage 2.5 (matches verbatim `"same-shot jump"`).

### 2.2 Design — a graded, content-relative verdict (no fixed threshold, no file_id gate)

Rewrite `join_continuity` to compare the **actual out-frame of the previous
segment vs the in-frame of the current segment** over the *current resolved
timeline*, keeping `feel` pure (descriptors passed in, no DB inside):

```python
def join_continuity(
    cuts: List[CutFeel],
    descriptors: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    *, nearest_ms: int = _JOIN_NEAREST_MS,
) -> List[dict]:
```

For each adjacent pair `(prev, cur)`:

1. **Fast-path (cheap, not an arbiter):** same file AND source-contiguous
   (`|cur.src_in_ms − prev.src_out_ms| ≤ _JOIN_CONTIG_MS`) ⇒ `seamless`.
   Continuous same-source playback is one shot by construction — no frame lookup
   needed. **This is the ONLY remaining use of file_id, and only as an
   optimization**, never to *deny* a match (band-aid B retired).
2. **Otherwise, look up the two seam frames** via
   `nearest_phash(descriptors[prev.file_id], prev.src_out_ms, nearest_ms=…)` and
   `nearest_phash(descriptors[cur.file_id], cur.src_in_ms, …)`. This runs
   **regardless of whether the two segments share a file** — a cross-file match
   can read `seamless`, and a same-file content change reads `cut`.
   - **Either side missing ⇒ fail open to `cut`** (never a false jump).
   - **Both present ⇒ grade** `d = hamming64(hA, hB)` (0..64) with the
     content-relative scorer below.

**Content-relative grading (replaces band-aid A).** Instead of one global cutoff,
map the distance through a small graded band, calibrated by the L1 empirical
anchors (~0 identical / ~6 similar / ~36 different) and made relative to the
project so it adapts to noisy vs clean footage:

- Compute a per-timeline **scale** once: `ref = median(all inter-seam distances
  for same-file non-contiguous pairs present in descriptors)`, floored/ceiled to
  a sane range. This turns "how far apart" into "far *for this piece*."
- Verdict bands (named constants in `feel.py`, each with a one-line rationale —
  no bare magic numbers inline):
  - `d ≤ _PHASH_SEAMLESS_BITS` (≈4) ⇒ **`seamless`** — same framing, negligible
    change (a match cut, or a same-shot resume the brain placed).
  - `_PHASH_SEAMLESS_BITS < d ≤ _PHASH_SAMESHOT_BITS` (≈4..12), **and same file**
    ⇒ **`same-shot jump`** — the real jump-cut (same continuous framing, source
    out of order, e.g. clip93 408→209). Cross-file in this band is a **`cut`**
    (a deliberate match-adjacent join, not a jump).
  - `d > _PHASH_SAMESHOT_BITS` ⇒ **`cut`** — picture genuinely changed
    (talking-head → slide inside one screen recording ⇒ NO false jump).
- The bands are **soft-anchored**: expose them as tunables and let the project
  `scale` widen/narrow the middle band, so a grainy handheld clip isn't punished
  by the same absolute bits as a clean screen capture. (Recommend starting values
  above; validate against the two regression fixtures in §6.)

Each verdict dict keeps its existing keys (`from_pos`, `to_pos`, `kind`,
`same_clip`, `src_gap_ms`, `backward`) and gains **`pic_bits`** (the Hamming
distance, or `None` when unavailable) and **`pic_scale`** so the mirror can
explain *why* a seam reads the way it does.

**Naming.** `kind` values become the editing-grounded set from §5
(`seamless` / `jump-cut` / `cut`), with `same-shot jump` narration preserved for
back-compat with `tools._verify_before_finish`'s verbatim string (or update that
string in lockstep — see §4).

### 2.3 CutFeel + narrate

- `CutFeel` already carries `src_in_ms`/`src_out_ms` (`feel.py:48-49`) — no new
  fields needed for the lookup. (Confirm they hold the *source* instant at the
  seam edge, which `simulate` already sets.)
- `narrate` (`feel.py:90-93`): keep the same-shot-jump clause; add a
  **`jump-cut` clause** for same-file content-changed seams that are still visible
  jumps, and word it in editing terms (§5) rather than "out of source order."

### 2.4 Data path to descriptors at classify time

- `observe.build_context` loads descriptors **once** for every file in play via
  `frame_descriptors.load_descriptors(conn, ctx_file_ids)` and stashes them on
  `EditContext` (new field `frame_descriptors`).
- `feel.simulate(timeline, meta_by_ref, descriptors=None)` accepts + stashes
  `descriptors` on `FeelReport` so `narrate` can call `join_continuity(self.cuts,
  self.descriptors)`.
- Update `read_state` / `diagnose` / `review` call sites to pass
  `ctx.frame_descriptors`.

### 2.5 Re-ingest / versioning

- **Read-side only** — no re-ingest beyond the Phase-1 descriptor backfill (§1).
- `feel`/`observe` outputs are computed live per turn (not cached under
  `TREE_VERSION`/`CUTRECORD_MAP_VERSION`), so no bump here.

---

## 3. Phase 3 — Transcript-truth speech labeling (replaces band-aid C)

### 3.1 Current-state (file:line)

- `cutrecord_map._audio_mute_for(channel, row)` (`cutrecord_map.py:476-488`):
  `channel=="said"` ⇒ unmuted; otherwise mute iff `pace.natural_sound` is false —
  **blind to whether words actually play under the cut** (the slide-voiceover
  mislabel).
- `footage_map.build_clip_tree` gates the verbatim words on channel:
  `said_text = … if cut.get("channel") == "said" else ""` (`footage_map.py:281`)
  — incidental speech under a `shown` cut is never surfaced.
- Whisper words live in `dialogue_segments` (L1, persisted). `_said_text_for_span`
  already reads them.

### 3.2 Design — presence from the transcript; mute is a gated decision; words always shown

- **Read helper** (`cutrecord_map` or a small L1 read util):
  `speech_words_in_span(file_id, in_ms, out_ms) -> (word_count, has_speech,
  is_musical)` — overlap `dialogue_segments` word timings against
  `[src_in_ms, src_out_ms]`. `is_musical` distinguishes lyric/music/ambience from
  spoken narration using the available audio-type signal (diarization/ASD /
  `natural_sound` / segment tag) — this is the **gate** that replaces blanket
  auto-mute.
- **`_audio_mute_for`:** before the `natural_sound` branch, if the span carries
  transcript words (`has_speech`):
  - the cut is **speech-bearing** regardless of `channel`;
  - **mute is gated, not automatic** — mute only when the words are incidental
    AND *musical/ambient* (`is_musical`) OR the picture cut's audio is explicitly
    unwanted; **spoken narration under a picture cut is NOT auto-muted** (that was
    the band-aid that ate the founder's voiceover). Emit flags
    `["speech", "incidental"]` (and `"muted"` only when actually muted) so the
    brain sees the choice and can override.
- **`footage_map.build_clip_tree` (`:281`):** drop the `channel=="said"` gate —
  compute `said_text` whenever the span has transcript words, so the beat index
  shows the brain the words that *will* play under every cut, muted or not.

### 3.3 Bounded transcript, not always-on full text (replaces band-aid D)

- Do **not** dump the full assembled transcript into `read_state` every turn.
  Instead the always-present mirror (§4) carries a **bounded, distilled** speech
  view: per placed segment, a short head/tail of the spoken text + `muted` +
  boundary flags, and a one-line `narrative` gist.
- Add an **on-demand full-text sense** (extend an existing `inspect`/`review`
  verb, or a new `read_transcript` sense) that returns the complete program-order
  transcript with verbatim words when the brain asks. Keeps the every-step mirror
  cheap while never hiding the script.
- **Boundary/broken-line detection** (flags, not mutations): mark a segment
  `clipped_head`/`clipped_tail` when the first/last overlapping word straddles the
  cut edge (`word.start_ms < in_ms` or `word.end_ms > out_ms`), and `mid_sentence`
  when the boundary text lacks sentence-final punctuation. These feed §5's
  `broken-line` flag.

### 3.4 Re-ingest / versioning

- **Read-side, NO re-ingest** — `dialogue_segments` already persisted.
- Bump **`CUTRECORD_MAP_VERSION` 5 → 6** (`cutrecord_map.py:59`) and
  **`TREE_VERSION` 22 → 23** (`footage_map.py:121`) so cached ladders / beat-index
  rows recompute with the new labeling + surfaced words.

---

## 4. Phase 4 — Always-present mirror (push model) + gate removal

### 4.1 Current-state (file:line)

- `read_state` per-cut rows at `observe.py:519-589`; the join is attached only
  when not a plain `cut` (`observe.py:580-587`).
- Surfacing today is pull + a **one-shot finish-gate**: `diagnose`
  (`observe.py:1198-1201`) and `tools._verify_before_finish` Stage 2.5
  (`tools.py:694-704`) which matches the verbatim `"same-shot jump"` string and
  fires at most once via `state["continuity_surfaced"]` (`tools.py:698`).

### 4.2 Design — the mirror is pushed at every step

- Build a compact **mirror** projection (distinct from the full `read_state`
  pull): ordered placed segments, **join-to-next verdict** (`seamless` /
  `jump-cut` / `cut` with `pic_bits`), **speech presence** (has-words / muted),
  and the small **flag set** from §5. Include the §3.3 bounded speech gist.
- **Inject the mirror into the brain's reasoning context on every turn** (push
  model), not only when a sense is called — the brain should never reason blind
  about the edit it just changed. Wire this where the per-turn context/system
  content is assembled for the loop (the `converse`/`tools` turn builder).
- **Remove the special finish-gate.** Because the mirror (including red flags) is
  always present, `_verify_before_finish` Stage 2.5's one-shot surface
  (`tools.py:694-704`) and `state["continuity_surfaced"]` (`tools.py:737`) are no
  longer needed to *reveal* problems. Delete the gate's revealing role; if a
  terminal sanity check is still wanted, it should only *read* the always-present
  flags, not be the sole place they appear. (Retire the verbatim
  `"same-shot jump"` string dependency by updating it to the §5 vocabulary in the
  same change.)
- Keep loop termination safe: with red flags always visible, no re-push counters
  are required, but ensure the mirror is **idempotent** (same state ⇒ same text)
  so it doesn't thrash the loop.

### 4.3 Re-ingest / versioning

- Read-side, none. Render changes that flow through `footage_map` are covered by
  the §3 `TREE_VERSION` bump.

---

## 5. Flag vocabulary regrounded in editing principles (user decision)

Replace ad-hoc flag names with a small, principled set. Each flag = a named
editing situation, its trigger (mapped to the §2 pHash verdict and §3
speech-overlap), and what it warns the brain about. Put the definitions in
`guidance_doc.md` and use the same names in `feel`/`observe`/mirror output.

| Flag | Editing principle | Trigger (from real signals) |
|---|---|---|
| `match-cut` / `seamless` | Continuity editing — the join is invisible | §2 `seamless` band (contiguous same-source, or `pic_bits ≤ _PHASH_SEAMLESS_BITS`) |
| `jump-cut` | Jarring same-subject jump; use only intentionally | §2 same-file, mid band ⇒ `same-shot jump`; visible discontinuity within one framing |
| `cut` (motivated) | A normal, motivated cut to new content | §2 `pic_bits > _PHASH_SAMESHOT_BITS`, or cross-file non-matching |
| `broken-line` | Never cut mid-sentence / mid-word (audio integrity) | §3.3 `clipped_head`/`clipped_tail`/`mid_sentence` on a **speech-bearing** seam |
| `orphan-audio` / `incidental` | Incidental speech playing under a picture cut | §3.2 `has_speech` under `channel != "said"`; `muted` sub-state shows the gate's choice |
| `dead-air` | A sag / low-energy lull | existing `feel` low-energy / gap derivations |
| `dangling` (optional, L/J) | Audio lead/lag across a cut (L-cut/J-cut) | speech words straddling the seam edge (from §3.3), when kept rather than trimmed |

- Update `guidance_doc.md`: add/confirm a **"Keep a continuous shot continuous"
  (sacred runs)** section — parts of one continuous source run should stay
  together and in source order unless deliberately broken — and a short glossary
  of the flags above so the brain reasons in editing terms, not signal jargon.
- Keep the existing trim-policy paragraph; ensure the new vocabulary is
  cross-referenced there.

---

## 6. Test plan (unit, mocked, no real spend)

**Phase 2 — seam classifier** (synthetic `CutFeel` + a fake `descriptors` dict):
- same clip, contiguous ⇒ `seamless` (no descriptor lookup needed).
- same clip, non-contiguous, **similar** frames (small Hamming) ⇒ `jump-cut`
  (clip93 408→209 backward case still flagged).
- same clip, non-contiguous, **different** frames (large Hamming) ⇒ `cut`
  (talking-head→slide inside one file ⇒ **no false jump**; the ~7-flag flood dies).
- **cross-file, similar** frames ⇒ `seamless`/`cut` per band (NOT blocked by a
  file_id gate) — proves band-aid B is gone.
- descriptor missing on either side ⇒ **fail-open `cut`**.
- content-relative scale: the same absolute bits yield different verdicts under a
  noisy vs clean project `scale` — proves band-aid A (fixed cutoff) is gone.
- Regression fixtures: `7144c9fc` multi-content take ⇒ **zero** jump-cuts;
  `57a517fd` clip93 backward seam ⇒ `jump-cut`.

**Phase 3 — speech truth + gated mute + bounded transcript:**
- a `shown` cut whose span overlaps `dialogue_segments` words ⇒ speech-bearing:
  `said_text` populated in the beat index; **spoken** narration ⇒ NOT auto-muted;
  **musical/ambient** words ⇒ muted with `incidental` flag.
- a genuinely silent `shown` cut ⇒ unchanged.
- mirror carries a **bounded** speech gist; the on-demand full-text sense returns
  the complete program-order transcript.
- mid-word boundary ⇒ `clipped_tail`/`clipped_head`; mid-sentence edge ⇒
  `mid_sentence`. Reconstruct `720713d9` read-only ⇒ `broken-line` fires on the
  fragmented slide spine.

**Phase 4 — always-present mirror + gate removal:**
- the mirror is present in the turn context every step (not only on a sense call)
  and is idempotent for a fixed state.
- red flags (`jump-cut`, `broken-line`) appear in the mirror without the one-shot
  finish-gate; `_verify_before_finish` no longer the sole reveal path; loop still
  terminates.
- `narrate`/mirror use the §5 editing vocabulary; verbatim string dependency on
  `"same-shot jump"` updated in lockstep.

Run with `backend/.venv/bin/python`; keep everything mocked; changed files
pyflakes-clean.

---

## 7. Exact change list per file

**Phase 2**
- `backend/app/services/l3/feel.py` — rewrite `join_continuity` (`:208-240`) to the
  graded, content-relative, pixel-decided classifier; add
  `_PHASH_SEAMLESS_BITS`/`_PHASH_SAMESHOT_BITS`/`_JOIN_NEAREST_MS` + per-project
  scale; `descriptors` param on `simulate` (`:112`) + `FeelReport` field; `narrate`
  (`:90`) `jump-cut` clause. Import `nearest_phash`/`hamming64` from
  `l1.frame_descriptors` (keep `feel` pure — descriptors passed in).
- `backend/app/services/l3/observe.py` — `EditContext` gains `frame_descriptors`;
  `build_context` calls `load_descriptors`; pass `ctx.frame_descriptors` into
  `read_state`/`diagnose`/`review`.

**Phase 3**
- `backend/app/services/l3/cutrecord_map.py` — `speech_words_in_span` helper;
  `_audio_mute_for` (`:476`) gated speech-bearing branch; bump
  `CUTRECORD_MAP_VERSION` 5→6 (`:59`).
- `backend/app/services/l3/footage_map.py` — drop `channel=="said"` gate at `:281`;
  bump `TREE_VERSION` 22→23 (`:121`).
- `backend/app/services/l3/observe.py` — bounded speech gist for the mirror + the
  on-demand full-text sense; boundary/broken-line detection.

**Phase 4**
- `backend/app/services/l3/observe.py` — per-row join-to-next verdict + §5 flags
  (`:519-589`); red-flag list in `diagnose` (`:1198`).
- `backend/app/services/l3/tools.py` — remove the Stage 2.5 one-shot reveal
  (`:694-704`) and `continuity_surfaced` (`:737`); read the always-present flags
  instead; update the verbatim `"same-shot jump"` dependency to §5 vocabulary.
- `backend/app/services/l3/converse.py` — inject the always-present mirror into the
  per-turn context; update continuity wording (`~:163`,`~:183`) to describe the
  frame-based, editing-grounded verdict.
- `backend/app/services/l3/guidance_doc.md` — §5 flag glossary + "Keep a continuous
  shot continuous" (sacred runs) section.

---

## 8. Re-ingest & version-bump summary

| Phase | Re-ingest? | `CUTRECORD_MAP_VERSION` | `TREE_VERSION` |
|---|---|---|---|
| 1 (shipped, L1) | one-time descriptor backfill (§1) | — | — |
| 2 seam continuity | No (uses Phase 1) | — | — |
| 3 speech truth + transcript | No | **5 → 6** | **22 → 23** |
| 4 mirror + surfacing | No | — | (covered by Phase 3) |
