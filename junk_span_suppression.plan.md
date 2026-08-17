# Span usability grading — killing "stale clips between the shots"

Status: proposal. One new L1 primitive, one new persisted grade, three dead
branches made reachable. No new thresholds tuned to one shoot, no keyword
blocklists, no per-project special cases.

---

## 0. The complaint, made concrete

User: *"Some stale clips between the shots etc were also considered. That is a
big pain point."*

Newest edit thread `34bb9967-97aa-4818-9f21-01145156a9bb` (2026-08-17
09:06:54Z, `edit_documents.version=1`, `created_by='auto'`), ingest run
`24894c8a-9daa-4aa3-aea2-1f3b60c93478`. Six cuts placed, `resolved.duration_ms
= 24217`.

Three of the six placed cuts contain **zero transcript words and sit at the
file's own noise floor**:

| placed ref | src span (ms) | dur | words | L1 `silence_intervals` cover | median rms | prog time |
|---|---|---|---|---|---|---|
| `93e94ec3:m05` | 32500–33605 | 1105 | 0 | 100% | −86.95 dBFS | 9176–10281 |
| `93e94ec3:m06` | 45480–47310 | 1830 | 0 | 100% | −87.80 dBFS | 10281–12111 |
| `aad020e3:m06` | 236000–240600 | 4600 | 0 | 100% | −88.73 dBFS | 19617–24217 |

The other three placed cuts read −18.0, −31.4 and −36.2 dBFS with 86–98%
word coverage. The gap between "good" and "junk" here is **~50 dB**, not a
judgment call.

`document.resolved.audio_layers` shows all three silent spans are `kind:
"spine"`, `role: "dialogue"`, and `operations` is empty — no bed, no music
underneath. So **7535 ms of the 24217 ms delivered program (31.1%) is
digital silence**, including the final 4.6 s.

---

## 1. Part A — the junk, by class

### Class 1 — dead tail of a recording ("between the shots", the actual complaint)

The 244 s pitch-deck run ends at 244331 ms. Its last ~9 seconds are the
presenter finishing, moving, and reaching to stop the capture. Three pool
moments live there, all `junk=false`:

| ref | span | dur | words | silence | mean action energy | pHash consec (mean/max) | `total_quality` |
|---|---|---|---|---|---|---|---|
| `7144c9fc:m07` | 235100–239600 | 4500 | 0 | 100% | 0.26 | 3.33 / 14 | 0.172 |
| `7144c9fc:m08` | 240100–244300 | 4200 | 0 | 100% | 0.74 | 13.25 / 34 | **0.047** |
| `aad020e3:m06` | 236000–240600 | 4600 | 0 | 100% | 0.40 | 8.44 / 36 | 0.144 |

`7144c9fc:m08` ends 31 ms before EOF. VLM labels: *"The speaker appears on
camera."* / *"The speaker touches his chin while thinking."* — a plausible
description of a man who has stopped presenting.

`aad020e3:m06` is the one that shipped, as the closing 19% of the program.
The brain's own reasoning (turn 13, `edit_turns.trace`):

> "those are silent b-roll of the speaker — a good visual close. […] Let me
> add a clean human close: aad020e3:m06 (Vivek on camera, medium shot). This
> is silent (ambient). Placing it at the end gives a settled visual button."

It knew there was no sound. Nothing told it this was the end of a take
rather than a held shot.

### Class 2 — exact duplicate ingest of the same file

`7144c9fc` and `aad020e3` are the **same bytes**: identical `filename`
(`video1895834262.mp4`), identical `file_size` (12131679), identical
`duration_seconds` (244.330667). Cross-file pHash over all 489 shared 500 ms
samples: **mean Hamming 0.131 bits, max 2 bits, 100% within 2 bits.**

All 7 of `aad020e3`'s pool moments duplicate 7 of `7144c9fc`'s 9. The Beat
Index presents them as two separate `CLIP` blocks with the same filename and
duration, 13 lines apart. `footage_map.assemble_map(...)['dup_groups']`
correctly pairs the *speech* duplicates (`7144c9fc:m01`↔`aad020e3:m00`,
`7144c9fc:m02`↔`aad020e3:m01`) and **misses every video duplicate**,
including the pair that mattered (`7144c9fc:m07`↔`aad020e3:m06`).

### Class 3 — silent screen-recording with a hallucinated action label

`93e94ec3` has a 43.2 s silence run (`silence_intervals` entry
30787–73992 ms) covering the whole product demo. Two video moments come out
of it and both shipped:

- `93e94ec3:m05` 32500–33605, 0 words, −86.95 dBFS, label *"The user selects
  multiple video files"*.
- `93e94ec3:m06` 45480–47310, 0 words, −87.80 dBFS, label *"The user types a
  prompt into the AI assistant chat box"*. Its 5 consecutive pHash samples
  are **bit-identical** (consecutive Hamming mean 0.0, max 0) — the
  low-frequency frame structure does not change at all across the span.

These are defensible as cutaways *under speech*. They are not defensible as
`role: "dialogue"` spine segments, which is what they became — 2.9 s of dead
program audio mid-edit.

### Class 4 — already-flagged speech junk that is still 39% of the index

Three speech moments carry `junk=true, junk_reason='non_speech_energy'`:
`93e94ec3:m00` (6274–7565), `m01` (12355–13920), and `m11`
(89255–209180 — **119 925 ms, a two-minute "beat"** whose transcript over
that span is 40 words of which ~22 are "uh", word coverage 0.21).

### Pool arithmetic

31 moments in the pool (`assemble_map.moment_count = 31`):

- 3 flagged junk (speech gate fired) — 9.7%
- 3 unflagged dead-tail video moments — 9.7%
- 7 exact duplicates of another clip's moment (all of `aad020e3`) — 22.6%
- union, de-duplicated: **12 of 31 = 39% of the offered pool is junk or
  redundant.**

Delivered: **31.1% of the program is silence**; **19.0% is the dead tail of a
recording.**

---

## 2. Part B — root cause

### 2.1 Where moments come from

Live path is **vcut** (`ingest_runs.pass1_model = 'vcut:gemini-3.1-flash-lite'`;
entry `routers/projects.py:73` → `vcut.orchestrate.defer_vcut_ingest` →
`run_vcut_ingest`, `vcut/orchestrate.py:295`). Two independent channels:

- **video / `channel='shown'`** — Gemini Pass 1 emits bare `MomentFlag`
  peaks (`vcut/pass1.py`), `resolve.py` turns each peak into a span
  deterministically from the seam function `S(t)`, `store.build_cut_records`
  writes the row. **Fusion: VLM proposes the instant, code derives the span.**
- **speech / `channel='said'`** — `vcut/speech/*`: an LLM segments beats,
  `boundaries.py` snaps to word edges, `delivery.py` scores, `select.py`
  crowns a take winner.

Pass 1's own prompt (`vcut/pass1.py:150`) is explicit:

> "Mark as MANY moments as this file's footage holds -- do NOT group them,
> **filter them, or judge which are good; that happens later**."

**"Later" never happens for the video channel.** That is the whole bug.

### 2.2 The gate that exists and cannot fire

`vcut/resolve.py:496-513`:

```496:513:backend/app/services/vcut/resolve.py
def _drop_dead_air(cands: List[_Candidate], hop_ms: int, S: List[float]) -> List[_Candidate]:
    """Drop any candidate whose median S over its extent is below
    S_DEAD_FRAC * (clip median S) AND which carries no peak. Every
    candidate this module builds carries >=1 flag by construction (nothing
    is ever dropped/selected, section 1's own invariant) -- this floor is a
    defensive invariant (see test_vcut_resolve.py, which exercises it
    directly against a synthetic peak-less candidate) rather than a path
    real Pass-1 output can trigger."""
    if not S:
        return cands
    clip_median = _median(S)
    if clip_median <= 0:
        return cands
    floor = S_DEAD_FRAC * clip_median
    return [
        c for c in cands
        if c.has_peak or _median_s_in_range(c.a, c.b, hop_ms, S) >= floor
    ]
```

`has_peak: bool = True` is the dataclass default at `vcut/resolve.py:173`.
**Nothing in `resolve.py` — or anywhere in `backend/app` — ever sets it
`False`.** The only `has_peak=False` in the repository is
`scripts/test_vcut_resolve.py:243` and `:247`, hand-built synthetic
candidates. So `c.has_peak` short-circuits the `or` on every real candidate
and `_median_s_in_range` is never evaluated in production. The docstring
says so out loud.

**This gate has a passing test and is unreachable code.** It is the third
instance of the pattern the user has been burned by.

**And the threshold is correct.** Re-running it by hand over all 26 spans of
this run with the module's own `_median`, `S_DEAD_FRAC = 0.35`
(`vcut/params.py:75`):

| span | median S | clip median S | floor | verdict |
|---|---|---|---|---|
| `aad020e3` 236000–240600 (**shipped**) | 0.2241 | 0.6533 | 0.2287 | **FAIL** |
| `7144c9fc` 235100–239600 | 0.2259 | 0.6503 | 0.2276 | **FAIL** |
| `7144c9fc` 240100–244300 | 0.0000 | 0.6503 | 0.2276 | **FAIL** |
| `7144c9fc` 0–2400 (EDSO logo card) | 0.6385 | 0.6503 | 0.2276 | pass |
| `7144c9fc` 45100–48800 (slide) | 0.4909 | 0.6503 | 0.2276 | pass |
| all 8 other good video spans | 0.49–1.04 | — | — | pass |

It catches **exactly the three dead-tail spans and nothing else**. Ranked by
`support = median_S / clip_median_S` the whole pool splits at a real gap:
`0.00, 0.00, 0.34, 0.35 | 0.51, 0.51, 0.54, 0.55, 0.65, …`. `S_DEAD_FRAC =
0.35` sits inside that gap. The margin on `7144c9fc:m07` is 0.7%, which is
why `support` must be *one axis of a fusion*, not the sole gate.

### 2.3 Two more hardcoded defaults that blind the read side

**(a) `junk` is hardcoded false for every video cut.** `vcut/store.py:135`:

```131:141:backend/app/services/vcut/store.py
        records.append(CutRecord(
            file_id=cut.file_id, src_in_ms=cut.in_ms, src_out_ms=cut.out_ms,
            kind="video", word_span=None, atom_ids=None,
            label=_short_label(cut.summary), summary=cut.summary,
            on_camera=None, junk=False, junk_reason="",
```

Every downstream junk consumer therefore never fires for video:
`observe.py:1821-1826` (`cutaway_pool` filters `not m.get("junk")`),
`footage_map.py:1110` (the terse `[JUNK: reason]` beat line),
`cutrecord_map.py:682`. `junk` is a correct, reversible presentation gate
with **no producer on the video channel**.
`l3/pass2.apply_junk_suspects` (`l3/pass2.py:415`) is the old non-vcut
pipeline and is dead for this project.

**`junk_confidence` is a fully dead column.** Its only occurrence in
`backend/app` or `frontend` is the SELECT list at `cuts_read.py:60`. No
module writes it (every row is at the migration default `'low'`), and no
module reads it — it is dropped in `cutrecord_map._to_cut_dict` and never
reaches the moment tree, the Beat Index, or the frontend. Migration
`026`'s `'high'`→hidden / `'low'`→visible "if in doubt, show" contract
exists only in SQL. **Any fix that writes `junk_confidence` and stops
there ships inert** — see §3.3.

**(b) `natural_sound` is hardcoded true, so `SND:silence` is unreachable.**
`vcut/store.py:80-81`:

```80:81:backend/app/services/vcut/store.py
    return PaceEnvelope(min_ms=natural, natural_ms=natural, max_ms=natural,
                        levels=list(_PACE_LEVELS), energy_grade="calm", natural_sound=True)
```

`cutrecord_map.py:620-623` is the only producer of the `"silent"` audio state:

```620:623:backend/app/services/l3/cutrecord_map.py
    natural_sound = bool((row.get("pace") or {}).get("natural_sound"))
    if natural_sound:
        return "sound", False, []
    return "silent", True, ["muted"]
```

Confirmed in the data: all 31 rows carry `pace.natural_sound = true`. So
`footage_map.py:969`'s `_AUDIO_STATE = {"speech": "talk", "sound":
"ambient", "silent": "silence"}` can never emit `silence` on the vcut path.
**Digital silence and real room tone both render as `SND:ambient`.** In the
Beat Index the dead tail and the title card are indistinguishable on the
sound axis:

```
m00 shown.object PIC:object (q.61) SND:ambient [0:00-0:02 2.4s] "The EDSO logo is displayed on"
m07 shown.object PIC:object (q.17) SND:ambient [3:55-3:59 4.5s] "The speaker appears on camera."
```

### 2.4 Signals that already exist and are not consulted

| L1 signal | stored | value over `aad020e3:m06` | consulted at pool build? |
|---|---|---|---|
| whisper words | `transcripts.segments[].words` | **0 words** | ✗ (`cutrecord_map.speech_words_in_span` is read-side only) |
| L1 silence intervals | `audio_features.silence_intervals` | **100% of span** | ✗ |
| loudness envelope | `audio_features.rms_db`, `seam_cache[f].rms_db` | median **−88.73 dBFS**, = file's own 15th-pctl floor | ✗ for video (speech gate uses it) |
| seam function `S(t)` | `ingest_runs.seam_cache[f].S` | support **0.34** vs 0.35 floor | gate exists, **unreachable** |
| optical flow | `motion_dynamics.action_energy` (100 ms hop) | mean 0.40, peaks 1.0 | used for peak refinement only, not usability |
| per-frame pHash | `frame_descriptors.phashes` (present for **114 files**, 500 ms hop) | consec Hamming mean 8.44 / max 36 | ✗ at pool build |
| face tracks | `face_tracks` | **no row for any of the 3 files** | n/a — must degrade gracefully |
| ASD / diarization | `transcripts.speaker_embeddings`, `dialogue_segments` | — | ✗ |

There is also a *latent* VLM opinion: Pass 2's question bank includes
`usable` — *"how usable this take/moment is as a shot: strong | ok | weak"*
(`vcut/questions.py:55-56`). It is display-only (`footage_map.py:828-830`,
no pool effect) and, on this run, **was never even collected**: none of the
14 video rows' `scene_specifics` contains a `usable` key. This is an
argument against reaching for it — a per-clip-selected LLM opinion that may
or may not be populated cannot be the substrate for a usability gate. The
deterministic signals below are present for every file.

**pHash is the starkest waste.** It is computed for every file
(`l1/pipeline.py:931`) and read by exactly one consumer: `l3/feel.py`
(`hamming64`, `nearest_phash`), and only to classify the *seam between two
already-placed cuts* as `seamless` / `jump-cut` / `cut`
(`feel.py:353-357`). It is never used to characterize a span's own content
(frozen frame) or to detect duplicate material in the pool — the two things
that would have caught Class 2 and Class 3 outright.

### 2.5 Pool / presentation / selection — it is all three, in that order

- **Pool (primary).** `aad020e3:m06`, `7144c9fc:m07`, `7144c9fc:m08` should
  never have been in `affordances()['cutaway_pool']`. Reproduced read-only
  against the live run: `cutaway_pool` is
  `['93e94ec3:m05', '93e94ec3:m06', '7144c9fc:m00', '7144c9fc:m03' … 'aad020e3:m06']`
  — 14 refs, and it is a **bare list of ids with no label, no duration, no
  quality**. The gate that would have removed the three dead ones is
  unreachable (§2.2); the `junk` flag that `cutaway_pool` filters on is
  hardcoded false (§2.3a).
- **Presentation (severe).** A quality number *does* reach the model as
  `q.NN` in the Beat Index — `q.14` for `aad020e3:m06`, `q.17`, `q.05` for
  the two twins. But `_qual` (`footage_map.py:953-955`, reached from
  `_pic_segment` at `:958-964` on the moment's `score`, which is
  `total_quality`) renders `total_quality * 100` for **both** channels, and
  video `total_quality` ∈
  [0, 1] (`vcut/store.py:127`, `max(0.0, min(1.0, mean_s))`) while speech
  `total_quality` ∈ [0, 4.5]. So the same tag means `q.05…q.100` for video
  and `q.201…q.450` for speech, and the prompt's only explanation is
  `converse.py:309`: *"PIC's q.XX is that visual score."* — **no range, no
  units, no cross-channel warning.** A model reading `q.14` next to `q.450`
  has no way to know `q.14` is near the bottom of its own scale.
  On the sound axis it is worse than uninformative: `SND:ambient` (§2.3b).
  Even `inspect_cut`, the deepest tool, min-max normalizes the loudness
  curve against the whole clip (`observe.py:798-799`,
  `_series_lohi` → `_norm_in_clip`), so −88.7 dBFS renders as a **flat
  0.289–0.307 curve** that reads as "quiet but present"; only the separate
  `silence: [{off: 0, dur: 4600}]` entry gives it away, with no absolute dB
  anywhere.
- **Presentation, worse: the system prompt states a guarantee that is false
  on the live path.** `converse.py:284-287`, in the `_PROVENANCE` block the
  brain reads as ground truth about how its material was made:

  > "A stretch that is purely static, silent, and motionless -- including a
  > stretch of otherwise-generous padding that itself carries no energy --
  > **is dropped entirely rather than becoming a cut of its own**, or padded
  > into, at all; **nothing manufactures a cut out of dead footage.**"

  That describes `l3/v4_segment.py`'s `DEAD_ENERGY_FLOOR = 0.15`
  (`v4_segment_params.py:147`) — the **v3 segmenter, which is dead code**
  under `cuts_pipeline = "vcut"` (`config.py:287`). On the vcut path
  nothing enforces it: `spans.py:81` admits *any* non-speech gap ≥
  `MIN_NONSPEECH_SPAN_MS = 500` (`vcut/params.py:13`) into Pass 1's domain,
  including inter-take dead space, and §2.2 shows the only floor downstream
  is unreachable. So the brain is **told that the junk in front of it cannot
  exist.**
- **Selection (real, but downstream and entailed).** The brain called
  `inspect_cut` on 4 refs and **never on `aad020e3:m06`** — it placed it
  from the Beat Index line alone (trace step 34). Its turn-13 reasoning read
  `SND:ambient` + "speaker appears on screen" + "medium shot" and concluded
  *"silent b-roll […] a good visual close."* Given a prompt that guarantees
  dead footage was already dropped, "silent + on-camera + medium shot" can
  only mean a deliberate held shot. **That inference is correct reasoning
  from a false premise.** Fixing selection alone would be a band-aid; the
  premise has to stop being false.

---

## 3. Part C — the fix

### 3.1 Layering decision: **grade at emission, never drop. Recommended: (b).**

Emit every moment. Attach a factual, multi-axis **usability grade**. Drive
the *existing reversible presentation gate* (`junk` / `junk_confidence`)
from that grade.

Why not (a) suppress at emission:

1. **It is irreversible and the pipeline is built on the opposite
   invariant.** `l3/pass1.py:931-933`: "the ONLY way something disappears is
   an explicit, recoverable junk label." `post.py:640` speaks of the
   "full-coverage invariant." Dropping a span at emission breaks
   `cut_no/of` numbering, whose *gaps* are themselves the signal that a junk
   beat sits there (`converse.py:333`, `post.py:453`).
2. **The signals cannot distinguish worthless from intentional.** Measured
   on this run, the legitimate EDSO title card (`7144c9fc:m00`, 0–2400) and
   the dead tail (`7144c9fc:m07`) are *identical* on the axes that matter
   most: word coverage 0.00 vs 0.00, silence 100% vs 100%, loudness at the
   file's own floor (`0.01` vs `−0.00` normalized) . They separate only on
   `support` (0.98 vs 0.35) and stillness (1.00 vs 0.67). A single-axis
   suppressor would have deleted the title card. An intentional silent
   beauty shot would fare worse.
3. The three offending spans in this run miss the dead-air floor by 0.7%,
   1.9% and 100%. A hard drop on a 0.7% margin is not something to ship.

Why not (c) both: "suppress *and* grade" collapses to (a) for anything the
suppressor catches, inheriting every problem above, and doubles the number
of places that decide what is junk — which is how the current three
hardcoded defaults happened.

Grading gets suppression's benefit for free, reversibly: the repo already
has the machinery — `junk_confidence='high'` hides by default while the ref
stays placeable (`migrations/026_junk_confidence.sql`: *"if in doubt, show"*),
`cutaway_pool` filters junk (`observe.py:1825`), and the Beat Index still
lists a junk beat as a one-liner so numbering stays honest
(`footage_map.py:1110`). **Nothing new needs inventing on the read side. It
needs a producer.**

### 3.2 New L1 primitive: `l1/span_usability.py`

Lives in L1, not vcut, because it is a pure function of L1 signals over an
arbitrary span and must be callable by both channels and by the backfill.

```python
@dataclass(frozen=True)
class SpanUsability:
    grade: str            # "carries" | "silent_hold" | "dead" | "unknown"
    voice: Optional[float]    # 0..1  spoken-word time coverage of the span
    audible: Optional[float]  # 0..1  (median_rms - floor_pctl) / (voiced_pctl - floor_pctl)
    support: Optional[float]  # 0..∞  median_S(span) / median_S(clip)
    still: Optional[float]    # 0..1  frac of consecutive pHash pairs within the seamless band
    motion: Optional[float]   # 0..1  mean normalized action_energy
    edge: str                 # "head" | "tail" | ""
    why: Tuple[str, ...]      # the axes that decided `grade`
    inputs: Tuple[str, ...]   # which signals were actually available
```

`compute(span, bundle) -> SpanUsability` where `bundle` is the per-file L1
projection **that already exists**: `seam_cache[file_id]` (`S`, `rms_db`,
`rms_hop_ms`, `action_energy`, `hop_ms`, `silence_intervals`) plus
transcript words, `frame_descriptors.phashes`, and `duration_ms`.

Every axis is normalized against **the file's own distribution**. No
absolute dB, no absolute flow magnitude, no pixel counts:

- `voice` — sum of word-interval overlap / span duration. `None` if the file
  has no transcript.
- `audible` — reuse the speech channel's already-shipped self-calibrating
  references verbatim: `RMS_VOICED_PCTL = 75`, `RMS_FLOOR_PCTL = 15`,
  `MIN_VOICED_SPREAD_DB = 8.0` (`vcut/speech/params.py:78-81`). `None` when
  the spread is too small to discriminate — the same fail-open
  `delivery.is_voiced_beat` already uses. **Do not introduce a second
  loudness calibration.**
- `support` — `resolve._median_s_in_range(span) / _median(S)`, i.e. exactly
  the quantity `_drop_dead_air` computes today. Compared against
  `S_DEAD_FRAC` (`vcut/params.py:75`), unchanged.
- `still` — fraction of consecutive pHash pairs within the seamless band.
  Reuse `feel.py`'s already-calibrated, project-relative bit bands
  (`_PHASH_SEAMLESS_BITS = 4.0`, `_PHASH_SAMESHOT_BITS = 12.0`,
  `feel.py:47-48`, scaled by `_project_scale`); do not invent a new Hamming
  threshold. `None` with fewer than 2 samples.
- `motion` — mean `action_energy` over the span (already normalized 0..1 at
  L1).
- `edge` — `"head"` / `"tail"` when the span starts/ends within
  `EDGE_FRAC * duration_ms` of the recording boundary, `EDGE_FRAC` a single
  scale-free constant. Positional evidence only; never decisive on its own
  (title cards live at the head).

Grade rule — three clauses, each a conjunction of *independently
unambiguous* facts:

```
unknown       if voice is None and audible is None and support is None
dead          if not_voiced and not_audible and support < S_DEAD_FRAC
silent_hold   if not_voiced and not_audible                      (support normal)
carries       otherwise

not_voiced  := voice is None or voice <= VOICE_EPS
not_audible := audible is None or audible <= AUDIBLE_EPS
```

`VOICE_EPS` / `AUDIBLE_EPS` are near-zero epsilons on already-normalized
quantities ("no words at all", "at this file's own noise floor"), not tuned
magnitudes. Measured on this run: the junk trio reads `voice = 0.00`,
`audible ∈ {−0.00, 0.00, −0.00}`; the worst good span reads `audible = 0.06`
and every voiced span reads `voice ≥ 0.16`.

**Graceful degradation is the default, not a fallback.** Any missing signal
makes its axis `None`; `None` never satisfies a `dead` clause. A file with no
`frame_descriptors` row, no `face_tracks` row (as here — none of the three
files has one), or a flat loudness profile grades `carries` or
`silent_hold`, i.e. today's behaviour. `inputs` records what was available so
a grade is never mistaken for a measurement it did not make.

### 3.3 Storage and flow to the brain

**Migration `056_span_usability.sql`** — additive:

```sql
alter table public.cut_records
    add column if not exists usability jsonb not null default '{}';
```

**Write side — one call site per channel, no third implementation:**

- `vcut/store.build_cut_records` (`vcut/store.py:120`) — compute per cut;
  set `usability`, and derive from it:
  - `grade == "dead"` → `junk=True`, `junk_reason="dead_span"`

    **Deliberately not `junk_confidence="high"`.** That column has no reader
    anywhere (§2.3). Writing it would add a fourth "value set, nothing
    consumes it" to a codebase that already has three, and would create a
    second, weaker source of truth for the same question. `usability.grade`
    *is* the confidence signal — it carries the deciding axes in `why` and
    degrades to `unknown` rather than guessing. **Sub-task, tracked
    separately, either/or, not both:** wire `junk_confidence` end-to-end
    (`_to_cut_dict` → moment dict → `_moment_line` → the frontend's default
    hide) *with* an assertion on the rendered output, **or** drop the column
    in a follow-up migration. Leaving it half-alive is what produced this
    bug class.
  - `grade == "silent_hold"` → `junk=False`, `pace.natural_sound=False`
    (this is what finally makes `cutrecord_map.py:623` reachable and turns
    `SND:ambient` into `SND:silence`)

    Side effects of `natural_sound=False` are bounded and desirable — its
    only two consumers are `cutrecord_map._audio_mute_for` (`:620`) and
    `observe.py:567-569`. It does not touch pacing math. It does set
    `mute=True` / `flags=["muted"]`, which is exactly right: muting a span
    that measures at its own file's noise floor loses nothing, and it is
    what stops a dead span from landing on the `role: "dialogue"` spine as
    all three did in the shipped edit.
  - `grade in ("carries", "unknown")` → today's values, byte-for-byte
- `vcut/speech/store.build_speech_cut_records` — compute and store
  `usability` for parity, but **do not** let it override the existing
  `non_speech_llm` / `non_speech_energy` gate. That gate already works; it
  gets a richer `usability` payload alongside, not a second opinion.

Delete the hardcoded `junk=False` at `vcut/store.py:135` and the hardcoded
`natural_sound=True` at `vcut/store.py:81`. **If either constant survives,
the fix is inert** — that is the revert signature the tests in §3.5 must
detect.

**Read side — three renders, all in existing functions:**

1. `cutrecord_map._to_cut_dict` (`cutrecord_map.py:626`) — carry
   `usability` through alongside the existing `junk` / `total_quality`
   passthrough at `:340`/`:360`.
2. `footage_map._moment_line` (`footage_map.py:1104`) — one compact token,
   rendered only when it says something:
   `use:silent_hold` / `use:dead(no-voice,floor,low-support)`, plus
   `edge:tail` when set. `carries` renders nothing (zero token cost on the
   common case). A `dead` moment already collapses to the terse
   `[JUNK: dead_span]` one-liner via `:1110`, so `use:` mostly annotates
   `silent_hold`.
3. `observe.inspect_cut` (`observe.py:916-928`) — add `"usability"` to the
   result dict, and add an **absolute** `rms_dbfs_median` next to the
   normalized `audio.curve` so the min-max normalization can no longer make
   digital silence look like a level (§2.5).

**Two accompanying presentation fixes** (they are part of the same root
cause — the brain cannot compare what it is shown):

4. `footage_map._qual` (`footage_map.py:953-955`) — put both channels on one
   0..1 scale before rendering (divide speech `total_quality` by its own
   4.5 ceiling, or render `q.` for visual and a distinct tag for delivery),
   and state the range in `converse.py:309`.
5. **`converse.py:284-287` — retract the false guarantee.** This is not
   cosmetic; it is the premise the brain reasoned from (§2.5). The
   `_PROVENANCE` block must describe what the *live* pipeline does: Pass 1
   marks every moment without filtering, boundaries are derived from the
   seam curve, and **each span carries a measured usability grade** — so a
   `silent_hold` or `dead` beat is expected in the index rather than
   impossible. Replace the "dropped entirely / nothing manufactures a cut
   out of dead footage" sentence with a pointer to the `use:` token and the
   `[JUNK: dead_span]` line. Also add the missing scale to
   `converse.py:309` per item 4.
6. `footage_map._annotate_dups` — extend duplicate grouping to the video
   channel using pHash. Two video moments whose sampled pHash sequences
   agree within the seamless band over their overlap are the same material.
   This is what makes `7144c9fc:m07`↔`aad020e3:m06` visible, and it
   generalizes to the whole-file duplicate-ingest case (mean cross-file
   Hamming 0.131 bits over 489 samples).

### 3.4 Already-ingested projects

**Backfill, not recompute, not lazy.**

- Not *recompute*: re-running ingest costs one Gemini Pass-1 call per file
  and would change every span boundary. The grade is a pure function of L1
  and the stored span — no VLM needed.
- Not *lazy*: computing at read time would create a second implementation
  that cannot write `junk`, so the frontend's junk hiding and
  `cutaway_pool`'s filter would disagree with the Beat Index. Divergent
  duplicate implementations of "what is junk" are exactly how the three
  hardcoded defaults in §2.3 arose.
- **Backfill**: `scripts/backfill_span_usability.py` — idempotent, one
  ingest run at a time, `--dry-run` first. For each run: load
  `ingest_runs.seam_cache` + transcripts + `frame_descriptors`, compute
  `usability` per `cut_records` row, and update `usability` /
  `junk` / `junk_reason` / `junk_confidence` / `pace.natural_sound` under
  the same derivation as the write path (imported, not re-coded). Rows whose
  run has no `seam_cache` grade `unknown` and are left untouched.
- Files ingested before `frame_descriptors` existed simply grade with
  `still = None`. 114 files already have descriptors, so this is rare.

### 3.5 Verification — by delivered effect, with stated revert signatures

The three bugs in §2.2–2.3 all shipped **with passing unit tests** against
code that cannot run. Every assertion below is therefore written against
either (i) real rows from ingest run `24894c8a-9daa-4aa3-aea2-1f3b60c93478`,
captured once into a fixture, or (ii) the actual rendered strings the model
receives. No assertion is of the form "a flag was set".

**V0 — reachability (the anti-dead-code assertion).** Over the fixture run,
assert `grade` takes **at least three distinct values** across the 31 spans,
and that the count of `dead` is `> 0` and `< 31`. *Fails on revert:* if the
grader is bypassed (an `or`-short-circuit like `has_peak`, an early return,
a hardcoded default), every span grades identically and the distinct-value
count collapses to 1. This is the specific assertion that would have caught
`_drop_dead_air`.

**V1 — pool membership, the delivered effect.** Call
`observe.affordances(empty_doc, ctx)` on the fixture run and assert:
- `'aad020e3:m06' not in cutaway_pool`
- `'7144c9fc:m07' not in cutaway_pool`
- `'7144c9fc:m08' not in cutaway_pool`
- `'7144c9fc:m00' in cutaway_pool` (the EDSO title card — silent, 100%
  silence, 0 words, and legitimately usable)
- `'93e94ec3:m05' in cutaway_pool` and `'93e94ec3:m06' in cutaway_pool`
  (silent screen-recording — graded `silent_hold`, still offered)
- `len(cutaway_pool) == 11` (14 today, minus exactly 3)

*Fails on revert:* `cutaway_pool` filters `not m.get("junk")`
(`observe.py:1825`). Restoring `junk=False` at `vcut/store.py:135`, or
reverting the grade→junk derivation, puts all 14 refs back and the
membership assertions fail. The positive assertions on `m00`/`m05`/`m06`
fail if the grade is replaced by a cruder silence-only rule.

**V2 — the rendered Beat Index line.** Assert on the exact substring the
model sees, from `footage_map.assemble_map(...)['text']`:
- the `aad020e3:m06` line contains `[JUNK: dead_span]` and does **not**
  contain `SND:ambient`
- the `7144c9fc:m00` line contains `SND:silence` and does **not** contain
  `SND:ambient` and does **not** contain `JUNK`
- the `7144c9fc:m01` line (a good voiced take) still contains
  `SND:speaking` and no `use:` token

*Fails on revert:* `SND:silence` requires `pace.natural_sound == False`
(`cutrecord_map.py:620-623`). Restoring `natural_sound=True` at
`vcut/store.py:81` makes `_audio_mute_for` return `"sound"` and the line
renders `SND:ambient` again — the assertion is on the string, so it cannot
pass against the old behaviour.

**V3 — grade values on named spans, with margins.** Table-driven over the
fixture, asserting `grade` **and** that the deciding axis clears its
threshold by a stated margin:

| ref | expected grade | asserted |
|---|---|---|
| `aad020e3:m06` | `dead` | `voice == 0.0`, `audible <= 0.02`, `support < 0.35` |
| `7144c9fc:m07` | `dead` | same |
| `7144c9fc:m08` | `dead` | `support == 0.0` |
| `7144c9fc:m00` | `silent_hold` | `voice == 0.0`, `support > 0.9` |
| `93e94ec3:m06` | `silent_hold` | `voice == 0.0`, `still == 1.0` |
| `7144c9fc:m01` | `carries` | `voice > 0.8`, `audible > 0.9` |
| `93e94ec3:m13` | `carries` | `voice > 0.95` |

*Fails on revert:* the negative rows (`m00`, `m06`, `m01`, `m13`) fail if
anyone replaces the fusion with a single silence or loudness threshold — the
failure mode that would delete title cards and beauty shots. The positive
rows fail if the grade never reaches `dead`.

**V4 — no silent audio on the delivered spine.** A regression assertion at
the level the user actually experiences: given the fixture pool, assert that
re-running the same six `place` calls now yields **either** a refusal for
`aad020e3:m06` **or** a `resolved.audio_layers` in which no `role:
"dialogue"` spine layer has `usability.grade == "dead"`. Additionally assert
directly on the current shipped document that the check *would have caught
it*: 7535 ms of its 24217 ms are `voice == 0.0 and audible <= 0.02`.

*Fails on revert:* this is computed from `resolved.audio_layers` and the
grade, not from a flag. It fails whenever a dead span reaches the spine
again, whatever the reason.

**V4b — the prompt no longer asserts what the pipeline does not do.** Assert
that `converse._context_block(...)` output does **not** contain the
substring `"dropped entirely rather than becoming a cut of its own"`, and
that it *does* contain an explanation of the `use:` token and of the `q.`
range. *Fails on revert:* the string is either present or absent; reverting
`converse.py` restores it verbatim. This is the one assertion that catches a
prompt regression, which no behavioural test would notice.

**V5 — video duplicate detection.** Assert
`assemble_map(...)['dup_groups']` contains a group whose members include
both `7144c9fc:m07` and `aad020e3:m06`, and that at least 5 of `aad020e3`'s
7 moments appear in some dup group. Today: **zero** video moments appear in
any dup group.

*Fails on revert:* the assertion counts video members of `dup_groups`, which
is 0 without the pHash extension.

**V6 — degradation, asserted not assumed.** Three synthetic files: (a) no
`frame_descriptors` row, (b) no transcript, (c) flat `rms_db` (spread <
`MIN_VOICED_SPREAD_DB`). Assert every span grades `carries` or
`silent_hold`, **never `dead`**, and that `inputs` names the missing signal.
*Fails on revert:* a fusion that treats `None` as 0 grades these `dead` and
silently deletes a whole file's pool.

**V7 — invariance.** Assert the grade is unchanged by the energy dial
(re-resolve at two energies, same file), matching the existing
energy-invariance guarantee that `vcut/store.py:104-118` already claims for
`framing` / `salience` / `scene_specifics`.

**Run-order note.** V0–V5 need the fixture captured *before* the change
(`scripts/capture_usability_fixture.py`, read-only, writes a JSON of the 31
rows + the run's `seam_cache` projection + the three files' L1 slices). They
must be observed **failing** against `HEAD` before the implementation lands;
a test that has never been seen red against the old code proves nothing.

---

## 4. What is explicitly not proposed

- No absolute dB / flow / pixel thresholds. Every axis is normalized against
  the file's own distribution, reusing the percentile references already
  shipped in `vcut/speech/params.py`.
- No new Hamming or loudness calibration. `feel.py`'s bit bands and
  `speech/params.py`'s percentiles are reused as-is.
- No keyword or label matching. The VLM label *"The speaker touches his chin
  while thinking"* is ignored entirely; only measurements decide.
- No special case for `93e94ec3` / `7144c9fc` / `aad020e3`, for screen
  recordings, or for pitch footage. Those file ids appear only inside test
  fixtures.
- No change to Pass 1's prompt. It is correct to emit everything; the fix
  supplies the "later" it defers to.
- No new LLM opinion. Pass 2's `usable` question stays display-only; a grade
  that is only present when a per-clip question planner happened to select
  it cannot gate a pool.
- No new write-only field. `junk_confidence` is either wired end-to-end with
  an assertion on rendered output, or dropped — see §3.3.
