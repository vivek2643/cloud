# speech_noise_gate.plan.md — Harden the speech noise gate (3 additive fixes)

## 0. Context & baseline (already implemented, uncommitted)

The "background noise → speech cut" bug (Whisper hallucinating confident text over
ambient noise, then that becoming a `kind='speech'` cut) already has a first,
working layer of defense in the working tree:

- **L1 VAD + thresholds** — `backend/app/services/l1/transcript.py`: `vad_filter=True`,
  `condition_on_previous_text=False`, explicit `no_speech_threshold` /
  `log_prob_threshold` / `compression_ratio_threshold` / `hallucination_silence_threshold`.
- **Confidence gate at load** — `backend/app/services/vcut/speech/inputs.py`:
  `_segment_is_non_speech(seg)` drops a whole Whisper segment when
  `no_speech_prob ≥ SEG_NO_SPEECH_MAX` or `avg_logprob < SEG_MIN_AVG_LOGPROB`
  (fail-open when the signal is absent — 0/71 existing transcripts have it).
- **Self-calibrating energy gate at beat resolution** —
  `backend/app/services/vcut/speech/delivery.py::is_voiced_beat(...)` +
  `orchestrate.py::_resolve_take_group`: a beat whose median `rms_db` sits near the
  file's noise floor is dropped. This is the only gate live on existing data and the
  one that actually catches the confident phantoms (validated: 12/205 dropped, all
  noise like "Thanks for your time" sitting at the noise floor).

**This plan is additive to that baseline.** It closes the three remaining gaps:

1. The acoustic gates are blind to **loud** non-speech (music/TV/lyrics/boilerplate).
2. The energy gate's floor is a blind 15th percentile, not the file's true noise floor.
3. Dropped beats vanish silently — no audit trail, not recoverable.

Not over-engineering / not band-aid: each item targets a *distinct* failure mode or
property (semantic coverage / calibration accuracy / observability), and item 3 makes
the whole gate *safe* (a false positive is recoverable, not lost).

Principles: keep pure functions pure (delivery/select/boundaries are I/O-free); keep
vcut isolated from L3 (only `l3.post.CutRecord` + `ingest_store` imported); every
new constant lives in `params.py`.

---

## 1. Improvement A — Semantic non-speech flag from the segment LLM

**Why:** The segment LLM is the only stage that reads the *words*. It can recognize
content that is not the subject's spoken performance — song lyrics from a background
track, audio bleeding from a TV/other source, or boilerplate/hallucinated phrases over
noise ("thanks for watching", "please subscribe"). Acoustic gates can't see this
(it's loud and "confident"). This is the missing coverage.

**File:** `backend/app/services/vcut/speech/segment_llm.py`.

### A.1 Schema — add one field to `_BeatOut` (lines ~64–70)

```python
class _BeatOut(BaseModel):
    id: str
    file_id: str
    word_span: Tuple[int, int]
    gist: str = ""
    fluency: float = 0.5
    flags: List[Literal["incomplete", "false_start"]] = []
    is_speech: bool = True   # NEW: false => not the subject's real spoken content
```

Default `True` = fail-open: the model must *affirmatively* judge a beat non-speech to
gate it, so an omitted/uncertain field never drops real speech.

### A.2 Prompt — add a short instruction to `_SYSTEM_PROMPT` (after the `flags` bullet)

Add (own wording, keep it tight):

```
  "is_speech": true for genuine spoken content by a person on the recording; set it
    to FALSE only when the words are clearly NOT that — e.g. song lyrics from
    background music, dialogue bleeding from a TV/other playback, or boilerplate the
    transcriber invented over ambient noise ("thanks for watching", "please
    subscribe"). When unsure, leave it true. A non-speech beat still gets an id,
    file_id, word_span, and gist; just mark is_speech=false. Do not put it in any
    take_group.
```

Note in a comment that this is text-only judgment and is one of three orthogonal
signals — its job is specifically the loud-non-speech case the acoustic gates miss.

### A.3 Carry the flag through to resolution

`_BeatOut.is_speech` must reach `orchestrate._resolve_take_group` (it already has the
`_BeatOut`). No plumbing needed beyond reading `beat.is_speech` there — see §3.

`_validate_beats` / `build_take_groups` (lines ~109–157) are unchanged; a non-speech
beat can still be validated and simply won't be selected as a winner (§3).

---

## 2. Improvement B — Calibrate the energy floor from `silence_intervals`

**Why:** `is_voiced_beat` compares a beat's median energy against a floor/voiced
reference. Today the floor is a blind 15th percentile of all `rms_db`
(`inputs._energy_refs`). L1 already detects real silence (`audio_features.silence_intervals`,
loaded as `FileSpeechInputs.silences`). Using the **median energy over actual silence
frames** as the floor makes the threshold track the file's *true* noise floor — more
accurate, fewer false drops on quiet-but-real speech, and it pins phantoms sitting
right at the measured floor. This sharpens the existing gate; it is **not** a new gate.

**File:** `backend/app/services/vcut/speech/inputs.py`.

### B.1 `_energy_refs` — take silences + hop, prefer measured floor

Change signature `._energy_refs(rms_db)` → `_energy_refs(rms_db, hop_ms, silences)`:

```python
def _energy_refs(rms_db, hop_ms, silences):
    """(voiced_ref_db, floor_ref_db). voiced_ref = 75th pct of all rms (typical
    speech level). floor_ref = MEDIAN rms over frames inside detected silence
    (the file's true noise floor) when there is enough silence to measure;
    otherwise the 15th-pct fallback. 0.0/0.0 when there isn't enough rms data to
    gate at all (is_voiced_beat then no-ops via MIN_VOICED_SPREAD_DB)."""
    vals = [float(v) for v in rms_db if v is not None]
    if len(vals) < 20:
        return 0.0, 0.0
    s = sorted(vals)
    voiced = _percentile(s, RMS_VOICED_PCTL)
    floor = None
    if hop_ms and hop_ms > 0 and silences:
        sil_frames = []
        for a, b in silences:
            lo, hi = a // hop_ms, min(len(rms_db) - 1, b // hop_ms)
            sil_frames.extend(float(rms_db[i]) for i in range(max(0, lo), hi + 1)
                              if rms_db[i] is not None)
        if len(sil_frames) >= SILENCE_FLOOR_MIN_FRAMES:
            floor = _median(sil_frames)   # reuse a local median (or import delivery._median)
    if floor is None:
        floor = _percentile(s, RMS_FLOOR_PCTL)   # fallback: existing behavior
    return voiced, floor
```

Add a small local `_median` (or reuse `delivery._median` — but prefer local to keep
`inputs` import-light). Add `SILENCE_FLOOR_MIN_FRAMES` to params (§4).

### B.2 `load_file_inputs` — pass hop + silences

`_silences_for_file` is already called there. Compute refs *after* loading silences:

```python
rms_db, rms_hop_ms = _rms_for_file(file_id)
silences = _silences_for_file(file_id)
voiced_db, floor_db = _energy_refs(rms_db, rms_hop_ms, silences)
return FileSpeechInputs(..., silences=silences, rms_voiced_db=voiced_db, rms_floor_db=floor_db)
```

`is_voiced_beat` (delivery.py) is **unchanged** — it still gates on
`energy ≥ floor + MIN_VOICED_FRAC*(voiced-floor)`; only the floor got more accurate.

---

## 3. Improvement C — Mark non-speech as junk, don't silently delete

**Why:** Today a gated beat is `continue`'d away — it never becomes a cut and leaves no
trace. Marking it `junk=True` (with a reason) instead makes gating **auditable and
reversible**: it's hidden from the default cuts view (same as video junk) but
recoverable if a real line was ever caught. This is what makes the acoustic + LLM
gates *safe* to run aggressively. It also aligns the speech channel with the video
pipeline's existing junk concept (speech currently hardcodes `junk=False`).

**Files:** `backend/app/services/vcut/speech/store.py`,
`backend/app/services/vcut/speech/orchestrate.py`.

### C.1 `store.ResolvedBeat` — carry junk (lines ~28–40)

```python
@dataclass
class ResolvedBeat:
    beat: _BeatOut
    in_ms: int
    out_ms: int
    take_group_key: str
    take_role: str            # "winner" | "take" (never a winner if junk)
    speech_quality: float
    junk: bool = False        # NEW
    junk_reason: str = ""     # NEW: "non_speech_energy" | "non_speech_llm"
```

### C.2 `store._build_record` — stop hardcoding `junk=False` (lines ~166–177)

Thread the flags through: `_build_record(..., junk: bool = False, junk_reason: str = "")`
and set `junk=junk, junk_reason=junk_reason` on the `CutRecord`. Update
`build_speech_cut_records`'s call to pass `rb.junk`, `rb.junk_reason`
(`ec.resolved_beat.junk/junk_reason`). Outlook fan-out inherits the winner's junk
(a winner is never junk by construction, so angles stay non-junk).

### C.3 `orchestrate._resolve_take_group` — gate → junk, not drop (lines ~44–79)

Replace the current drop-on-gate with a two-pool split:

- For each beat, compute boundaries + metrics as today.
- **Determine junk reason (never a winner):**
  - `beat.is_speech == False` → `junk_reason = "non_speech_llm"`.
  - else if `not delivery.is_voiced_beat(metrics.energy, fi.rms_voiced_db, fi.rms_floor_db)`
    → `junk_reason = "non_speech_energy"`.
  - else → real candidate.
- **Real candidates** go through `delivery.group_delivery_scores` + `select.select_winner`
  exactly as today (so junk beats never dilute normalization or steal the winner slot).
- **Junk beats** are emitted as `ResolvedBeat(..., take_role="take", speech_quality=0.0,
  junk=True, junk_reason=...)` — their own singletons, never fanned out (only winners
  fan out), never entering `select`.
- Return real-resolved + junk-resolved beats together.
- Keep the existing `logger.info` line for energy drops; add the reason.

**Guardrail (state in a comment):** junk beats are excluded from
`group_delivery_scores`/`select_winner`, so a phantom can never win a take group or
suppress a real take. A group whose beats are *all* junk yields only junk records
(no winner) — acceptable; they're hidden by the junk filter.

### C.4 Frontend must hide junk speech cuts from the default view

Verify `cuts-view.tsx` (or the `GET /cuts` serializer) already filters `junk=True`
for `kind='speech'` the same way it does for video. If speech junk isn't currently
filtered, add it so the user's original complaint ("noise appears AS cuts") stays
solved — junk is visible only under an explicit "show junk"/debug toggle if one
exists. (If there is no junk filter for speech at all today, that filter is the one
frontend change this plan needs; note it in the change list.)

---

## 4. `params.py` additions (`backend/app/services/vcut/speech/params.py`)

Under the existing "Noise/hallucination gate" block, add:

```python
# Improvement B: measure the true noise floor from detected silence frames rather
# than a blind percentile, when there's enough silence to be representative.
SILENCE_FLOOR_MIN_FRAMES = 10
```

`RMS_VOICED_PCTL`, `RMS_FLOOR_PCTL`, `MIN_VOICED_FRAC`, `MIN_VOICED_SPREAD_DB`,
`SEG_NO_SPEECH_MAX`, `SEG_MIN_AVG_LOGPROB` already exist — unchanged.

No new junk-reason constants needed (short literal strings at the call site are fine).

---

## 5. Testing

Pure-unit style under `backend/scripts/test_*.py` (match `test_speech_delivery.py`).

1. **`_energy_refs` with silence:** synthetic `rms_db` + `silences` → floor == median
   over silence frames (not the 15th pct). With `<SILENCE_FLOOR_MIN_FRAMES` silence →
   falls back to 15th pct. With `<20` samples → (0.0, 0.0) (gate disabled).
2. **`is_voiced_beat` unchanged:** loud beat keeps, floor-level beat drops, sub-spread
   file fail-opens (regression guard — behavior must not change from the better floor
   alone except via the floor value).
3. **Segment schema:** `_BeatOut` parses with and without `is_speech`; default is
   `True`; `is_speech=False` round-trips.
4. **`_resolve_take_group` routing:** a beat with `is_speech=False` → emitted junk
   (`non_speech_llm`), excluded from `select`; a near-floor beat → junk
   (`non_speech_energy`); a real beat → real candidate and can win. A group of
   [real, junk] → real wins, junk emitted with `take_role="take"`, `junk=True`.
5. **`store` junk threading:** `_build_record(..., junk=True, junk_reason="x")` →
   `CutRecord.junk is True` and reason set; default call → `junk=False`.
6. **Pyflakes** clean on all touched modules.

**Read-only replay (no re-ingest):** extend/rerun `backend/scripts/_diag_noise_gate.py`
to also report, per existing speech cut, which junk_reason it *would* now get (energy
vs — for that script — n/a for LLM since it needs a re-run). Confirms the silence-
calibrated floor doesn't newly drop real speech vs the current 12/205.

**Live acceptance (spends money, separate step — do NOT auto-run):** re-ingest one
project with background music/TV and confirm those beats come back `junk=True,
non_speech_llm` and are hidden from the default cuts view.

---

## 6. File-by-file change list

| File | Change |
|---|---|
| `backend/app/services/vcut/speech/segment_llm.py` | Add `is_speech: bool = True` to `_BeatOut`; add the `is_speech` instruction to `_SYSTEM_PROMPT` (§1). |
| `backend/app/services/vcut/speech/inputs.py` | `_energy_refs(rms_db, hop_ms, silences)` → silence-measured floor w/ percentile fallback; local `_median`; `load_file_inputs` passes hop+silences (§2). |
| `backend/app/services/vcut/speech/orchestrate.py` | `_resolve_take_group`: gate → junk routing (`non_speech_llm` / `non_speech_energy`), junk beats excluded from `select`, returned alongside real beats (§3.3). |
| `backend/app/services/vcut/speech/store.py` | `ResolvedBeat.junk/junk_reason`; `_build_record` threads junk instead of hardcoded `False`; `build_speech_cut_records` passes them (§3.1–3.2). |
| `backend/app/services/vcut/speech/params.py` | Add `SILENCE_FLOOR_MIN_FRAMES` (§4). |
| `frontend/src/components/cuts-view.tsx` (or `GET /cuts` serializer) | Ensure `kind='speech'` `junk=True` is hidden from the default view (§3.4) — only if not already filtered. |
| `backend/scripts/test_speech_*.py` | New/updated unit tests (§5). |

---

## 7. Out of scope (explicitly)

- No change to the L1 VAD/threshold config or the existing `_segment_is_non_speech` /
  `is_voiced_beat` *decision math* (only the floor's calibration improves).
- No re-transcribe and no batch re-ingest here — that's a separate, money-spending
  step. These changes take effect on the next ingest; the silence-floor improvement
  needs no re-transcribe (works on existing `rms_db`/`silences`), while `is_speech`
  needs the next speech-channel run.
- No new take-selection semantics beyond "junk never competes / never wins."
