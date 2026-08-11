# Restore the identity map for the vcut pipeline (wire L3 identity onto L1)

**Status:** implementation plan / handoff doc. Someone codes directly from this.
**Scope:** re-wire the EXISTING `l3/identity` reconciliation into the new `vcut`
pipeline so `ingest_runs.identity_map` is populated again — "exactly as before,"
maximum reuse, minimal deviation. Do **not** redesign identity.

All file:line references below were read at plan time. All DB facts were verified
read-only against the shared prod Postgres via `backend/.venv/bin/python`.

---

## 1. Problem statement + confirmed current-state evidence

`ingest_runs.identity_map` (jsonb, migration `037_identity_map.sql`) is the
persisted "reconciled cast" payload. The **read/render** path is intact:
`footage_map` prefixes a one-line `CAST:` summary onto the Tier-0 footage index
the brain reads. But the **write/reconciliation** step lives ONLY in the old
`l3` cuts pipeline (`backend/app/services/l3/ingest.py:447-450`) and the new
`vcut` pipeline never runs it. `backend/app/services/vcut/` has **zero**
identity references.

Verified read-only against prod:

| Query | Result |
|---|---|
| `ingest_runs` total / with `identity_map` / last populated | **395 / 57 / 2026-07-26 21:13:26Z** |
| `vcut:%` runs total / with `identity_map` | **136 / 0** |
| demo-trail run `6f31d648…` (`pass1_model='vcut:gemini-3.1-flash-lite'`, status `ready`) `identity_map` | **NULL** |

So every vcut run's `identity_map` is NULL, no run has been reconciled since the
old pipeline was retired (2026-07-26), and the brain's CAST line is dark on all
current projects. **The only thing missing is the write side.**

### Proof the fix works (read-only dry-run, no DB writes)

Against a real 2-person vcut project (`57b689b3…`, run `b40020d6…`, 4 files that
DO have `face_tracks`), I assembled the identity inputs purely from L1 + the
shared leaf turn-loader and called the existing `identity_apply.run(...)` with an
**empty** cut list. It produced a byte-shaped-identical, correctly-reconciled
payload:

```json
{
  "persons": [
    {"person_id": "P0", "appearance_count": 5523, "is_major": true, "owned_voices": ["V0"]},
    {"person_id": "P1", "appearance_count": 5453, "is_major": true, "owned_voices": ["V1"]}
  ],
  "voice_owner": {"V0": "P0", "V1": "P1"},
  "off_camera_voices": []
}
```

and `footage_map._cast_line(persons)` on it rendered:

```
CAST: P0 (P0) [voice:V0]; P1 (P1) [voice:V1]
```

This validates the recommended design (option **a**) end to end using the
existing modules unchanged.

---

## 2. Anatomy of the OLD identity pipeline

> Note: the folder is `backend/app/services/l3/identity/` = `apply.py`, `voices.py`,
> `faces.py`, `bind_asd.py`, `__init__.py`. There is **no** `reconcile.py` /
> `voice_id.py` / `bind.py` any more — those were deleted when `asd_identity.plan.md`
> replaced LLM/motion binding with deterministic CV+code. Several docstrings still
> reference `reconcile.py`; treat those as historical.

Identity is **CV + code, no model call anywhere**. It has four pure-ish
functions plus one final assembler.

### 2.1 `identity/voices.py` — cross-clip VOICE clustering
- `assign_voices(embeddings_by_file, groups, all_speakers_by_file, threshold=0.75)`
  → `Dict[(file_id, local_speaker) -> "V0"/"V1"/…]` (`voices.py:143-153`, core in
  `cluster_voices` `voices.py:52-140`).
- **Inputs:**
  - `embeddings_by_file: {file_id: {local_speaker: [float,…]}}` — pyannote
    voiceprints (L1).
  - `groups: {group_id: {"members": [file_id,…], …}}` — outlook groups; only
    `.get("members")` is read (`voices.py:107-114`).
  - `all_speakers_by_file: {file_id: [local_speaker,…]}` — the full speaker
    roster (every speaker that actually talks).
- Deterministic union-find: outlook-group members' same-label speakers unify by
  construction; cross-file pairs with cosine ≥ 0.75 merge. Conservative
  ("over-split is the safe failure").

### 2.2 `identity/faces.py` — cross-file FACE clustering + per-cut visibility
- `cluster(face_tracks_by_file, threshold=0.45)` → `(track_to_person, persons)`
  (`faces.py:59-125`).
  - `track_to_person: {(file_id, track_id) -> "P0"/"P1"/…}`.
  - `persons: {pid: {"person_id","appearance_count","is_major":True,"owned_voices":[]}}`
    — the deterministic cast core; `owned_voices` filled later by `apply.py`.
  - Clusters ArcFace embeddings from L1 `FaceTrack.embedding` (union-find, cosine
    ≥ 0.45). **Does NOT use any LLM appearance traits.**
- `visible_persons_by_cut(track_to_person, face_tracks_by_file, cuts, lattices, span_override=None)`
  → `{(file_id, source_ref) -> [person_id,…]}` (`faces.py:148-190`). Duck-typed on
  `cut.file_id / cut.source_ref / cut.kind / cut.word_span`; a speech cut resolves
  its ms span from `word_span` via the lattice, a video cut from `span_override`.
  Capped at `MAX_VISIBLE_PER_CUT=6`. **Only needed for the per-cut rewrite, not
  for the CAST payload.**

### 2.3 `identity/bind_asd.py` — bind voices → persons via ASD overlap
- `bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)`
  → `(owner_by_voice: {voice -> person|None}, off_camera_voices: set)`
  (`bind_asd.py:60-105`).
- **Inputs:**
  - `turns_by_file: {file_id: [(start_ms, end_ms, local_speaker), …]}` —
    diarization turns.
  - `voice_of` (from 2.1), `face_tracks_by_file`, `track_to_person` (from 2.2).
- For each global voice, sums (ms) each candidate person's ASD-speaking overlap
  inside that voice's turns across all files; the top person wins iff it clears
  `MIN_OVERLAP_MS=1000` and beats runner-up by `MIN_MARGIN_RATIO=1.2`, else the
  voice is left unbound/off-camera. Never guesses.

### 2.4 `identity/apply.py` — final assembly (THE entrypoint)
- `run(pass2_output, voice_of, persons, visible_persons, owner_by_voice, unbound_voices)`
  → `(Pass2Output, payload)` (`apply.py:72-106`).
- Does two things:
  1. **Rewrites cuts** (`_rewrite_cuts`, `apply.py:45-69`): per cut attaches
     `visible_persons`, sets `speaker_person = owner_by_voice.get(cut.voice_ids[0])`
     (dominant voice only; None if unbound/unknown), and `on_camera =
     speaker_person in visible_persons`. Needs real cut objects with `voice_ids`.
  2. **Builds the payload** (`apply.py:91-102`): attaches `owned_voices` onto
     each person, then returns
     `{"persons": [...], "voice_owner": {v: p for bound}, "off_camera_voices": sorted(...)}`.
     This depends ONLY on `persons` / `owner_by_voice` / `unbound_voices` /
     `voice_of` — **not on the cuts.** Feeding empty cuts yields the exact
     payload with a no-op rewrite (verified in §1).

### 2.5 How it was wired in `l3/ingest.py` (`ingest.py:423-450`)

Import: `from app.services.l3.identity import apply as identity_apply` (+
`identity_faces`, `identity_bind_asd`, `identity_voices`) at `ingest.py:57-60`.

Input assembly (all upstream of the identity block):
- `embeddings_by_file = _embeddings_for_files(file_ids)` — `select speaker_embeddings from transcripts` (`ingest.py:129-143`).
- `all_speakers_by_file` — union of `pass1_output.speech_cuts[].speaker_ids` (`ingest.py:319-321`).
- `voice_of = identity_voices.assign_voices(embeddings_by_file, groups, all_speakers_by_file)` (`ingest.py:322`).
- `turns_by_file = {fid: lat.turns for fid, lat in lattices.items()}` (`ingest.py:323`).
- `face_tracks_by_file = _face_tracks_for_files(file_ids)` — `select tracks from face_tracks` → `active_speaker.FaceTrack.from_dict` (`ingest.py:154-167`, `434`).

The block itself:
```447:450:backend/app/services/l3/ingest.py
        pass2_output, identity_map = identity_apply.run(
            pass2_output, voice_of, persons, visible_persons, owner_by_voice, unbound_voices)
        if identity_map.get("persons"):
            store.set_identity_map(ingest_run_id, identity_map)
```
`pass2_output` here is the pass-2 cut set (rewritten in place); `visible_persons`
came from `identity_faces.visible_persons_by_cut(...)` with a `v4_span_override`
for video cuts (`ingest.py:440-444`). The `if identity_map.get("persons")` guard
is the fail-open: nothing clustered → no write → NULL.

### 2.6 The persisted store (`l3/ingest_store.py`)
- `set_identity_map(ingest_run_id, identity_map)` → `update ingest_runs set identity_map = %s` (`ingest_store.py:61-69`).
- `get_identity_map(ingest_run_id)` → dict or **None** for "older run / nothing
  reconciled / unknown id" (fail-open, `ingest_store.py:72-83`).
- Real payload shape (dumped from newest populated run `a38613c0…`, 2026-07-26):
  top-level keys `["persons","voice_owner","off_camera_voices"]`; each person
  `{"person_id","appearance_count","is_major","owned_voices"}`. Example:
  ```json
  {"persons":[{"is_major":true,"person_id":"P0","owned_voices":[],"appearance_count":7},
              {"is_major":true,"person_id":"P1","owned_voices":["V0"],"appearance_count":26}],
   "voice_owner":{"V0":"P1"},"off_camera_voices":[]}
  ```

---

## 3. The READ/render side is pipeline-agnostic and already works

- `footage_map._cast_table_for(file_ids, run_id)` resolves the run
  (`run_id or cuts_read.latest_run_for_files`), calls
  `ingest_store.get_identity_map(resolved_run_id)`, and returns
  `_cast_line(identity_map["persons"])` — wrapped in try/except that degrades to
  `""` on any error (`footage_map.py:1291-1308`).
- `_cast_line` (`footage_map.py:1271-1288`) renders `CAST: Pn (display) [voice:…]; …; other: …`.
- `assemble_map` prefixes the cast line onto the index when present
  (`footage_map.py:1333-1335`).
- Per-cut identity render: `_pic_who` reads `m.get("visible_persons")`
  (`footage_map.py:556-568`); SND/`speaker_person` read the same way per moment.

None of this cares which pipeline produced the run — it reads `ingest_runs.identity_map`
and the `cut_records.visible_persons/speaker_person` columns by id. **Confirmed:
only the write side is missing.**

---

## 4. Inputs: available from shared L1 for vcut vs. the "gap"

L1 is shared and runs identically for vcut projects. Everything reconciliation
needs is L1-derived; **nothing** comes from the old `l3` Pass-2.

| Input | Source | Available to vcut? |
|---|---|---|
| `embeddings_by_file` | `transcripts.speaker_embeddings` (L1 diarization, `diarization.py:144-175`) | ✅ same table, read directly |
| `turns_by_file` | `l3/diarize.load_turns(file_id)` — a LEAF module (imports only config+psycopg) that merges `transcripts.segments[].words[].speaker` into `(s,e,spk)` turns (`diarize.py:75-112`) | ✅ reusable as-is |
| `all_speakers_by_file` | same `load_turns` call → `speaker_ids` | ✅ |
| `face_tracks_by_file` | `face_tracks.tracks` → `active_speaker.FaceTrack.from_dict` (full embedding+frames+speaking, `active_speaker.py:102-112`) | ✅ same table, read directly |
| `groups` (outlook) | vcut already has `speech/outlooks.load_sync_groups` (`outlooks.py:36-67`) | ✅ trivial adapter (see §5) |

**The "gap" the old `l3` Pass-2 produced but vcut Pass-2 does not:** the old
`l3/pass2.py` emitted per-person LLM appearance traits (`Appearance`/`PersonLook`,
`pass2.py:130-159,219-226`) and the model's per-cut `people` list. **These are
vestigial for identity** — `asd_identity.plan.md` replaced LLM-label clustering
with ArcFace-embedding clustering (`faces.py:1-22`). `identity_apply.run` and its
inputs (`voices/faces/bind_asd`) consume **none** of the Pass-2 VLM traits. So
there is **no real gap for the CAST payload**: reconciliation runs on L1 alone.

The only thing that genuinely needs cuts is the **per-cut rewrite** (§2.4 item 1),
because it stamps `visible_persons`/`speaker_person` onto each cut. vcut speech
cuts do not currently carry `voice_ids` (they default `[]`, `speech/store.py:174-185`;
`vcut/store.py:131-141`), so per-cut `speaker_person` would need voice labels
derived first. That is the only extra work, and it is **not** required for the
CAST line — it is a scoped Phase-2 enhancement (§5.4).

---

## 5. Design — chosen wiring, rationale, insertion point

### 5.1 Decision: option (a) — a dedicated L1-adjacent reconciliation step in vcut

**Reject option (b)** (piggyback vcut Pass-2 to emit identity traits, then
reconcile): it would re-introduce LLM appearance traits that `asd_identity.plan.md`
deliberately deleted, add a model dependency to a CV+code process, and deviate
from "exactly as before." Reconciliation does **not** need per-cut VLM face
traits — it needs L1 face embeddings + L1 ASD + diarization turns, all of which
vcut projects already have (§1 dry-run, §4).

**Adopt option (a):** a small vcut function that assembles the identity inputs
from L1 (same reads `l3/ingest.py` does) and calls the **existing**
`identity_apply.run(...)` unchanged, then `l3store.set_identity_map(...)` behind
the same `if payload.get("persons")` guard. This reuses `voices.py`, `faces.py`,
`bind_asd.py`, `apply.py`, `l3/diarize.py` **all unchanged**, and is the least
possible deviation from the old wiring.

### 5.2 What is reused UNCHANGED
- `l3.identity.voices.assign_voices`
- `l3.identity.faces.cluster` (and `visible_persons_by_cut` only if Phase 2)
- `l3.identity.bind_asd.bind`
- `l3.identity.apply.run`
- `l3.diarize.load_turns` (LEAF)
- `l3.ingest_store.set_identity_map` / `get_identity_map`
- `l1.active_speaker.FaceTrack`

No new identity logic. No changes to any `l3/identity/*` file. No changes to
vcut Pass-2. (This crosses vcut's usual "import nothing from l3 except
ingest_store/post" isolation — but that is the explicit intent of this task, and
vcut already imports `l3.post.CutRecord` and `l3.ingest_store`; `l3.diarize` is a
documented LEAF safe to import from anywhere.)

### 5.3 Exact insertion point in `vcut/orchestrate.py`

In `run_vcut_ingest`, after the speech channel + video cuts are written and
before `set_status("ready")`:

```405:407:backend/app/services/vcut/orchestrate.py
        l3store.set_status(ingest_run_id, "ready")
        logger.info("vcut_ingest run %s: %d video cut(s), %d speech cut(s), %d file(s)",
                   ingest_run_id, len(video_ids), n_speech, len(file_ids))
```

Insert the reconciliation call **immediately before** `set_status("ready")`
(i.e. right after `n_speech = _run_speech_channel(...)` at
`orchestrate.py:376`, and after `qplan`/`persist_seam_and_plan`). Rationale:
mirrors l3's order (identity after cuts, before "ready"); the payload is
energy-invariant (doesn't depend on resolved cuts), so it runs exactly once here
and NOT in the energy re-resolve path (`routers/projects.py`).

Wrap the whole call in try/except that logs and continues — identity must never
fail a vcut run (fail-open, §7).

### 5.4 Phasing
- **Phase 1 (required — lights up the CAST line):** compute & persist
  `identity_map` via `apply.run(Pass2Output(cuts=[]), …)`. No per-cut work. This
  is exactly what §1's dry-run proved.
- **Phase 2 (optional — full "as before" per-cut PIC/SND ids):** also stamp
  `visible_persons`/`speaker_person` onto vcut `cut_records`. Requires deriving
  per-speech-cut `voice_ids` (map each cut's `word_span` → its words' `speaker`
  labels → `voice_of`) and building cut-shaped objects for
  `visible_persons_by_cut` (video cuts pass `src_in_ms/src_out_ms` as the
  `span_override`), then `update cut_records set voice_ids/speaker_person/visible_persons`.
  Recommend shipping Phase 1 first; Phase 2 behind the same fail-open guard.

---

## 6. Exact change list per file

### 6.1 NEW: `backend/app/services/vcut/identity.py` (new module)
A single function that owns the L1 reads + the reconcile call. Keeps
`orchestrate.py` thin and mirrors the pattern of other `vcut/*` helpers.

```python
"""Wire the existing l3 identity reconciliation onto vcut's shared-L1 inputs
(identity_map_vcut.plan.md). No new identity logic -- assembles inputs from L1
and calls l3.identity.apply.run unchanged. Fail-open: never raises."""
from __future__ import annotations
import logging
from typing import Dict, List

from app.services.l1 import active_speaker as asd
from app.services.l3 import diarize as l3diarize
from app.services.l3 import ingest_store as l3store
from app.services.l3.identity import apply as identity_apply
from app.services.l3.identity import bind_asd as identity_bind_asd
from app.services.l3.identity import faces as identity_faces
from app.services.l3.identity import voices as identity_voices
from app.services.l3.pass2 import Pass2Output
from app.services.vcut.speech.outlooks import load_sync_groups

logger = logging.getLogger(__name__)


def _pg_conn():
    from app.services import db
    return db.connection()


def _embeddings_for_files(file_ids: List[str]) -> Dict[str, Dict[str, list]]:
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, speaker_embeddings from transcripts "
            "where file_id = any(%s::uuid[])", (file_ids,)).fetchall()
    return {fid: (emb or {}) for fid, emb in rows if emb}


def _face_tracks_for_files(file_ids: List[str]) -> Dict[str, List[asd.FaceTrack]]:
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, tracks from face_tracks "
            "where file_id = any(%s::uuid[])", (file_ids,)).fetchall()
    return {fid: [asd.FaceTrack.from_dict(t) for t in (tracks or [])] for fid, tracks in rows}


def _groups_for_files(file_ids: List[str]) -> Dict[str, dict]:
    # voices.cluster_voices only reads grp["members"] (a list of file_ids).
    return {gid: {"members": list(g.members.keys())}
            for gid, g in load_sync_groups(file_ids).items()}


def reconcile_and_store(ingest_run_id: str, file_ids: List[str]) -> None:
    """Compute this run's identity_map from L1 signals (voices/faces/ASD) and
    persist it, exactly like l3/ingest.py:447-450. Fail-open: any error is
    logged and swallowed -- the run stays 'ready' with identity_map NULL, the
    same byte-identical fallback footage_map already handles."""
    try:
        embeddings_by_file = _embeddings_for_files(file_ids)
        face_tracks_by_file = _face_tracks_for_files(file_ids)
        groups = _groups_for_files(file_ids)

        turns_by_file: Dict[str, list] = {}
        all_speakers_by_file: Dict[str, list] = {}
        for fid in file_ids:
            _text, spk_ids, turns = l3diarize.load_turns(fid)
            turns_by_file[fid] = turns
            all_speakers_by_file[fid] = spk_ids

        voice_of = identity_voices.assign_voices(embeddings_by_file, groups, all_speakers_by_file)
        track_to_person, persons = identity_faces.cluster(face_tracks_by_file)
        owner_by_voice, unbound = identity_bind_asd.bind(
            turns_by_file, voice_of, face_tracks_by_file, track_to_person)

        # Phase 1: CAST payload only. Empty cuts -> no-op rewrite, exact payload.
        _out, identity_map = identity_apply.run(
            Pass2Output(cuts=[]), voice_of, persons, {}, owner_by_voice, unbound)
        if identity_map.get("persons"):
            l3store.set_identity_map(ingest_run_id, identity_map)
            logger.info("vcut identity: run %s -> %d person(s), %d voice(s) bound",
                        ingest_run_id, len(identity_map["persons"]),
                        len(identity_map["voice_owner"]))
    except Exception:
        logger.exception("vcut identity: reconciliation failed for run %s "
                         "(identity_map left NULL -- cuts unaffected)", ingest_run_id)
```

### 6.2 EDIT: `backend/app/services/vcut/orchestrate.py`
Add, immediately before `l3store.set_status(ingest_run_id, "ready")`
(`orchestrate.py:405`):

```python
        from app.services.vcut import identity as vcut_identity
        vcut_identity.reconcile_and_store(ingest_run_id, file_ids)
```
(`file_ids` is already in scope, `orchestrate.py:314`.) The function is
self-guarding (fail-open), so no extra try/except is needed at the call site.

### 6.3 (Phase 2 only, optional) EDITs for per-cut ids
- `backend/app/services/vcut/identity.py`: after `apply.run`, build cut-shaped
  objects (duck-typed `file_id/source_ref/kind/word_span/voice_ids`), derive
  each speech cut's `voice_ids` from its `word_span` words' `speaker` → `voice_of`,
  build `visible_persons_by_cut` (video cuts via `span_override={source_ref:(in,out)}`),
  then `update cut_records set voice_ids=…, speaker_person=…, visible_persons=…`.
- No new columns needed: `cut_records` already has `voice_ids`, `speaker_person`,
  `visible_persons` (`ingest_store.py:178,205`).
- Must also re-run in the energy re-resolve path only if video cut ids change;
  simplest is to key the update by `(ingest_run_id, file_id, src_in_ms, src_out_ms)`.

### 6.4 No changes anywhere else
No migration (column exists). No edits to `l3/identity/*`, `l3/diarize.py`,
`l3/ingest_store.py`, `footage_map.py`, or vcut Pass-2.

---

## 7. Fail-open behavior (preserved)

Every failure mode degrades to `identity_map = NULL` → `get_identity_map` returns
None → `_cast_table_for` returns `""` → the footage index is byte-identical to
today (no CAST line, PIC/SND carry no ids). Specifically:
- No `face_tracks` for the project (e.g. the demo-trail project `8621c012…`,
  verified: **0 face_tracks rows**, single speaker `S0`, empty
  `speaker_embeddings`) → `persons=[]` → `if identity_map.get("persons")` guard
  skips the write. **Same as old behavior.**
- Any exception in `reconcile_and_store` → logged + swallowed, run stays `ready`.
- `load_turns` empty / no embeddings → voices over-split to singletons, nothing
  binds, `persons` may still exist but with no `owned_voices`; still a valid
  payload, still guarded.

The reconciliation runs **after** cuts are inserted and status work is done, so
even a hypothetical crash inside it cannot affect the already-written cuts.

---

## 8. Test plan + verification

### 8.1 Unit / dry-run (no DB writes) — already demonstrated
Re-run the §1 dry-run for a candidate project and assert the payload shape:
- Candidate: project `57b689b3-39db-4cb4-8385-9e87a996fe9a` (run `b40020d6…`),
  4 files WITH face_tracks + 2-speaker embeddings.
- Expected: `persons` has ≥1 `is_major` person, `voice_owner` non-empty, keys
  exactly `{"persons","voice_owner","off_camera_voices"}`, each person keys
  exactly `{"person_id","appearance_count","is_major","owned_voices"}`.
- `footage_map._cast_line(identity_map["persons"])` starts with `"CAST: "`.

### 8.2 Integration — re-ingest a real vcut project
1. Pick a project whose files have `face_tracks` (§8.1's `57b689b3…`).
2. Trigger a fresh vcut ingest (`defer_vcut_ingest` / the normal ingest entry).
   **Do not** do this as part of writing the plan — this is the verification step
   for the implementer after coding.
3. After it reaches `ready`, verify:
   ```sql
   select identity_map from ingest_runs where id = '<new_run_id>';
   ```
   → non-NULL, shape matches §2.6, `persons`/`voice_owner` populated.
4. Assert against the OLD-shape contract: top-level keys and person keys are
   identical to the `a38613c0…` dump in §2.6.
5. Render check: call `footage_map.assemble_map(file_ids, run_id=<new_run_id>)`
   and confirm the returned `text` begins with a `CAST:` line.

### 8.3 Fail-open regression
1. Re-ingest the demo-trail project `8621c012…` (no face_tracks).
2. Verify the run reaches `ready` and `identity_map` stays **NULL** (no write,
   no error), and `assemble_map` `text` has **no** `CAST:` line — byte-identical
   to pre-change output.

### 8.4 Idempotency / energy re-resolve
1. Hit the energy re-resolve endpoint (`routers/projects.py`) for a reconciled
   run; confirm `identity_map` is unchanged (reconciliation is NOT wired into the
   re-resolve path, only into `run_vcut_ingest`).

### 8.5 Acceptance criteria
- A re-ingested vcut project with face_tracks has a populated `identity_map`
  whose JSON shape matches the old runs exactly.
- The brain's Tier-0 index shows the `CAST:` line for that run.
- Projects without the L1 signals stay NULL with no errors and unchanged output.
- Zero changes to any `l3/identity/*` module or vcut Pass-2.
