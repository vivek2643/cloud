# Cumulative Identity Map — deterministic "show the speaker"

## Why

`on_camera` today is a **per-still LLM guess** ("does the visible person match the
diarized speaker"), decided independently for every cut. It's unreliable: on a
2-camera / 2-person shoot the physically-only-possible split for a beat is
**1 on-cam + 1 off-cam** (the speaker's camera on, the listener's off), yet
measured runs produce (2,0) — *both* cameras claim the speaker — and (0,2) —
*neither* does. Those are impossible; the flag is noise. As a result the editor
often shows the listener, not the talker.

The fix is to stop guessing per frame and instead **reconcile identity once,
cumulatively, after cuts**, from signals we already have, then *derive*
`on_camera` deterministically. Two independent facts must be established:

1. **Voice ↔ face (who is *talking*)** — solved by correlating each camera's
   subject motion (`action_energy`) with the authoritative diarization turns:
   the person who's talking moves more during their own turns. Deterministic,
   physical, aggregated over the whole clip.
2. **Cross-file *person* identity (who is *shown*, unified across files)** —
   solved with **structured appearance characteristics** (categorical
   attributes the LLM emits per cut; code clusters them). Deterministic matching
   over LLM-authored categorical data.

Together these produce a global identity map that finally populates
`footage_map`'s already-built-but-dead `oncam` / `alias` slots, and rewrites
`on_camera` from a coin-flip into a stable, derived fact → automatic
1-on/1-off, and the brain always knows the speaker's angle.

### Design principle (unchanged house rule)

- **Code owns everything quantitative & structural**: the motion↔turns
  correlation, the clustering, the assignment, the `on_camera` derivation.
- **The LLM owns only categorical *description***: it says "hair: bald,
  facial_hair: beard" per cut. It never assigns ids, scores, or matches people.
- **Deterministic keep / never fabricate**: when evidence is ambiguous (energies
  too close, fingerprints too sparse or too similar to tell apart), **do not
  force** a binding/merge — fall back to the existing per-still `on_camera` (or
  leave `unknown`). A wrong identity is worse than an honest unknown.

---

## Current state (what exists, what's dead)

- **Per-cut identity guess**: `pass2a.IdentityCut.on_camera` (bool|None) — the
  unreliable per-still flag. Kept as fallback only.
- **Per-cut appearance**: `pass2b.PersonLook` (`description` free prose,
  `position`, `speaking`), collected in `VisualJudgment.people`, carried to
  `CutRecord.characteristics` (via `post.assemble_cut_records`, `cut.people`),
  surfaced to the brain via `cutrecord_map._people_for`.
- **Motion signal**: L1 `motion_dynamics` → `action_energy` (per-hop subject
  motion, camera motion removed) + `hop_ms`, loaded at ingest as
  `motion_by_file[fid]["action_energy"]` / `["hop_ms"]`.
- **Diarization turns**: `Lattice.turns` = `[(start_ms, end_ms, speaker), …]`,
  already re-based onto each angle's clock inside an outlook group
  (`sync.lattice_merge.authoritative_view`).
- **Outlook grouping**: `ingest.run_ingest` computes `outlook_group_by_file`
  `{file_id: group_id}` (post the all-pairs sync fix) — the set of cameras that
  share one authoritative audio.
- **DEAD reconciled-cast machinery** (the receiving socket, currently never
  fed):
  - `footage_map._shown_and_cam(file_id, handle, alias, oncam)` derives on/off
    cam from `oncam = {file_id: shown_person}` and `alias = {(file_id, voice):
    global_person}`.
  - `footage_map._pic_who` / `_speaker_handle` consume them.
  - `footage_map.assemble_map` calls `_clip_block(t, compact=compact)` — **never
    passing `oncam`/`alias`**, so they're always `None`. This plan fills them.

---

## Phase 0 — Structured appearance descriptor (LLM side)

Give the LLM categorical fields so code can match exactly, instead of parsing
free prose.

**`pass2b.PersonLook`** — add stable-attribute fields (all optional, categorical,
`ConfigDict(extra="forbid")` stays):

```python
class Appearance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    apparent_gender: str | None = None   # "male" | "female" | "unsure"
    apparent_age_band: str | None = None # "child"|"teen"|"20s"|"30s"|"40s"|"50s"|"60s+"|"unsure"
    hair: str | None = None              # "bald"|"very_short"|"short"|"medium"|"long"|"unsure"
    hair_color: str | None = None        # "black"|"brown"|"blonde"|"grey"|"white"|"red"|"unsure"
    facial_hair: str | None = None       # "none"|"stubble"|"moustache"|"beard"|"goatee"|"unsure"
    glasses: str | None = None           # "yes"|"no"|"unsure"
    skin_tone: str | None = None         # "light"|"medium"|"tan"|"dark"|"unsure"
    build: str | None = None             # "slim"|"average"|"heavy"|"unsure"

class PersonLook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str                     # keep the free prose (display only)
    appearance: Appearance = Field(default_factory=Appearance)   # NEW, matching
    position: str | None = None
    speaking: bool | None = None
```

**Prompt (`pass2b._SYSTEM`)**: extend the `people` instruction to also fill
`appearance` with the categorical fields above, using **only stable identity
traits** (hair, facial hair, build, skin tone, glasses, apparent age/gender) —
**never clothing, pose, or current action** (those are volatile and must not
enter the fingerprint). Use `"unsure"`/omit when a trait isn't clearly visible.

**Carry-through**: `characteristics` (= `cut.people`) already flows to
`CutRecord.characteristics` verbatim, so the new `appearance` object rides along
with no change to `post` — confirm the dict round-trips (it will, it's nested in
the `people` dicts).

---

## Phase 1 — Voice↔camera binding (motion, deterministic)

New module: **`backend/app/services/l3/identity/bind.py`**.

Input per file: `Lattice.turns` (re-based) + `motion_by_file[fid]`
(`action_energy`, `hop_ms`).

```
mean_energy_during(file, voice) =
    mean(action_energy[h] for hop h whose [h*hop_ms,(h+1)*hop_ms) overlaps a turn of `voice`)
```

**Per file → bound voice** = `argmax_voice mean_energy_during(file, voice)`,
**only if** the margin between the top and second voice is meaningful:
`(top - second) / (top + eps) >= BIND_MARGIN` (start `BIND_MARGIN = 0.15`, a
tunable module constant; below it → `bound_voice = None`, unknown).

**Outlook-group bipartite refine**: within each `outlook_group_by_file` group,
build the `cameras × voices` mean-energy matrix and solve a max-weight
assignment (`scipy.optimize.linear_sum_assignment`, already a dep via detect's
scipy) so two cameras **cannot** bind to the same voice. Keep a per-camera
`confidence` = its assignment margin; drop (→ unknown) any camera whose margin is
below `BIND_MARGIN`.

Output: `bound_voice_by_file: {file_id: voice|None}` + `bind_confidence_by_file`.

**Edge cases** (fail-open, never fabricate):
- File with no turns, or empty `action_energy` → `None`.
- Single-speaker file → that speaker (trivially bound).
- Single camera framing *two* people → whole-frame `action_energy` can't
  separate them → `None` (leave per-still `on_camera` in place). Detect via >1
  distinct `people` position across the file's cuts *and* not in an outlook
  group; document as a known limitation, not a hack.

---

## Phase 2 — Cross-file person unification (structured characteristics)

Same module: **`identity/reconcile.py`**.

**2a. Per-file fingerprint** — each camera frames one person; aggregate its cuts'
`appearance` objects into ONE fingerprint by **majority vote per stable field**
(ignore `"unsure"`/`None` when voting; a field with no clear majority → unset).
This denoises per-cut variation. Use ONLY stable fields (Phase 0 list); clothing/
pose/action are never present in `appearance` by construction.

**2b. Cluster files → persons** — deterministic agglomeration over fingerprints:
distance = count of stable fields that are **both set and disagree** (unset on
either side = no evidence, distance 0 for that field). Merge two files iff
`disagreements == 0` AND `shared_set_fields >= MIN_SHARED_FIELDS` (start `3`).
Assign each cluster a `person_id` (`P0`, `P1`, … stable by first appearance) and
a display label from the longest `description` prose in the cluster.

- **No forced merge**: files whose fingerprints are too sparse
  (`shared_set_fields < MIN_SHARED_FIELDS`) or that disagree stay their **own**
  person. Honest over-splitting beats a wrong merge (the documented 1%).

**2c. Compose the maps**:
- `file_person: {file_id: person_id}` — from clustering (2b).
- `oncam: {file_id: person_display}` — `file_person` → display label. This is
  "whose face this camera shows".
- `alias: {(file_id, voice): person_display}` — for a file's **bound voice**
  (Phase 1): that file's `person_display`. For the **other** voice(s) heard in a
  grouped file, resolve via the group's *other* camera's bound voice→person (the
  two cameras of an outlook group cover both voices), then unify across groups by
  `person_id`. A voice with no resolvable person → omit (stays unresolved, same
  as today).

---

## Phase 3 — Derive `on_camera`, persist the map

New module: **`identity/apply.py`**, called from `ingest.run_ingest` **after
`pass2.*` and before/at `post.assemble_cut_records`**.

**3a. Rewrite `on_camera`** per Pass2 cut (deterministic; overrides the per-still
guess ONLY when we have a binding):
- Speech cut, file has `bound_voice` and cut has a `speaker`:
  `on_camera = (cut.speaker == bound_voice)`. Multi-speaker (`"S0,S1"`):
  `on_camera = bound_voice in speakers`.
- File has no binding (`None`) → **leave `cut.on_camera` as pass 2a set it**
  (fallback). Never null-out an existing guess.
- Video cut (no speaker) → unchanged.

**3b. Persist the identity map** for the run so `footage_map` can load it.
Migration: add **`ingest_runs.identity_map jsonb null`** (or a dedicated
`identity_maps` table keyed by `ingest_run_id`; either is fine — one row of
JSON). Shape:

```json
{
  "persons": [{"person_id":"P0","display":"curly, clean-shaven man","fingerprint":{...}}],
  "file_person": {"<file_id>":"P0"},
  "bound_voice": {"<file_id>":"S0"},
  "oncam": {"<file_id>":"curly, clean-shaven man"},
  "alias": {"<file_id>|S0":"curly, clean-shaven man"}   // key = "file_id|voice"
}
```

Write it in `run_ingest` right after the maps are computed (a
`store.set_identity_map(ingest_run_id, payload)` helper in `l3/store.py`).
`on_camera` is already written through `assemble_cut_records` from the mutated
Pass2 cuts, so cut_records pick up the derived value with no schema change.

---

## Phase 4 — Fill the dead `oncam`/`alias` (brain side)

**`footage_map.assemble_map`**: load the run's `identity_map` (via
`store.get_identity_map(run_id)`), rebuild `oncam` (`{file_id: display}`) and
`alias` (`{(file_id, voice): display}`, splitting the `"file_id|voice"` key),
and pass them into `_clip_block(..., alias=alias, oncam=oncam)`. That's the whole
integration — `_shown_and_cam` / `_pic_who` / `_speaker_handle` already consume
them. `None` map (older runs / no identity map) → behaves exactly as today.

Result: `PIC` resolves to the real shown person from the reconciled cast (right
even where a per-cut flag was wrong), `alt-PIC` labels become real people, and
"show the speaker" is answerable because on-cam is derived from the binding.

---

## Files to touch

| File | Change |
|---|---|
| `l3/pass2b.py` | add `Appearance` model + `PersonLook.appearance`; extend `_SYSTEM` people instruction (stable traits only) |
| `l3/identity/bind.py` (new) | motion↔turns voice binding + bipartite refine |
| `l3/identity/reconcile.py` (new) | per-file fingerprint, cluster→persons, compose `oncam`/`alias` |
| `l3/identity/apply.py` (new) | rewrite `on_camera`; build the persisted payload |
| `l3/ingest.py` | call bind → reconcile → apply after `pass2.*`; persist map |
| `l3/store.py` | `set_identity_map` / `get_identity_map` |
| `migrations/…` | `ingest_runs.identity_map jsonb` (or `identity_maps` table) |
| `l3/footage_map.py` | `assemble_map` loads map, passes `oncam`/`alias` to `_clip_block` |
| `l3/cutrecord_map.py` | (optional) surface `appearance` in `_people_for` if useful to the brain; `on_camera` already flows through |

No changes to pass 1, pass 2a's schema, the sync layer, or cuts logic.

---

## Determinism & fallback summary

- Every numeric/structural decision (binding, clustering, assignment,
  `on_camera`) is code. The LLM only emits categorical `appearance`.
- **Never fabricate**: unknown binding → keep per-still `on_camera`; unmergeable
  fingerprints → separate persons. Two tunables gate this: `BIND_MARGIN`,
  `MIN_SHARED_FIELDS`.
- Older ingest runs with no `identity_map` are byte-identical to today (all
  fallbacks engage).

## Known 1% (documented, not patched over)

- Look-alikes / identical uniforms → fingerprints collide → possible mis-merge.
- A single camera framing two people → no motion separation → binding `unknown`
  (per-still fallback).
- Very sparse `appearance` on a camera → stays its own person (over-split, not
  mis-merged).

## Verification

1. **Unit**: synthetic turns + `action_energy` → `bind` returns the higher-energy
   voice; margin below `BIND_MARGIN` → `None`. Synthetic fingerprints → clustering
   merges only on zero-disagreement + enough shared fields.
2. **Real re-ingest** (podcast, 2 outlook groups): per 2-angle beat the on/off
   split becomes **(1,1)** — no (2,0)/(0,2); aggregate `on_camera` → ~50/50.
   `identity_map.persons` has exactly **2** people, each mapped to both its
   cameras across the two blocks (cross-file unification worked).
3. **Brain**: `footage_map` `PIC:` resolves to real persons (no `PIC:?` on
   grounded speech beats), `alt-PIC` names the alternate angle's person.

## Non-goals

- No speaker embeddings, no face-recognition model, no per-speaker audio routing.
- Sync/cuts unchanged. Person *naming* is display-only (never an identity claim).
- Level-2 (per-speaker audio) and single-camera-two-people separation are future.
