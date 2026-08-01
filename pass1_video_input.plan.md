# pass1_video_input.plan.md — Feed Pass 1 non-speech VIDEO (cut speech), cached per file

## 0. Goal & non-goals

**Goal.** Replace Pass 1's sampled-JPEG input with the actual **non-speech video** of each
file, sent to Gemini per file (matching the just-landed per-file Pass 1 fan-out). Cut out
the speech regions before sending (cost ∝ non-speech seconds, not total duration; also drops
the audio track), downsample cheaply (`media_resolution: low`, a chosen fps), and — in a
second phase — **cache the per-file clip** so Pass 1 and Pass 2 both read it at the discounted
rate.

**Why.** (a) Motion continuity — the model sees the *action* (build/settle, camera moves,
"walks up to the phone"), which sampled stills throw away; (b) cost — cutting speech saves
proportional to the speech fraction and strips ~12% audio tokens; (c) it composes cleanly
with per-file Pass 1 (one clip → one call → bounded output).

**Cost frame of reference (flash-lite, established earlier):** frames-vs-video is a red
herring — cost ≈ fps × tokens/frame × seconds. `media_resolution: low` ≈ 66 tok/frame (~4× off
default), audio ≈ 32 tok/s. Non-speech sub-clip at low-res, ~2 fps ≈ **$0.01/30-min file**.

**Non-goals.**
- No change to the resolver, seam function, speech channel, or `MomentPlan` shape.
- No change to what Pass 1 *emits* (still flag-only `t_ms/shape/summary`). Only the INPUT
  medium and the timestamp coordinate space change.
- Keep the sampled-frame path intact as a **fallback** (any upload/cut/cache failure → frames).

**Principles.** vcut isolation stays (only `l3.post.CutRecord`/`ingest_store`/generic `l3.frames`
+ the generic `llm` adapter are imported); every knob in `params.py`/config; the video path
must fail-open to frames, never hard-fail an ingest.

---

## 1. Architecture

Per file, inside `run_vcut_ingest`:

```
proxy + non_speech_spans
   │  (§2) ffmpeg: trim spans → concat → low-res, target-fps, NO audio sub-clip
   ▼
non_speech_subclip.mp4  +  time-map [(orig_start,orig_end,sub_start), ...]   (§3)
   │  (§4) Files API upload → wait ACTIVE → file handle (uri, name)
   ▼
Pass 1 call (§5): complete_gemini(video_block(uri, fps, media_resolution), Pass1FileSchema)
   │  model returns t_ms in SUB-CLIP time
   ▼
remap sub_ms → orig_ms (§3) ; drop join-boundary artifacts ; _clamp_moment to non-speech spans
   ▼
FilePlan(flags) → (merge) MomentPlan  → resolver UNCHANGED
```

Phase 2 (§6) inserts a per-file **CachedContent** between upload and the calls, reused by Pass 1
and Pass 2. Uploaded files + caches are torn down at run end (§7).

---

## 2. Cut the non-speech sub-clip (`backend/app/services/vcut/subclip.py`, NEW)

Pure-ish helper (ffmpeg + R2 I/O, no LLM). One function:

```python
@dataclass
class SubClip:
    path: str                      # local temp mp4
    segments: List[Tuple[int, int, int]]  # (orig_start_ms, orig_end_ms, sub_start_ms)
    duration_ms: int

def cut_non_speech_subclip(proxy_key: str, non_speech_spans: List[Tuple[int,int]],
                           *, fps: float, width_px: int) -> Optional[SubClip]:
    ...
```

- **Fetch the proxy** the same way `sampling.py`/`l3.frames` already do (reuse their R2→local
  primitive; do NOT re-implement R2 auth). A byte-range isn't enough (spans are scattered) —
  pull the whole proxy to a temp file (proxies are small).
- **ffmpeg**: trim each span, reset PTS, concat, scale to `width_px`, force `fps`, **`-an`**
  (strip audio), `libx264 -preset veryfast`. Sketch:

  ```
  ffmpeg -i proxy.mp4 -filter_complex
    "[0:v]trim=start=S0/1000:end=E0/1000,setpts=PTS-STARTPTS[v0]; ... ;
     [v0][v1]...concat=n=K:v=1:a=0[c]; [c]scale=WIDTH:-2,fps=FPS[outv]"
    -map "[outv]" -an -c:v libx264 -preset veryfast -movflags +faststart out.mp4
  ```

  Frame-accurate cuts via the `trim` filter (not `-ss -c copy`, which snaps to keyframes).
- **Build `segments`**: iterate spans in order; `sub_start_i = Σ (orig_end_j-orig_start_j)` for
  j<i. This is the whole time-map (§3).
- **Guards**: if there are no non-speech spans, or total non-speech < a floor
  (`SUBCLIP_MIN_MS`), return `None` (caller falls back to frames). If ffmpeg fails, log +
  return `None`. Always clean up the temp proxy; the SubClip's own temp file is deleted by the
  caller after upload.
- **Cap**: bound total sub-clip duration to `SUBCLIP_MAX_MS` (evenly drop/likely just trust
  spans) so a pathological all-non-speech 2-hour file can't blow up upload/token cost; the fps
  is already the main lever.

---

## 3. Time remapping (in `subclip.py`, PURE)

```python
def map_sub_to_orig(sub_ms: int, segments: List[Tuple[int,int,int]]) -> Optional[int]:
    """Sub-clip ms -> original-file ms. None if it lands in no segment
    (shouldn't happen for an in-range t)."""
    for orig_s, orig_e, sub_s in segments:
        seg_len = orig_e - orig_s
        if sub_s <= sub_ms < sub_s + seg_len:
            return orig_s + (sub_ms - sub_s)
    return None

def is_near_join(sub_ms: int, segments, guard_ms: int) -> bool:
    """True if sub_ms is within guard_ms of any concat boundary -- the hard
    cut between two spliced spans is an ARTIFACT, not a real moment."""
```

Both pure → unit-testable with synthetic segments.

---

## 4. Gemini adapter: video block + Files API (generic layer)

### 4.1 Neutral block (`backend/app/services/llm/base.py`)

```python
def video_file_block(file_uri: str, *, mime_type: str = "video/mp4",
                     fps: Optional[float] = None,
                     start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> Block:
    return {"type": "video_file", "file_uri": file_uri, "media_type": mime_type,
            "fps": fps, "start_ms": start_ms, "end_ms": end_ms}
```

### 4.2 Converter (`backend/app/services/llm/gemini_client.py::_parts_for_content`)

Add a branch alongside `text`/`image`:

```python
elif btype == "video_file":
    vm = None
    if b.get("fps") or b.get("start_ms") is not None or b.get("end_ms") is not None:
        vm = types.VideoMetadata(
            fps=b.get("fps"),
            start_offset=_secs(b.get("start_ms")), end_offset=_secs(b.get("end_ms")),  # "1.5s" strings
        )
    parts.append(types.Part.from_uri(file_uri=b["file_uri"], mime_type=b.get("media_type","video/mp4"),
                                     video_metadata=vm))
```

(Confirm the installed `google-genai` `VideoMetadata`/`Part.from_uri` field names against the
SDK; degrade gracefully if `video_metadata` isn't accepted — log + send without it.)

### 4.3 Upload + lifecycle (`backend/app/services/llm/ingest_gemini.py`)

```python
def upload_video(path: str, *, mime_type: str = "video/mp4", poll_s: float = 1.0,
                 timeout_s: float = 120.0) -> Optional[dict]:
    """Upload a local clip, poll until state==ACTIVE, return {uri, name}. None on
    failure/timeout (caller falls back to frames)."""
    # client.files.upload(file=path, config={"mime_type": mime_type})
    # loop client.files.get(name=...) until state == "ACTIVE" or FAILED/timeout

def delete_uploaded_file(name: Optional[str]) -> None:
    # best-effort client.files.delete(name=name); files also auto-expire (~48h)
```

### 4.4 `media_resolution` passthrough

`complete_gemini` and `create_pass2_cache` currently don't set `media_resolution`. Add an
optional `media_resolution: Optional[str] = None` param threaded into `_build_config`
(`types.GenerateContentConfig(media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)`) and
into `CreateCachedContentConfig`. Default None = SDK default; vcut passes `"low"`.

### 4.5 Video cache (Phase 2 only, §6)

```python
def create_video_cache(system: str, file_uri: str, *, fps, media_resolution, model, ttl_seconds) -> Optional[str]:
    # client.caches.create(model=..., config=CreateCachedContentConfig(
    #   contents=[Content(role="user", parts=[Part.from_uri(file_uri, video_metadata=...)])],
    #   system_instruction=system, ttl=f"{ttl_seconds}s"))
```

Reuse existing `delete_pass2_cache` for teardown.

---

## 5. Pass 1 uses video (`backend/app/services/vcut/pass1.py`) — PHASE 1

`run_pass1` already fans out per file. Change the per-file worker (`_run_pass1_for_file`) to,
**when video mode is on and a sub-clip + upload succeeded**, send a `video_file_block` instead
of `build_frame_blocks(...)`:

- **Input:** the file's `SubClip` + uploaded `file_uri`, plus `fps` and `media_resolution` from
  config.
- **Prompt:** keep `_task_text()` flag-only, but add ONE line: *"You are shown one continuous
  clip assembled from this file's footage; give each moment's `t_ms` as milliseconds from the
  START of THIS clip."* (timestamps are sub-clip time; we remap).
- **Call:** `complete_gemini(_task_text(...), [video_file_block(uri, fps=FPS)], _Pass1FileSchema,
  model=vcut_pass1_model, media_resolution="low", max_tokens=_PER_FILE_MAX_TOKENS, thinking="low")`.
- **Post-process (NEW, per file):** for each returned moment with sub-clip `t_ms`:
  1. drop it if `is_near_join(t_ms, segments, JOIN_GUARD_MS)` (concat artifact),
  2. `orig = map_sub_to_orig(t_ms, segments)`; drop if None,
  3. build `_MomentOut(t_ms=orig, ...)` and run the existing `_clamp_moment` against the file's
     real non-speech spans (unchanged safety net).
- **Fallback:** if video mode is off, or `cut_non_speech_subclip`/`upload_video` returned None,
  fall back to the EXISTING `build_frame_blocks` inline-frames path for that file. Per-file, so
  one un-cuttable file silently uses frames while the rest use video.

`run_pass1`'s signature gains the per-file `SubClip`/upload info (or a small
`video_by_file: Dict[str, VideoHandle]` arg built by orchestrate). Merge/concurrency/failure
isolation from the per-file refactor are unchanged.

---

## 6. Pass 2 reuse of the per-file video cache — PHASE 2 (separate, optional)

The read-twice discount (Pass 1 + Pass 2 on one cache) only pays once BOTH passes read the
same cached clip. Phase 2:

- Orchestrate creates ONE `create_video_cache` per file (its non-speech clip), stores
  `{file_id: cache_handle}` via a generalized `store.persist_vcut_cache` (today it stores a
  single handle — widen to a per-file map, back-compat on read).
- Pass 1 passes `cached_content=that file's cache` instead of an inline `video_file_block`.
- `pass2.run_enrich` looks up the cut's `file_id` → that file's video cache and calls with
  `cached_content=...` + a `video_file_block` scoped via `video_metadata` `start_ms/end_ms`
  around the cut window (so Pass 2 "looks at" just that cut's slice). This replaces Pass 2's
  hero-frame re-extraction for files that have a video cache; keep the frame path as fallback.
- Teardown all per-file caches + uploaded files at run end.

Phase 2 touches `pass2.py` + `store.persist_vcut_cache`/`load` + orchestrate; ship Phase 1
first and measure before doing it.

---

## 7. Orchestrate (`backend/app/services/vcut/orchestrate.py`) — PHASE 1

In `run_vcut_ingest`, replace the Pass-1 frame prep with video prep (guarded by config):

- If `vcut_pass1_input_mode == "video"`:
  - For each file: `sub = subclip.cut_non_speech_subclip(proxy_key, non_speech_by_file[fid],
    fps=VCUT_VIDEO_FPS, width_px=VCUT_VIDEO_WIDTH_PX)`; if `sub`,
    `h = ingest_gemini.upload_video(sub.path)`; collect `video_by_file[fid] = (sub, h)`;
    delete the local temp file after upload succeeds.
  - **Drop** the shared frame cache (`_create_vcut_cache`) and the Pass-1 `sample_frames_for_files`
    call. Pass 2 (Phase 1) then runs its EXISTING uncached hero-frame re-extraction path
    (already supported — see `_create_vcut_cache` docstring "Pass 2: re-extracted hero
    frames"). Note the minor transitional Pass-2 cost regression (no cache); Phase 2 restores it.
  - Call `p1.run_pass1(prompt_rows, non_speech_by_file, images_by_key={}, video_by_file=video_by_file)`.
  - After the run (finally-block, success OR failure): `ingest_gemini.delete_uploaded_file(h.name)`
    for every uploaded handle.
- Else (`"frames"`, the default until validated): today's exact path, untouched.

`seam_cache`, resolver, speech channel, `persist_seam_and_plan` — all unchanged.

---

## 8. Config / params

`backend/app/config.py` (Settings):
- `vcut_pass1_input_mode: str = "frames"` — flip to `"video"` to enable; lets us A/B and roll
  back instantly.
- `vcut_video_fps: float = 2.0`
- `vcut_video_media_resolution: str = "low"`
- `vcut_video_cache_ttl_s` (Phase 2) — reuse `vcut_cache_ttl_s`.

`backend/app/services/vcut/params.py`:
- `VCUT_VIDEO_WIDTH_PX = 640`
- `SUBCLIP_MIN_MS = 1500` (below this, use frames — too little to bother cutting/uploading)
- `SUBCLIP_MAX_MS = 20 * 60 * 1000` (safety ceiling)
- `JOIN_GUARD_MS = 200` (drop moments landing on a concat seam)

---

## 9. Fallbacks (must all fail-open to frames)

- No non-speech / clip too short / ffmpeg fails → `cut_non_speech_subclip` returns None → that
  file uses frames.
- Upload fails / never ACTIVE → that file uses frames.
- `video_file` part rejected by the SDK / model errors on video → per-file try/except (already
  there) logs and that file yields an empty `FilePlan`; a whole-project video failure is caught
  by the run's frames-mode default (keep `"frames"` as the shipped default until validated).
- Never let a video-path failure fail an ingest that frames would have completed.

---

## 10. Testing

Pure-unit (no network, no ffmpeg-in-CI unless a tiny fixture is cheap):
1. `test_vcut_subclip.py`: `map_sub_to_orig` round-trips across a 3-segment map; boundary
   points map correctly; out-of-range → None. `is_near_join` flags seams within `JOIN_GUARD_MS`.
   Segment-building math (`sub_start` cumulative) from synthetic spans.
2. `test_vcut_pass1.py` (extend): with a mocked `complete_gemini`, video mode remaps sub-clip
   `t_ms`→orig, drops join-artifact + out-of-range moments, and still `_clamp_moment`s; a
   None-subclip file falls back to the frames path (mock both).
3. Adapter: `_parts_for_content` builds a `video_file` part with/without `video_metadata`
   (mock `types`); `upload_video` polls to ACTIVE then returns handle, returns None on timeout
   (mock `client.files`).
4. Pyflakes on every touched module.

Live (spends money / does ffmpeg + upload — SEPARATE, do not auto-run): flip
`vcut_pass1_input_mode="video"` on ONE small project; confirm moments land in real non-speech
regions (remap correct), no seam-artifact moments, cost per file in the expected range, and
uploaded files are deleted at run end. A/B the same project frames-vs-video for moment quality.

---

## 11. File-by-file change list

| File | Phase | Change |
|---|---|---|
| `backend/app/services/vcut/subclip.py` (NEW) | 1 | `cut_non_speech_subclip` (ffmpeg trim+concat, low-res, `-an`), `SubClip`, `map_sub_to_orig`, `is_near_join`. |
| `backend/app/services/llm/base.py` | 1 | `video_file_block(...)`. |
| `backend/app/services/llm/gemini_client.py` | 1 | `_parts_for_content`: handle `"video_file"` → `Part.from_uri` + `VideoMetadata`. |
| `backend/app/services/llm/ingest_gemini.py` | 1 | `upload_video`, `delete_uploaded_file`, `media_resolution` passthrough in `complete_gemini`/`_build_config`; (Phase 2) `create_video_cache`. |
| `backend/app/services/vcut/pass1.py` | 1 | per-file worker sends `video_file_block` when a handle exists; sub-clip remap + join/clamp post-process; frames fallback. |
| `backend/app/services/vcut/orchestrate.py` | 1 | video prep loop (cut+upload), drop shared frame cache when video mode, upload teardown in finally; (Phase 2) per-file video cache lifecycle. |
| `backend/app/services/vcut/pass2.py` | 2 | look up per-file video cache, scope via `video_metadata` around the cut; frames fallback. |
| `backend/app/services/vcut/store.py` | 2 | `persist_vcut_cache`/load → per-file cache map (back-compat). |
| `backend/app/config.py` | 1 | `vcut_pass1_input_mode`, `vcut_video_fps`, `vcut_video_media_resolution`. |
| `backend/app/services/vcut/params.py` | 1 | `VCUT_VIDEO_WIDTH_PX`, `SUBCLIP_MIN_MS`, `SUBCLIP_MAX_MS`, `JOIN_GUARD_MS`. |
| `backend/scripts/test_vcut_subclip.py` (NEW), `test_vcut_pass1.py` | 1 | tests (§10). |

---

## 12. Rollout

1. Ship **Phase 1** behind `vcut_pass1_input_mode="frames"` (default). Land code + unit tests.
2. Flip ONE project to `"video"` locally; validate remap correctness, seam-artifact absence,
   cost, and moment quality vs frames (A/B). Tune `vcut_video_fps` / `media_resolution`.
3. If quality wins, default `"video"`. Then do **Phase 2** (per-file video cache reused by
   Pass 2) for the read-twice discount, and drop the frame-sampling + hero re-extraction paths
   once video is the committed default.
4. Re-ingest is a separate, money-spending step (not part of this plan). The frame path stays
   as the permanent fallback for un-cuttable files.
```

