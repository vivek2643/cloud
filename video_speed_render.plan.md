# Video-speed render pass (deferred)

The `retime` verb is fully wired on the brain side. For SPEECH it applies today
(dead-air/filler trim = tighter source spans, which the pipeline already
renders). For VIDEO it currently only **records** the chosen `speed` on the
segment (`seg["speed"]`, `seg["pace_level"]`) and surfaces it in `read_state`,
because the render + resolve geometry assumes program-time == source-time (1:1)
everywhere. This plan is the deferred work to make video speed actually play.

## Why it's deferred (the 1:1 assumption)
`layers.spine_spans` sets `prog_dur = src_out - src_in`. That 1:1 map is baked
into `prog_to_source`, `_slice_video`, `_apply_split_edits`, `_dest_spine_window`
and the layout-region slicing, and it's mirrored in the frontend
`resolve-timeline.ts`. The compositor is explicitly "hard cuts only -- no
transitions/speed/text yet". So a real speed factor has to thread through all of
that consistently, or the declared timeline length and the exported video
disagree.

## Surfaces to change
1. **Backend resolve (`layers.py`)**
   - Add `speed` (float, default 1.0) to `VideoLayer` + `AudioLayer` (+ `to_dict`).
   - `spine_spans`: `prog_dur = round((src_out - src_in) / speed)`; read
     `seg.get("speed", 1.0)`.
   - Make the source<->program map speed-aware everywhere it's currently 1:1:
     `prog_to_source` (src = in + (prog - prog_start) * speed), `_slice_video`,
     `_apply_split_edits` (offset math), `_dest_spine_window`, layout slicing.
   - Decision: SCOPE video-speed to plain V1 spine cuts that are NOT under a
     layout region and NOT a split-edit seam participant (validate rejects speed
     there) to keep the geometry tractable in v1 of this pass; widen later.
2. **ffmpeg (`render/compositor.py`)**
   - Spine-concat fast path: `_produce_segment` applies `setpts=PTS/speed`
     (video) + `atempo=speed` (audio, chain for >2x/<0.5x). Add `speed` to
     `_segment_cache_path` key so cached segments don't collide.
   - Layered path: same `setpts`/`atempo` per layer before the shift-to-program.
3. **Frontend (`resolve-timeline.ts` + preview player)**
   - Mirror the speed-aware `spineSpans` geometry (keep it identical to backend).
   - Preview: set `video.playbackRate = speed` per spine segment during playback
     (mirrors `cuts-v3-view.tsx`'s `paceRate`).
4. **Cleanup of the phase-1 flag**
   - Once render honors speed, drop the `speed_note`/"recorded; not yet applied"
     wording in `observe.read_state` and update the `retime` tool + `_LOOP_SYSTEM`
     copy ("recorded, not yet baked" -> "applied").
   - `predict` should then also account for video speed in projected length.

## Tests to add
- `layers.resolve` program length halves at speed 2x for a single spine cut;
  audio stays coupled.
- compositor cache key differs by speed; a smoke render at 1.5x produces the
  expected duration.
- validate rejects `speed != 1.0` on a segment under a layout region / split
  seam (until the geometry supports it).

## Not in scope here
Speech is never sped (retime already trims dead-air instead) -- this pass is
video-only.
