# Client-side analysis proxies + early L2 — plan

**Goal:** cut the wall-clock time from "user drops a file" to "clip is fully
analyzed", on the **desktop web app**, without touching output quality.

Today the ingest path is serial and gated on the raw upload:

```
raw upload (client → R2, minutes for a multi-GB file)
  → worker RE-downloads the whole raw (R2 → worker)
    → ONE overloaded 1080p editing proxy (full 4K→1080p transcode)
      → all of L1 (speech | audio | motion)
        → L2 (Gemini) — only fires AFTER motion, which it never uses
```

Two levers, both pure latency wins:

- **Generate the two TINY analysis proxies on the client**, from the local file,
  *while the raw is still uploading*. Analysis inputs become a few MB that upload
  in seconds — so analysis is **decoupled from the multi-GB raw upload**.
- **Fire L2 as soon as its inputs exist** (the 480p proxy + the speech track),
  overlapping the Gemini round-trip with the rest of L1 instead of waiting for
  motion (which L2 does not read).

The **1080p editing proxy stays server-side, exactly as it is today** — it's only
needed for smooth scrub playback, not for analysis, so it rides along unchanged.

---

## The proxies

One client-side decode (at the higher, 10fps rate) feeds BOTH analysis proxies +
audio, so the expensive decode is paid once.

| Artifact | Spec | Serves | Made where |
|---|---|---|---|
| **Proxy A** (analysis + audio) | 480p video @ **1 fps** + full **audio** (AAC) | L2 Gemini perception **and** (server demuxes WAV from it) transcript / audio_features / diarization / dialogue_segments | **client** |
| **Proxy B** (motion) | **160×90 @ 10 fps**, video-only | motion_dynamics (optical flow) | **client** |
| Editing proxy | 1080p, CFR, ~1s keyframes, faststart, AAC | scrub/playback in the editor | **server (unchanged)** |

Why these specs:
- The audio/speech L1 stages need **audio only** — no video frames — so Proxy A's
  audio track covers all of them; its 1fps video is just for Gemini.
- Motion needs the **10 fps cadence** (optical flow between ~100ms-apart frames)
  but only **160×90** resolution — it downsamples internally anyway. So Proxy B is
  tiny in bytes even at 10 fps.
- A 1fps proxy is enough for L2 (Gemini samples at `l2_video_fps=1.0`) but would
  break motion — hence two proxies, not one.

---

## Sequencing

```
CLIENT (desktop, on the local File — does not wait for the upload):
  pick file
   ├─ start raw upload (background; minutes for 2GB) ───────────────┐
   └─ decode local file ONCE @10fps                                 │
        ├─ Proxy A (480p@1fps + audio)                              │
        └─ Proxy B (160×90@10fps)                                   │
             └─ upload Proxy A + Proxy B FIRST (tiny, seconds)      │
                                                                    │
SERVER (runs on the proxies, NOT the raw):                         │
  demux WAV from Proxy A                                            │
   ├─ speech track:  transcript → diarization → dialogue_segments  │
   ├─ audio  track:  audio_features → beat_cut                     │
   └─ motion track:  motion_dynamics (Proxy B)                     │
  ── L2 fires when (Proxy A uploaded  AND  speech track done) ──►   │
        Gemini perception  (overlaps the motion track)             │
  dialogue_cut  after speech+audio join                            │
                                                                    │
SERVER (on the raw, as today — deferred / background): ◄───────────┘
  1080p editing proxy + thumbnail
```

Key property: **the 2 GB raw upload is off the analysis critical path.** Analysis
starts as soon as the ~few-MB proxies land, not when the raw finishes.

---

## Phases

| Phase | Goal | Work | Guard / risk | Verify |
|---|---|---|---|---|
| **A — client dual-proxy** | make Proxy A + Proxy B in-browser | WebCodecs: one shared 10fps decode of the local file → mux (mp4box.js/mp4-muxer) two tiny outputs + audio | HEVC 4K decode is patchier than H.264 → detect + fall back to server proxy gen; weak machine → same fallback | proxies play; sizes are a few MB; total gen time < upload time |
| **B — upload ordering** | analysis inputs before the raw | upload Proxy A + Proxy B (+ ids) first; raw uploads in background; keep the local `File` for instant local playback | ensure server can start L1 without the raw present | server begins L1 with only proxies uploaded |
| **C — server accepts client proxies** | point stages at the proxies, keep 1080p as-is | demux WAV from Proxy A; motion reads Proxy B; **1080p editing proxy still generated server-side from the raw, deferred** | if a client proxy is missing (fallback path), server regenerates it from the raw | L1 runs identically on client- vs server-made proxies |
| **D — early L2 trigger** | L2 overlaps motion | move the L2 enqueue to fire after **speech track + Proxy A**, not after the full L1 join | keep AV speaker fusion + off-camera flagging correct → gate on diarization done, not just proxy | L2 starts while motion is still running; outputs unchanged |

---

## Expected timing — 2 GB / 4K / 10-min clip

Assumptions: desktop Chromium w/ HW decode, H.264, ~50 Mbps uplink, server GPU
(or Groq for ASR). Ranges, not promises.

| Phase | Time |
|---|---|
| Raw upload (2 GB @ 50 Mbps) — **background** | ~5–6 min (does NOT gate analysis) |
| Client proxy+audio gen (one 10fps decode) | ~20–45 s (hidden under the upload) |
| Tiny proxies upload (~15–30 MB, sent first) | ~3–5 s |
| Server L1 (parallel tracks) | ~30–60 s wall |
| L2 Gemini (fires after speech track; overlaps motion) | ~15–30 s |
| **Analysis ready** | **~90 s – 2 min from drop** |

vs today (~8–12 min): wait for full 2 GB upload → re-download 2 GB → full 4K→1080p
transcode → L1 → L2. Roughly a **5–6× reduction**, mostly from decoupling analysis
from the raw upload and letting the client absorb the decode.

Note: motion becomes cheap on the server because its real cost was always the
*decode* — which the client already paid to make Proxy B.

---

## Caveats (desktop-scoped, manageable — not N-browser fallbacks)

- **Codec:** H.264 4K decodes well in-browser; **HEVC/H.265 4K** support varies by
  OS/GPU → the one case that may fall back to server-side proxy gen.
- **Muxer needed:** WebCodecs emits frames, not `.mp4` — pull in mp4box.js /
  mp4-muxer (demux source + mux outputs).
- **Memory:** decode-and-stream a 4K source; don't buffer all frames in RAM.
- **Guaranteed baseline:** keep the existing **server-side proxy generation** as
  the fallback for any file/machine the client can't handle. Client gen is an
  accelerator, never the only path.
- **Playback:** the 1fps proxy isn't scrub-smooth; desktop web can play the local
  raw via a blob URL instantly, and the 1080p editing proxy (unchanged) covers
  post-upload playback.

---

## Out of scope (now)

- Client-side **editing** proxy (fragile at full res) — stays server-side.
- Self-hosting the VLM locally; chunk/fan-out within a file; any cost/model-routing
  work (e.g. the Opus thought-segmentation) — separate threads.
- Mobile / non-desktop browsers — this plan assumes the desktop web app.
