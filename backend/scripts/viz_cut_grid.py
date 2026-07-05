#!/usr/bin/env python3
"""
Multi-channel cut-grid visualizer (READ-ONLY).

Renders a self-contained HTML page that overlays, on one shared time axis, every
L1 cut-cost channel we derive:

  - DIALOGUE -- "safe to cut" (1 - cost) area + RMS envelope + per-word boxes
    (coloured by diarization speaker, fillers outlined) + discrete seam points.
  - BEAT     -- safe-to-cut area; dots on each musical onset (cut ON the beat).
  - ACTION   -- safe-to-cut area; dots on each subject-motion impact.
  - CAMERA   -- safe-to-cut area (avoid channel: low when the camera moves/blurs).

So you can *eyeball* whether each channel's seams land where a human editor would
cut. No matplotlib, no R2 -- everything is pulled straight from Postgres. Output
is a portable .html file.

Run from the backend/ directory:

    python scripts/viz_cut_grid.py --list            # files that have a grid
    python scripts/viz_cut_grid.py --file-id <uuid>  # render one file
    python scripts/viz_cut_grid.py --demo            # synthetic data, no DB
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

# Make `app...` importable when run as `python scripts/viz_cut_grid.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Per-channel render colors (kept in sync with the legend in HTML_TEMPLATE).
CH_COLORS = {"beat": "#ffb454", "action": "#ff6e9c", "camera": "#56b6ff",
             "fused": "#9d7bff"}


def _fused_channel(duration_ms, dialogue, af, md, energy=0.5):
    """Build the FUSED seam field from whatever component grids exist, as a
    viz channel (cost + the ranked discrete seams as points)."""
    from app.services.l1.fused_seams import compute_fused_field

    field = compute_fused_field(
        duration_ms=duration_ms, energy=energy,
        dialogue_cost=(dialogue or {}).get("cut_cost"),
        dialogue_hop=(dialogue or {}).get("hop_ms", 100),
        dialogue_points=(dialogue or {}).get("cut_points"),
        camera_cost=(md or {}).get("camera_cut_cost"),
        action_cost=(md or {}).get("action_cut_cost"),
        action_points=(md or {}).get("action_points"),
        motion_hop=(md or {}).get("hop_ms", 100),
        beat_cost=(af or {}).get("beat_cut_cost"),
        beat_points=(af or {}).get("beat_cut_points"),
        beat_hop=(af or {}).get("beat_cut_hop_ms", 100),
    )
    return {
        "label": "FUSED", "color": CH_COLORS["fused"],
        "hop_ms": field.hop_ms, "cost": field.cost,
        "points": [{"ts_ms": s.ts_ms, "kind": s.kind, "score": round(s.q, 3)}
                   for s in field.seams[:80]],
    }


def _load_from_db(file_id: str) -> dict:
    import psycopg
    from psycopg.rows import dict_row
    from app.config import get_settings

    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        f = conn.execute(
            "select name, duration_seconds from files where id = %s", (file_id,)
        ).fetchone()
        if not f:
            raise SystemExit(f"No file {file_id}")
        tr = conn.execute(
            "select segments from transcripts where file_id = %s", (file_id,)
        ).fetchone()
        af = conn.execute(
            """
            select rms_db, prosody_hop_ms, f0_hz,
                   dialogue_cut_cost, dialogue_cut_hop_ms, dialogue_cut_points,
                   beat_cut_cost, beat_cut_hop_ms, beat_cut_points
              from audio_features where file_id = %s
            """,
            (file_id,),
        ).fetchone()
        md = conn.execute(
            """
            select hop_ms, action_cut_cost, camera_cut_cost, action_points
              from motion_dynamics where file_id = %s
            """,
            (file_id,),
        ).fetchone()

    duration_ms = int((f["duration_seconds"] or 0) * 1000)
    dialogue = None
    if af and af.get("dialogue_cut_cost"):
        words = []
        for seg in (tr["segments"] if tr else []) or []:
            for w in seg.get("words") or []:
                words.append({
                    "start_ms": w.get("start_ms", 0),
                    "end_ms": w.get("end_ms", 0),
                    "text": w.get("text", ""),
                    "is_filler": bool(w.get("is_filler", False)),
                    "speaker": w.get("speaker"),
                })
        dialogue = {
            "hop_ms": af["dialogue_cut_hop_ms"],
            "cut_cost": af["dialogue_cut_cost"] or [],
            "cut_points": af["dialogue_cut_points"] or [],
            "rms_db": af["rms_db"] or [],
            "prosody_hop_ms": af["prosody_hop_ms"] or 0,
            "f0_hz": af["f0_hz"] or [],
            "words": words,
        }

    channels = []
    if af and af.get("beat_cut_cost"):
        channels.append({
            "label": "beat", "color": CH_COLORS["beat"],
            "hop_ms": af["beat_cut_hop_ms"], "cost": af["beat_cut_cost"] or [],
            "points": af["beat_cut_points"] or [],
        })
    if md and md.get("action_cut_cost"):
        channels.append({
            "label": "action", "color": CH_COLORS["action"],
            "hop_ms": md["hop_ms"], "cost": md["action_cut_cost"] or [],
            "points": md["action_points"] or [],
        })
    if md and md.get("camera_cut_cost"):
        channels.append({
            "label": "camera/distortion", "color": CH_COLORS["camera"],
            "hop_ms": md["hop_ms"], "cost": md["camera_cut_cost"] or [],
            "points": [],
        })

    if not dialogue and not channels:
        raise SystemExit(
            "No cut grids for this file. Apply migrations 010/011 and re-index a "
            "clip, then retry."
        )

    channels.append(_fused_channel(duration_ms, dialogue, af, md))

    return {"name": f["name"], "duration_ms": duration_ms,
            "dialogue": dialogue, "channels": channels}


def _demo_data() -> dict:
    """Synthesize a believable clip and run the REAL derivations (dialogue + beat;
    motion is faked since it needs pixels), so the visualizer can be validated
    without any DB or re-index."""
    from app.services.l1 import cut_cost as cc
    from app.services.l1 import beat_cost as bc
    from app.services.l1.cut_grid_common import hit_cost_curve

    duration_ms = 6000
    hop = 100

    words = [
        {"start_ms": 200,  "end_ms": 600,  "text": "So",        "is_filler": False, "speaker": "S0"},
        {"start_ms": 650,  "end_ms": 950,  "text": "um",        "is_filler": True,  "speaker": "S0"},
        {"start_ms": 1000, "end_ms": 1500, "text": "what",      "is_filler": False, "speaker": "S0"},
        {"start_ms": 1520, "end_ms": 2000, "text": "happened?", "is_filler": False, "speaker": "S0"},
        {"start_ms": 2800, "end_ms": 3300, "text": "Honestly",  "is_filler": False, "speaker": "S1"},
        {"start_ms": 3340, "end_ms": 3800, "text": "I",         "is_filler": False, "speaker": "S1"},
        {"start_ms": 3820, "end_ms": 4600, "text": "don't",     "is_filler": False, "speaker": "S1"},
        {"start_ms": 4620, "end_ms": 5200, "text": "know.",     "is_filler": False, "speaker": "S1"},
    ]
    n = duration_ms // hop
    rms = [-55.0] * n
    for w in words:
        for i in range(w["start_ms"] // hop, w["end_ms"] // hop):
            if 0 <= i < n:
                rms[i] = -18.0
    dgrid = cc.compute_dialogue_cut_grid(words, rms_db=rms, prosody_hop_ms=hop, duration_ms=duration_ms)
    dialogue = {
        "hop_ms": dgrid.hop_ms, "cut_cost": dgrid.cost_payload(),
        "cut_points": dgrid.points_payload(), "rms_db": rms,
        "prosody_hop_ms": hop, "words": words,
    }

    # BEAT: real derivation over synthetic 120bpm onsets (every 500ms).
    onsets = list(range(300, duration_ms, 500))
    beat = bc.compute_beat_grid(is_musical=True, bpm=120.0, onsets_ms=onsets, duration_ms=duration_ms)

    # ACTION / CAMERA: fake (real ones need an optical-flow pass over pixels).
    impacts = [1200, 2900, 4100, 5400]
    action_cost = hit_cost_curve(impacts, duration_ms, hop, 120)
    # Camera: two settled regions and a whip-pan in the middle (high cost ~3-4s).
    camera_cost = []
    for i in range(n):
        t = i * hop
        camera_cost.append(round(0.9 if 3000 <= t <= 4000 else max(0.05, 0.3 - 0.00004 * abs(t - 1500)), 3))

    channels = [
        {"label": "beat", "color": CH_COLORS["beat"], "hop_ms": beat.hop_ms,
         "cost": beat.cost, "points": beat.points},
        {"label": "action", "color": CH_COLORS["action"], "hop_ms": hop,
         "cost": action_cost, "points": [{"ts_ms": t, "kind": "action_impact", "score": 1.0} for t in impacts]},
        {"label": "camera/distortion", "color": CH_COLORS["camera"], "hop_ms": hop,
         "cost": camera_cost, "points": []},
    ]
    channels.append(_fused_channel(
        duration_ms, dialogue,
        {"beat_cut_cost": beat.cost, "beat_cut_points": beat.points, "beat_cut_hop_ms": beat.hop_ms},
        {"camera_cut_cost": camera_cost, "action_cut_cost": action_cost,
         "action_points": [{"ts_ms": t} for t in impacts], "hop_ms": hop},
    ))
    return {"name": "DEMO — synthetic clip (dialogue+beat real, motion faked)",
            "duration_ms": duration_ms, "dialogue": dialogue, "channels": channels}


def _list_files() -> None:
    import psycopg
    from psycopg.rows import dict_row
    from app.config import get_settings

    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            select f.id, f.name,
                   coalesce(jsonb_array_length(af.dialogue_cut_points),0) as dlg,
                   coalesce(jsonb_array_length(af.beat_cut_points),0) as beat,
                   coalesce(jsonb_array_length(md.action_points),0) as act
              from files f
              left join audio_features af on af.file_id = f.id
              left join motion_dynamics md on md.file_id = f.id
             where coalesce(jsonb_array_length(af.dialogue_cut_cost),0) > 0
                or coalesce(jsonb_array_length(af.beat_cut_cost),0) > 0
                or coalesce(jsonb_array_length(md.action_cut_cost),0) > 0
             order by f.created_at desc
             limit 50
            """
        ).fetchall()
    if not rows:
        print("No files with a cut grid yet (apply migrations 010/011 + re-index).")
        return
    for r in rows:
        print(f"  {r['id']}  dlg={r['dlg']:>3} beat={r['beat']:>3} act={r['act']:>3}  {r['name']}")


HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cut grid</title>
<style>
  body { margin:0; background:#0e0f13; color:#e6e6e6; font:13px/1.4 ui-sans-serif,system-ui,sans-serif; }
  header { padding:12px 16px; border-bottom:1px solid #23252b; }
  h1 { font-size:15px; margin:0 0 4px; font-weight:600; }
  .meta { color:#8b8f98; font-size:12px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; color:#b9bdc6; }
  .sw { width:12px; height:12px; border-radius:2px; display:inline-block; }
  .wrap { overflow-x:auto; padding:12px 0; }
  canvas { display:block; }
  .ctl { padding:8px 16px; display:flex; gap:12px; align-items:center; border-bottom:1px solid #23252b; }
  input[type=range]{ width:220px; }
</style></head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>
  <div class="legend">
    <span><i class="sw" style="background:#3ddc84"></i>dialogue safe-to-cut</span>
    <span><i class="sw" style="background:#ffb454"></i>beat</span>
    <span><i class="sw" style="background:#ff6e9c"></i>action</span>
    <span><i class="sw" style="background:#56b6ff"></i>camera/distortion</span>
    <span><i class="sw" style="background:#9d7bff"></i>FUSED seam field</span>
    <span><i class="sw" style="background:#5b6470"></i>RMS energy</span>
    <span><i class="sw" style="background:#e0533d"></i>filler word</span>
  </div>
</header>
<div class="ctl">
  <label>zoom <input id="zoom" type="range" min="40" max="400" value="120"></label>
  <span class="meta" id="zoomlabel"></span>
</div>
<div class="wrap"><canvas id="c"></canvas></div>
<script>
const DATA = __DATA__;
const SPEAKER_COLORS = ["#4f7cff","#1aa179","#d98a2b","#b5539e","#7a8a9a","#c0563f"];
function speakerColor(s){ if(s==null) return "#5b6470"; const idx=Math.abs(hash(String(s)))%SPEAKER_COLORS.length; return SPEAKER_COLORS[idx]; }
function hash(s){let h=0;for(let i=0;i<s.length;i++){h=(h*31+s.charCodeAt(i))|0;}return h;}
function kindColor(k){return {speaker_change:"#c792ea",sentence_end:"#82aaff",filler_edge:"#e0533d",pause:"#8b8f98",word_gap:"#6b7280"}[k]||"#8b8f98";}

const cv=document.getElementById("c"), ctx=cv.getContext("2d");
const dlg=DATA.dialogue, chans=DATA.channels||[];
document.getElementById("title").textContent=DATA.name;
document.getElementById("meta").textContent=
  `${(DATA.duration_ms/1000).toFixed(1)}s · ${dlg?dlg.words.length+" words · ":""}${chans.length} extra channels`;

const AXIS=22, PAD=16;
const LANE_SAFE=140, LANE_RMS=60, LANE_WORDS=30, LANE_CH=64, GAP=8;
let H=AXIS;
if(dlg){ H += LANE_SAFE+GAP+LANE_RMS+GAP+LANE_WORDS+GAP; }
H += chans.length*(LANE_CH+GAP) + PAD;

// Draw one "safe to cut" (1 - cost) filled area + outline into [top, top+h].
function drawSafe(cost, hop, top, h, color, fillA, x){
  const bot=top+h;
  ctx.beginPath(); ctx.moveTo(0,bot);
  for(let i=0;i<cost.length;i++){const s=1-cost[i];ctx.lineTo(x(i*hop),bot-s*h);}
  ctx.lineTo(x(cost.length*hop),bot); ctx.closePath();
  ctx.fillStyle=fillA; ctx.fill();
  ctx.strokeStyle=color; ctx.lineWidth=1.2; ctx.beginPath();
  for(let i=0;i<cost.length;i++){const s=1-cost[i];const px=x(i*hop),py=bot-s*h;i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
  ctx.stroke();
}
function hexA(hex,a){const n=parseInt(hex.slice(1),16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}

function draw(pxPerSec){
  const W=Math.max(800, Math.ceil(DATA.duration_ms/1000*pxPerSec));
  const dpr=window.devicePixelRatio||1;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.width=W+"px"; cv.style.height=H+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
  const x=ms=>ms/DATA.duration_ms*W;

  ctx.fillStyle="#6b7280"; ctx.font="10px sans-serif"; ctx.strokeStyle="#1c1e24";
  for(let s=0;s<=DATA.duration_ms/1000;s++){const px=x(s*1000);ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,H);ctx.stroke();ctx.fillText(s+"s",px+2,AXIS-6);}

  let y=AXIS;
  if(dlg){
    drawSafe(dlg.cut_cost, dlg.hop_ms, y, LANE_SAFE, "#3ddc84", "rgba(61,220,132,0.28)", x);
    ctx.fillStyle="#3ddc84"; ctx.fillText("dialogue safe to cut ↑",4,y+12);
    const safeBot=y+LANE_SAFE; y=safeBot+GAP;

    const rms=dlg.rms_db||[], rhop=dlg.prosody_hop_ms||dlg.hop_ms;
    if(rms.length){
      let lo=Math.min(...rms), hi=Math.max(...rms); if(hi-lo<1)hi=lo+1;
      const rBot=y+LANE_RMS;
      ctx.strokeStyle="#5b6470"; ctx.beginPath();
      for(let i=0;i<rms.length;i++){const nn=(rms[i]-lo)/(hi-lo);const px=x(i*rhop),py=rBot-nn*LANE_RMS;i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
      ctx.stroke(); ctx.fillStyle="#5b6470"; ctx.fillText("RMS energy",4,y+12);
    }
    y+=LANE_RMS+GAP;

    const wTop=y, wH=LANE_WORDS;
    for(const w of dlg.words){
      const px=x(w.start_ms), pw=Math.max(2,x(w.end_ms)-px);
      ctx.fillStyle=speakerColor(w.speaker); ctx.globalAlpha=w.is_filler?0.35:0.8;
      ctx.fillRect(px,wTop,pw,wH); ctx.globalAlpha=1;
      if(w.is_filler){ctx.strokeStyle="#e0533d";ctx.lineWidth=1.5;ctx.strokeRect(px+0.5,wTop+0.5,pw-1,wH-1);}
      if(pw>16){ctx.fillStyle="#0e0f13";ctx.font="10px sans-serif";ctx.save();ctx.beginPath();ctx.rect(px,wTop,pw,wH);ctx.clip();ctx.fillText(w.text,px+3,wTop+15);ctx.restore();}
    }
    for(const p of dlg.cut_points){
      const px=x(p.ts_ms); ctx.strokeStyle=kindColor(p.kind); ctx.lineWidth=1.2;
      ctx.beginPath(); ctx.moveTo(px,AXIS); ctx.lineTo(px,wTop+wH); ctx.stroke();
      ctx.fillStyle=kindColor(p.kind); const r=2+3*(p.score||0);
      ctx.beginPath(); ctx.arc(px,AXIS+2,r,0,7); ctx.fill();
    }
    y+=LANE_WORDS+GAP;
  }

  for(const ch of chans){
    drawSafe(ch.cost||[], ch.hop_ms||100, y, LANE_CH, ch.color, hexA(ch.color,0.22), x);
    ctx.fillStyle=ch.color; ctx.fillText(ch.label+" safe to cut ↑",4,y+12);
    for(const p of (ch.points||[])){
      const px=x(p.ts_ms); ctx.fillStyle=ch.color;
      ctx.beginPath(); ctx.arc(px,y+LANE_CH-4,3,0,7); ctx.fill();
    }
    y+=LANE_CH+GAP;
  }
}
const zoom=document.getElementById("zoom"), zl=document.getElementById("zoomlabel");
function render(){zl.textContent=zoom.value+" px/s";draw(+zoom.value);}
zoom.addEventListener("input",render); render();
</script>
</body></html>"""


def _render(data: dict, out_path: str, open_browser: bool = True) -> None:
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"Wrote {out_path}")
    if open_browser:
        try:
            webbrowser.open("file://" + os.path.abspath(out_path))
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize the multi-channel cut grid.")
    ap.add_argument("--file-id", help="render the grid for this file")
    ap.add_argument("--list", action="store_true", help="list files that have a grid")
    ap.add_argument("--demo", action="store_true", help="render synthetic data (no DB)")
    ap.add_argument("--out", help="output html path")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

    if args.list:
        _list_files()
        return
    if args.demo:
        data = _demo_data()
        out = args.out or os.path.join(out_dir, "cut_grid_demo.html")
    elif args.file_id:
        data = _load_from_db(args.file_id)
        out = args.out or os.path.join(out_dir, f"cut_grid_{args.file_id}.html")
    else:
        ap.error("pass --file-id <uuid>, --list, or --demo")
        return

    _render(data, out, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
