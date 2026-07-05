#!/usr/bin/env python3
"""
Cuts-v2 partition scrubber (READ-ONLY) -- the Phase B "eyeball metric" harness.

Renders the REAL video with the ACTUAL partition (``l3.partition.build_partition``
+ ``l3.tightness``) drawn as a lane of tag-colored blocks underneath, plus the
scene/shot detection markers (Phase B1) and the same cut-cost/fused lanes
``viz_video_signal.py`` already draws, all on one shared, scrubbable time axis.

This is the plan's "detection robustness" validation step: before building the
API/frontend surface (Phase B4/F1/F2), look at a handful of real clips and
judge whether cuts land where an editor would actually cut, and iterate the
thresholds in ``scene_cuts_params.py`` / ``partition_params.py`` accordingly.

    python scripts/viz_cuts.py --list
    python scripts/viz_cuts.py --file-id <uuid> [--energy 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import viz_cut_grid as viz  # noqa: E402
from viz_video_signal import _playback_url  # noqa: E402

PRIMARY_COLORS = {"said": "#3ddc84", "done": "#ff6e9c", "shown": "#56b6ff"}


def _load_cuts(file_id: str, energy: float) -> dict:
    """The real partition for this file, with tightness applied -- exactly
    what a caller of ``partition.build_partition`` would get."""
    from app.services.l3 import partition as pt
    from app.services.l3 import tightness as tt
    from app.services.l3 import score_span as ss

    cuts = pt.build_partition(file_id)
    if not cuts:
        return {"cuts": [], "scene": None}

    source = ss.load_source(file_id)
    words = list(source.words or []) if source else []
    cuts = tt.apply_tightness_all(cuts, energy, words_by_file={file_id: words})

    clip = pt.load_clip_artifacts(file_id)
    scene = clip.scene if clip else None
    return {
        "cuts": [c.to_dict() for c in sorted(cuts, key=lambda c: c.src_in_ms)],
        "scene": scene,
    }


def _load_from_db(file_id: str, energy: float) -> dict:
    data = viz._load_from_db(file_id)
    data.update(_load_cuts(file_id, energy))
    return data


HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cuts v2 partition scrubber</title>
<style>
  body { margin:0; background:#0e0f13; color:#e6e6e6; font:13px/1.4 ui-sans-serif,system-ui,sans-serif; }
  header { padding:10px 16px; border-bottom:1px solid #23252b; }
  h1 { font-size:15px; margin:0 0 4px; font-weight:600; }
  .meta { color:#8b8f98; font-size:12px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:6px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; color:#b9bdc6; }
  .sw { width:12px; height:12px; border-radius:2px; display:inline-block; }
  #stage { display:flex; justify-content:center; padding:10px; background:#000; }
  video { max-height:40vh; max-width:96vw; background:#000; }
  .ctl { padding:8px 16px; display:flex; gap:16px; align-items:center; border-bottom:1px solid #23252b; }
  input[type=range]{ width:200px; }
  .wrap { overflow-x:auto; padding:8px 0; position:relative; }
  .lanes { position:relative; }
  canvas { display:block; }
  #play { position:absolute; left:0; top:0; pointer-events:none; }
  .hint { color:#6b7280; }
  #tip { position:fixed; display:none; background:#1a1c22; border:1px solid #30333c; padding:6px 8px;
         border-radius:6px; font-size:12px; max-width:340px; pointer-events:none; z-index:10; }
</style></head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>
  <div class="legend">
    <span><i class="sw" style="background:#3ddc84"></i>said</span>
    <span><i class="sw" style="background:#ff6e9c"></i>done</span>
    <span><i class="sw" style="background:#56b6ff"></i>shown</span>
    <span><i class="sw" style="background:#ff3b3b"></i>shot cut</span>
    <span><i class="sw" style="background:#ffb454"></i>composition change</span>
    <span><i class="sw" style="background:#9d7bff"></i>FUSED</span>
    <span><i class="sw" style="background:#f33"></i>playhead</span>
  </div>
</header>
<div id="stage"><video id="vid" controls preload="auto" src="__VIDEO__"></video></div>
<div class="ctl">
  <label>zoom <input id="zoom" type="range" min="40" max="400" value="120"></label>
  <span class="meta" id="zoomlabel"></span>
  <span class="hint">click any lane to seek · space = play/pause · hover a cut for its label</span>
</div>
<div class="wrap"><div class="lanes"><canvas id="c"></canvas><canvas id="play"></canvas></div></div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const SPEAKER_COLORS = ["#4f7cff","#1aa179","#d98a2b","#b5539e","#7a8a9a","#c0563f"];
function speakerColor(s){ if(s==null) return "#5b6470"; const idx=Math.abs(hash(String(s)))%SPEAKER_COLORS.length; return SPEAKER_COLORS[idx]; }
function hash(s){let h=0;for(let i=0;i<s.length;i++){h=(h*31+s.charCodeAt(i))|0;}return h;}
const PRIMARY_COLORS = {"said":"#3ddc84","done":"#ff6e9c","shown":"#56b6ff"};

const cv=document.getElementById("c"), ctx=cv.getContext("2d");
const pl=document.getElementById("play"), pctx=pl.getContext("2d");
const vid=document.getElementById("vid");
const wrap=document.querySelector(".wrap");
const tip=document.getElementById("tip");
const dlg=DATA.dialogue, chans=DATA.channels||[], cuts=DATA.cuts||[], scene=DATA.scene;
document.getElementById("title").textContent=DATA.name;
document.getElementById("meta").textContent=
  `${(DATA.duration_ms/1000).toFixed(1)}s · ${cuts.length} cuts · ${dlg?dlg.words.length+" words · ":""}${chans.length} channels`;

const AXIS=22, PAD=16;
const LANE_CUTS=56, LANE_SCENE=20, LANE_SAFE=100, LANE_RMS=40, LANE_WORDS=24, LANE_CH=52, GAP=8;
let H=AXIS;
H += LANE_CUTS+GAP+LANE_SCENE+GAP;
if(dlg){ H += LANE_SAFE+GAP+LANE_RMS+GAP+LANE_WORDS+GAP; }
H += chans.length*(LANE_CH+GAP) + PAD;

let curW=800, curX=ms=>0;
let cutBoxes=[];  // {x,y,w,h,cut} for hit-testing on hover

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

function drawCuts(top, h, x){
  cutBoxes=[];
  for(const c of cuts){
    const px=x(c.src_in_ms), pw=Math.max(1, x(c.src_out_ms)-px);
    const color=PRIMARY_COLORS[c.primary]||"#8b8f98";
    ctx.fillStyle=hexA(color,0.30); ctx.fillRect(px,top,pw,h);
    ctx.strokeStyle=color; ctx.lineWidth=1.4; ctx.strokeRect(px+0.5,top+0.5,Math.max(0,pw-1),h-1);
    // keep_spans (tightness jump-cut edit-list): shade the excised gaps.
    if(c.keep_spans){
      let prevEnd=c.src_in_ms;
      for(const [ks,ke] of c.keep_spans){
        if(ks>prevEnd){ const gx=x(prevEnd), gw=x(ks)-gx; ctx.fillStyle="rgba(0,0,0,0.55)"; ctx.fillRect(gx,top,gw,h); }
        prevEnd=ke;
      }
    }
    if(pw>28){
      ctx.fillStyle="#0e0f13"; ctx.font="10px sans-serif"; ctx.save();
      ctx.beginPath(); ctx.rect(px,top,pw,h); ctx.clip();
      ctx.fillText(c.tags.join("+")+" "+(c.label||"").slice(0,40), px+3, top+13);
      ctx.restore();
    }
    cutBoxes.push({x:px,y:top,w:pw,h:h,cut:c});
  }
}

function drawScene(top, h, x){
  if(!scene) return;
  ctx.strokeStyle="#3a3d46"; ctx.fillRect(0,top,curW,h);
  for(const p of (scene.shot_points||[])){
    const px=x(p.ts_ms); ctx.strokeStyle="#ff3b3b"; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(px,top); ctx.lineTo(px,top+h); ctx.stroke();
  }
  for(const p of (scene.composition_points||[])){
    const px=x(p.ts_ms); ctx.strokeStyle="#ffb454"; ctx.lineWidth=1.2;
    ctx.beginPath(); ctx.moveTo(px,top+h*0.3); ctx.lineTo(px,top+h); ctx.stroke();
  }
}

function draw(pxPerSec){
  const W=Math.max(800, Math.ceil(DATA.duration_ms/1000*pxPerSec));
  const dpr=window.devicePixelRatio||1;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.width=W+"px"; cv.style.height=H+"px";
  pl.width=W*dpr; pl.height=H*dpr; pl.style.width=W+"px"; pl.style.height=H+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0); pctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  const x=ms=>ms/DATA.duration_ms*W;
  curW=W; curX=x;

  ctx.fillStyle="#6b7280"; ctx.font="10px sans-serif"; ctx.strokeStyle="#1c1e24";
  for(let s=0;s<=DATA.duration_ms/1000;s++){const px=x(s*1000);ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,H);ctx.stroke();ctx.fillText(s+"s",px+2,AXIS-6);}

  let y=AXIS;
  ctx.fillStyle="#8b8f98"; ctx.fillText("PARTITION (cuts-v2)",4,y+10);
  drawCuts(y+2, LANE_CUTS-2, x); y+=LANE_CUTS+GAP;
  ctx.fillStyle="#8b8f98"; ctx.fillText("scene / shot",4,y+10);
  drawScene(y, LANE_SCENE, x); y+=LANE_SCENE+GAP;

  if(dlg){
    drawSafe(dlg.cut_cost, dlg.hop_ms, y, LANE_SAFE, "#3ddc84", "rgba(61,220,132,0.28)", x);
    ctx.fillStyle="#3ddc84"; ctx.fillText("dialogue safe to cut ↑",4,y+12);
    y+=LANE_SAFE+GAP;
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
      if(pw>16){ctx.fillStyle="#0e0f13";ctx.font="10px sans-serif";ctx.save();ctx.beginPath();ctx.rect(px,wTop,pw,wH);ctx.clip();ctx.fillText(w.text,px+3,wTop+15);ctx.restore();}
    }
    y+=LANE_WORDS+GAP;
  }
  for(const ch of chans){
    drawSafe(ch.cost||[], ch.hop_ms||100, y, LANE_CH, ch.color, hexA(ch.color,0.22), x);
    ctx.fillStyle=ch.color; ctx.fillText(ch.label+" safe to cut ↑",4,y+12);
    for(const p of (ch.points||[])){
      const px=x(p.ts_ms); ctx.fillStyle=ch.color;
      const r=2+3*(p.score||0);
      ctx.beginPath(); ctx.arc(px,y+LANE_CH-4,r,0,7); ctx.fill();
    }
    y+=LANE_CH+GAP;
  }
}

function playhead(){
  const W=curW;
  pctx.clearRect(0,0,W,H);
  const t=(vid.currentTime||0)*1000;
  const px=curX(t);
  pctx.strokeStyle="#ff3b3b"; pctx.lineWidth=2;
  pctx.beginPath(); pctx.moveTo(px,0); pctx.lineTo(px,H); pctx.stroke();
  pctx.fillStyle="#ff3b3b"; pctx.beginPath(); pctx.moveTo(px-4,0); pctx.lineTo(px+4,0); pctx.lineTo(px,6); pctx.closePath(); pctx.fill();
  if(!vid.paused){
    if(px < wrap.scrollLeft+50 || px > wrap.scrollLeft+wrap.clientWidth-50)
      wrap.scrollLeft = px - wrap.clientWidth/2;
  }
  requestAnimationFrame(playhead);
}

function seek(e){ const x=e.offsetX; vid.currentTime=(x/curW)*DATA.duration_ms/1000; }
cv.addEventListener("click", seek);
cv.addEventListener("mousemove", e=>{
  const hit=cutBoxes.find(b=>e.offsetX>=b.x&&e.offsetX<=b.x+b.w&&e.offsetY>=b.y&&e.offsetY<=b.y+b.h);
  if(!hit){ tip.style.display="none"; return; }
  const c=hit.cut;
  tip.style.display="block"; tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+14)+"px";
  tip.innerHTML = `<b>${c.tags.join(" + ")}</b> [${(c.src_in_ms/1000).toFixed(2)}s–${(c.src_out_ms/1000).toFixed(2)}s]`
    + (c.speaker?` · ${c.speaker}`:"") + `<br>${(c.label||"").slice(0,200)}`;
});
cv.addEventListener("mouseleave", ()=>{ tip.style.display="none"; });
pl.style.pointerEvents="none";
document.addEventListener("keydown", e=>{ if(e.code==="Space"){ e.preventDefault(); vid.paused?vid.play():vid.pause(); }});

const zoom=document.getElementById("zoom"), zl=document.getElementById("zoomlabel");
function render(){zl.textContent=zoom.value+" px/s";draw(+zoom.value);}
zoom.addEventListener("input",render);
render();
requestAnimationFrame(playhead);
</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Cuts-v2 partition scrubber.")
    ap.add_argument("--file-id", help="render the real partition for this file")
    ap.add_argument("--list", action="store_true", help="list files that have a grid")
    ap.add_argument("--energy", type=float, default=0.5, help="tightness (Phase B3)")
    ap.add_argument("--out", help="output html path")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if args.list:
        viz._list_files()
        return
    if not args.file_id:
        ap.error("pass --file-id <uuid> or --list")
        return

    data = _load_from_db(args.file_id, args.energy)
    if not data.get("cuts"):
        print("Warning: no partition for this file yet (no thoughts/motion/scene "
              "artifacts, or duration not set). Rendering signal lanes only.")
    video_url = _playback_url(args.file_id)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    out = args.out or os.path.join(out_dir, f"cuts_{args.file_id}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__VIDEO__", video_url))
    with open(out, "w") as fh:
        fh.write(html)
    print(f"Wrote {out}")
    if not args.no_open:
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:
            pass


if __name__ == "__main__":
    main()
