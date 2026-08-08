"""
Cut-signal visualization (DEBUG / local exercise tool).

A self-contained page + JSON endpoints to *see* the L1 signals that drive cut
segmentation, overlaid on the playable proxy, with live knobs for the weighted
"Cut Score" model (content x quality) and an energy threshold. Lets us tune by
looking at real curves instead of guessing.

Not part of the product surface. Gated by env var CUTVIZ_DEBUG (default on);
set CUTVIZ_DEBUG=0 to disable (e.g. in production). No auth -- intended for
local use against the local user's own data.

Endpoints:
  GET /api/debug/cutviz                 -> the HTML page
  GET /api/debug/cutviz/projects        -> projects + their video files
  GET /api/debug/cutviz/data/{file_id}  -> all signal tracks + proxy URL (recomputes motion on demand)
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/debug/cutviz", tags=["debug"])

_LOCAL_USER = "00000000-0000-0000-0000-000000000001"


def _enabled() -> bool:
    return os.getenv("CUTVIZ_DEBUG", "1") not in ("0", "false", "no", "")


def _pg():
    from app.services.l3.cuts_read import _pg_conn
    return _pg_conn()


@router.get("/projects")
def list_projects():
    if not _enabled():
        raise HTTPException(404, "disabled")
    with _pg() as c:
        projs = c.execute(
            """
            select p.id::text as id,
                   coalesce((select fo.name from files f join folders fo on fo.id = f.folder_id
                              where f.id = p.source_file_ids[1]),
                            nullif(p.name, ''),
                            'project') as name,
                   cardinality(p.source_file_ids) as n
              from projects p
             where p.user_id = %s
             order by p.created_at desc nulls last
            """,
            (_LOCAL_USER,),
        ).fetchall()
        out = []
        for p in projs:
            files = c.execute(
                """
                select f.id::text as id, f.filename as filename, f.file_type as file_type
                  from files f
                 where f.id = any(
                        select unnest(source_file_ids) from projects where id = %s
                       )
                   and f.file_type = 'video'
                 order by f.filename
                """,
                (p["id"],),
            ).fetchall()
            if files:
                out.append({"id": p["id"], "name": p["name"], "n": p["n"], "files": files})
    return {"projects": out}


def _vcut_run_for_file(file_id: str):
    """(ingest_run_id, seam_cache, loose_plan) for the latest vcut run
    covering this file's project (seam_cut_pipeline.plan.md section 4/13.1),
    or None if there isn't one -- e.g. this environment's cuts_pipeline is
    still "v3", or this file's project was never vcut-ingested."""
    with _pg() as c:
        row = c.execute(
            """
            select ir.id::text as id, ir.seam_cache, ir.loose_plan
              from ingest_runs ir
              join projects p on p.id = ir.project_id
             where %s = any(p.source_file_ids)
               and ir.seam_cache is not null and ir.loose_plan is not null
             order by ir.created_at desc limit 1
            """,
            (file_id,),
        ).fetchone()
    if not row:
        return None
    seam_cache = row["seam_cache"] or {}
    loose_plan_dict = row["loose_plan"] or {}
    if file_id not in seam_cache or file_id not in loose_plan_dict:
        return None
    return row["id"], seam_cache, loose_plan_dict


def _vcut_resolve_for_file(file_id: str, energy: float):
    """(run_id, FilePlan, List[ResolvedCut]) for this ONE file at
    ``energy`` -- reuses the real app.services.vcut.resolve.resolve_cuts
    (pure, no I/O) so this debug overlay can never drift from the actual
    algorithm. None if this file has no vcut run to resolve from."""
    found = _vcut_run_for_file(file_id)
    if not found:
        return None
    run_id, seam_cache, loose_plan_dict = found
    from app.services.vcut.resolve import FilePlan, MomentPlan, resolve_cuts

    file_plan = FilePlan.from_dict(file_id, loose_plan_dict[file_id])
    plan = MomentPlan(files=[file_plan])
    seam_for_file = {file_id: seam_cache[file_id]}
    resolved = resolve_cuts(plan, seam_for_file, max(0.0, min(1.0, energy)))
    return run_id, file_plan, resolved


def _resolved_to_json(resolved):
    return [{"in_ms": r.in_ms, "out_ms": r.out_ms, "peak_ms": r.peak_ms, "tag": r.tag} for r in resolved]


@router.get("/vcut_resolve/{file_id}")
def vcut_resolve(file_id: str, energy: float = 0.5):
    """Cheap re-derive for the energy slider (section 9's actual mechanism,
    exercised here read-only/non-mutating for the debug overlay): no model
    call, no R2 download, no motion recompute -- pure resolve_cuts off the
    persisted seam_cache/loose_plan."""
    if not _enabled():
        raise HTTPException(404, "disabled")
    found = _vcut_resolve_for_file(file_id, energy)
    if not found:
        raise HTTPException(404, "no vcut run for this file")
    _run_id, _clip, resolved = found
    return JSONResponse({"resolved": _resolved_to_json(resolved)})


@router.get("/data/{file_id}")
def signal_data(file_id: str):
    if not _enabled():
        raise HTTPException(404, "disabled")
    from app.services.l1 import motion_dynamics as motion_mod
    from app.services.l1.snapshot import build_l1_snapshot
    from app.services.processing import _download_from_r2, _probe_video
    from app.services.r2 import generate_presigned_get

    with _pg() as c:
        row = c.execute(
            "select filename, r2_proxy_key, r2_key from files where id = %s and user_id = %s",
            (file_id, _LOCAL_USER),
        ).fetchone()
    if not row:
        raise HTTPException(404, "file not found")
    proxy_key = row["r2_proxy_key"] or row["r2_key"]
    if not proxy_key:
        raise HTTPException(404, "no media for file")

    # Recompute motion on the proxy so the NEW signals (frame_diff, raw
    # magnitudes) are present even before the L1 schema/RunPod update.
    motion = {}
    duration_ms = 0
    cpd_boundaries_ms = []
    seam_block = {}
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "proxy.mp4")
        try:
            _download_from_r2(proxy_key, local)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"proxy download failed: {e}")
        try:
            info = _probe_video(local)
            duration_ms = int(float(info.get("format", {}).get("duration", 0)) * 1000)
        except Exception:
            duration_ms = 0
        try:
            md = motion_mod.compute_motion_dynamics(local, duration_ms=duration_ms or 60000)
            motion = md.to_dict()
        except Exception as e:  # noqa: BLE001
            logger.exception("motion recompute failed")
            raise HTTPException(500, f"motion compute failed: {e}")

        # cpd_boundary_segmenter.plan.md Phase G.2 (optional overlay): the
        # trained CPD model's own predicted boundaries on this SAME proxy,
        # for side-by-side comparison against the current segmenter's cuts
        # and the raw signal tracks -- lets this debug tool double as the
        # in-domain labeling/validation UI the plan's "Honest risks"
        # section calls for. Best-effort: no trained model yet (or any
        # inference failure) simply omits the overlay, never breaks the
        # rest of the page. Must run INSIDE this block -- `local` is
        # deleted with the temp dir once it exits.
        try:
            cpd_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "scripts", "cpd")
            if cpd_dir not in sys.path:
                sys.path.insert(0, cpd_dir)
            from common import MODEL_PATH_DEFAULT  # noqa: PLC0415
            if os.path.exists(MODEL_PATH_DEFAULT):
                from infer_cpd import predict as cpd_predict  # noqa: PLC0415
                cpd_boundaries_ms = cpd_predict(local, duration_ms=duration_ms)
        except Exception:
            logger.exception("cpd overlay inference failed (continuing without it)")

        # seam_function.plan.md §7: the seam-quality curve S(t) overlay --
        # a quality field only, no cut decision (see the plan's non-goals).
        # Reuses THIS SAME downloaded proxy (proxy_path=local) so it never
        # triggers a second R2 download; must run INSIDE this block for the
        # same reason as the CPD overlay above. Best-effort: any failure
        # (e.g. no audio track, decode issue) omits the block entirely
        # rather than breaking the rest of the page.
        try:
            from app.services.seam import build_seam_signals, compute_seam_curve
            seam_signals = build_seam_signals(file_id, proxy_path=local)
            seam_curve = compute_seam_curve(seam_signals)
            seam_block = {
                "hop_ms": seam_curve.hop_ms,
                "S": seam_curve.S,
                "g_sharp": seam_curve.g_sharp,
                "g_gest": seam_curve.g_gest,
                "still": seam_curve.still,
                "audio": seam_curve.audio,
                "w_aud": seam_curve.w_aud,
                "beats_ms": seam_signals.beats_ms,
                "onsets_ms": seam_signals.onsets_ms,
                "is_musical": seam_signals.is_musical,
                "meta": seam_curve.meta,
            }
        except Exception:
            logger.exception("seam curve computation failed (continuing without it)")
            seam_block = {}

    if not duration_ms and motion.get("hop_ms"):
        duration_ms = motion["hop_ms"] * len(motion.get("action_energy") or [])

    snap = build_l1_snapshot(file_id)
    af = snap.get("audio_features") or {}
    sc = snap.get("scene_cuts") or {}

    # Best-effort overlay of what the current segmenter produced for this file.
    cuts = []
    try:
        with _pg() as c:
            crows = c.execute(
                """
                select src_in_ms, src_out_ms from cut_records
                 where file_id = %s order by src_in_ms limit 200
                """,
                (file_id,),
            ).fetchall()
        cuts = [{"in_ms": r["src_in_ms"], "out_ms": r["src_out_ms"]} for r in crows]
    except Exception:
        cuts = []

    # vcut_moment_energy.plan.md section 13.1 (seam_cut_pipeline.plan.md's
    # own original overlay, updated for the flag-based model): if this
    # file's project has a vcut run, show its moment flags + the resolved
    # cuts at the default energy (the frontend slider then calls
    # /vcut_resolve for a live re-derive on drag). No DB write anywhere in
    # this path -- resolve_cuts is pure.
    vcut_block = {}
    try:
        from app.services.vcut.params import DEFAULT_ENERGY
        found = _vcut_resolve_for_file(file_id, DEFAULT_ENERGY)
        if found:
            run_id, file_plan, resolved = found
            vcut_block = {
                "run_id": run_id,
                "energy_default": DEFAULT_ENERGY,
                "flags": [{"t_ms": f.t_ms, "shape": f.shape, "summary": f.summary} for f in file_plan.flags],
                "resolved": _resolved_to_json(resolved),
            }
    except Exception:
        logger.exception("vcut overlay computation failed (continuing without it)")
        vcut_block = {}

    proxy_url = generate_presigned_get(proxy_key, expires_in=7200)

    def pts(lst):
        return [p.get("ts_ms") for p in (lst or []) if isinstance(p, dict) and "ts_ms" in p]

    def raw_pts(lst):
        return [p for p in (lst or []) if isinstance(p, (int, float))]

    return JSONResponse({
        "file_id": file_id,
        "filename": row["filename"],
        "duration_ms": duration_ms,
        "proxy_url": proxy_url,
        "motion": {
            "hop_ms": motion.get("hop_ms") or 100,
            "action_energy": motion.get("action_energy") or [],
            "action_energy_raw": motion.get("action_energy_raw") or [],
            "camera_motion": motion.get("camera_motion") or [],
            "camera_motion_raw": motion.get("camera_motion_raw") or [],
            "frame_diff": motion.get("frame_diff") or [],
            "frame_diff_raw": motion.get("frame_diff_raw") or [],
            "camera_coherence": motion.get("camera_coherence") or [],
            "camera_stability": motion.get("camera_stability") or [],
            "blur": motion.get("blur") or [],
            "action_points": pts(motion.get("action_points")),
            "transition_points": [
                {"ts_ms": p.get("ts_ms"), "kind": p.get("kind")}
                for p in (motion.get("transition_points") or []) if isinstance(p, dict)
            ],
        },
        "audio": {
            "hop_ms": af.get("prosody_hop_ms") or 0,
            "rms_db": af.get("rms_db") or [],
        },
        "scene": {
            "hop_ms": sc.get("hop_ms") or 0,
            "shot_points": raw_pts(sc.get("shot_points")),
            "composition_points": raw_pts(sc.get("composition_points")),
        },
        "cuts": cuts,
        "cpd_boundaries_ms": cpd_boundaries_ms,
        "seam": seam_block,
        "vcut": vcut_block,
    })


@router.get("")
def page():
    if not _enabled():
        raise HTTPException(404, "disabled")
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cut Signal Viz</title>
<style>
  :root{--bg:#0d0d0f;--panel:#161619;--line:#26262b;--fg:#e8e8ea;--mut:#8a8a92;--acc:#ff7a1a;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  select,button{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 9px;font:inherit}
  button{cursor:pointer}
  button.acc{border-color:var(--acc);color:var(--acc)}
  main{display:grid;grid-template-columns:minmax(360px,1fr) 320px;gap:12px;padding:12px}
  video{width:100%;max-height:34vh;object-fit:contain;background:#000;border-radius:8px}
  canvas{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:8px;display:block;margin-top:10px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
  .knob{display:flex;align-items:center;gap:8px;margin:7px 0}
  .knob label{width:120px;color:var(--mut)}
  .knob input[type=range]{flex:1}
  .knob .val{width:38px;text-align:right;color:var(--fg)}
  .stat{display:flex;justify-content:space-between;margin:4px 0;color:var(--mut)}
  .stat b{color:var(--fg)}
  h3{margin:2px 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
  .leg{display:flex;flex-wrap:wrap;gap:8px 12px;margin-top:8px;font-size:11px;color:var(--mut)}
  .leg i{display:inline-block;width:10px;height:3px;vertical-align:middle;margin-right:4px}
  small{color:var(--mut)}
</style></head>
<body>
<header>
  <b style="color:var(--acc)">Cut Signal Viz</b>
  <select id="proj"></select>
  <select id="file"></select>
  <button id="rand">random clip</button>
  <button id="load" class="acc">load</button>
  <span id="status" style="color:var(--mut)"></span>
</header>
<main>
  <div>
    <video id="vid" controls></video>
    <canvas id="cv" height="460"></canvas>
    <div class="leg" id="leg"></div>
    <small>Shaded orange = cuts kept at current energy. Grey bars under axis = what the current segmenter produced. Yellow bars = CPD model's predicted boundaries (cpd_boundary_segmenter.plan.md), when a trained model exists. Bold white curve = the seam-quality S(t) (seam_function.plan.md) -- a quality field only, never shaded/thresholded, since it makes no cut decision. Vertical ticks: green=action impact, blue=shot cut, purple=composition change, red=transition, light-purple=beat/onset feeding S(t)'s audio term. Diamond markers = vcut's raw moment FLAGS (vcut_moment_energy.plan.md), colored by shape; green bars under axis = vcut's RESOLVED cuts at the energy slider's value, with a marker at each cut's peak (orange=build, blue=settle, purple=both) -- drag the vcut energy knob to watch cuts fuse (energy 0, long+loose) or fall apart (energy up, tighter) live.</small>
  </div>
  <div>
    <div class="panel">
      <h3>Cut Score model</h3>
      <div class="knob"><label>action source</label>
        <select id="actsrc" style="flex:1">
          <option value="action_energy">normalized (current)</option>
          <option value="action_energy_raw">RAW residual</option>
          <option value="frame_diff">frame-diff only</option>
        </select></div>
      <div class="knob"><label>blend</label>
        <select id="blend" style="flex:1">
          <option value="product">content x quality</option>
          <option value="min">min(content,quality)</option>
          <option value="avg">average</option>
          <option value="content">content only</option>
        </select></div>
      <h3 style="margin-top:12px">Content weights</h3>
      <div class="knob"><label>action</label><input type="range" id="w_action" min="0" max="1" step="0.05" value="0.5"><span class="val" id="w_action_v"></span></div>
      <div class="knob"><label>frame-diff</label><input type="range" id="w_fd" min="0" max="1" step="0.05" value="0.5"><span class="val" id="w_fd_v"></span></div>
      <div class="knob"><label>audio rms</label><input type="range" id="w_rms" min="0" max="1" step="0.05" value="0.3"><span class="val" id="w_rms_v"></span></div>
      <h3 style="margin-top:12px">Quality weights</h3>
      <div class="knob"><label>sharpness</label><input type="range" id="w_sharp" min="0" max="1" step="0.05" value="0.5"><span class="val" id="w_sharp_v"></span></div>
      <div class="knob"><label>stability</label><input type="range" id="w_stab" min="0" max="1" step="0.05" value="0.5"><span class="val" id="w_stab_v"></span></div>
      <div class="knob"><label>coherence</label><input type="range" id="w_coh" min="0" max="1" step="0.05" value="0.2"><span class="val" id="w_coh_v"></span></div>
      <h3 style="margin-top:12px">Cutting</h3>
      <div class="knob"><label>energy (thresh)</label><input type="range" id="thr" min="0" max="1" step="0.02" value="0.3"><span class="val" id="thr_v"></span></div>
      <div class="knob"><label>smoothing</label><input type="range" id="smooth" min="1" max="21" step="2" value="5"><span class="val" id="smooth_v"></span></div>
      <div class="knob"><label>min cut (ms)</label><input type="range" id="mindur" min="0" max="3000" step="100" value="700"><span class="val" id="mindur_v"></span></div>
    </div>
    <div class="panel" style="margin-top:12px">
      <h3>Result</h3>
      <div class="stat"><span>file</span><b id="s_file">-</b></div>
      <div class="stat"><span>duration</span><b id="s_dur">-</b></div>
      <div class="stat"><span># cuts</span><b id="s_ncuts">-</b></div>
      <div class="stat"><span>kept</span><b id="s_kept">-</b></div>
      <div class="stat"><span>coverage</span><b id="s_cov">-</b></div>
    </div>
    <div class="panel" style="margin-top:12px" id="vcutpanel">
      <h3>vcut (vcut_moment_energy.plan.md)</h3>
      <div id="vcutnone" style="color:var(--mut)">no vcut run for this file's project</div>
      <div id="vcutknob" style="display:none">
        <div class="knob"><label>energy</label><input type="range" id="vcut_energy" min="0" max="1" step="0.02" value="0"><span class="val" id="vcut_energy_v"></span></div>
        <div class="stat"><span>resolved cuts</span><b id="s_vcut_n">-</b></div>
        <div class="stat"><span>moment flags</span><b id="s_vcut_flags">-</b></div>
      </div>
    </div>
  </div>
</main>
<script>
const $=id=>document.getElementById(id);
let DATA=null, playhead=0;
const API='/api/debug/cutviz';

const TRACKS=[
  {key:'action_energy',src:'motion',color:'#4fd1c5',label:'action (norm)'},
  {key:'action_energy_raw',src:'motion',color:'#2c7a72',label:'action (raw)',rawnorm:true},
  {key:'frame_diff',src:'motion',color:'#f6ad55',label:'frame-diff'},
  {key:'camera_motion',src:'motion',color:'#9f7aea',label:'camera'},
  {key:'blur',src:'motion',color:'#e53e3e',label:'blur'},
  {key:'camera_stability',src:'motion',color:'#63b3ed',label:'stability'},
  {key:'camera_coherence',src:'motion',color:'#4a5568',label:'coherence'},
  {key:'rms',src:'audio',color:'#48bb78',label:'rms'},
];

// seam_function.plan.md §7: S(t)'s own thin component tracks -- separate
// from TRACKS (raw L1 signals) since these are seam.hop_ms-gridded, not
// motion.hop_ms (in practice the same grid today, but kept distinct since
// the seam module owns its own recompute).
const SEAM_TRACKS=[
  {key:'g_sharp',color:'#fc8181',label:'g_sharp (1-blur)'},
  {key:'g_gest',color:'#f6e05e',label:'g_gest (1-.6*act)'},
  {key:'still',color:'#63b3ed',label:'still'},
  {key:'audio',color:'#68d391',label:'audio'},
];

// seam_cut_pipeline.plan.md section 13.1: vcut's resolved-cut overlay.
// Peak tag -> tick color (matches the plan's build/settle/both vocabulary).
const VCUT_TAG_COLORS={build:'#f6ad55',settle:'#63b3ed',both:'#9f7aea'};
let vcutFetchSeq=0;

async function loadProjects(){
  const r=await fetch(API+'/projects'); const j=await r.json();
  $('proj').innerHTML='';
  j.projects.forEach(p=>{
    const o=document.createElement('option'); o.value=p.id;
    o.textContent=(p.name||'(unnamed)')+' — '+p.files.length+' clips'; o._files=p.files;
    $('proj').appendChild(o);
  });
  $('proj').onchange=fillFiles; fillFiles();
}
function fillFiles(){
  const o=$('proj').selectedOptions[0]; if(!o)return;
  $('file').innerHTML='';
  (o._files||[]).forEach(f=>{const e=document.createElement('option');e.value=f.id;e.textContent=f.filename;$('file').appendChild(e);});
}
$('rand').onclick=()=>{const o=$('proj').selectedOptions[0];if(!o||!o._files.length)return;
  const f=o._files[Math.floor(Math.random()*o._files.length)];$('file').value=f.id;load();};
$('load').onclick=load;

async function load(){
  const fid=$('file').value; if(!fid)return;
  $('status').textContent='computing signals (optical-flow pass)…';
  try{
    const r=await fetch(API+'/data/'+fid); if(!r.ok){$('status').textContent='error: '+(await r.text());return;}
    DATA=await r.json();
    $('vid').src=DATA.proxy_url;
    $('s_file').textContent=DATA.filename;
    $('s_dur').textContent=(DATA.duration_ms/1000).toFixed(1)+'s';
    $('status').textContent='';
    // normalize rms into 0..1 on the audio grid
    const rms=DATA.audio.rms_db||[];
    if(rms.length){const mn=Math.min(...rms),mx=Math.max(...rms);
      DATA.audio.rms=rms.map(v=>mx>mn?(v-mn)/(mx-mn):0);}else{DATA.audio.rms=[];}
    setupVcutPanel();
    buildLegend(); redraw();
  }catch(e){$('status').textContent='error: '+e;}
}

function setupVcutPanel(){
  const v=DATA.vcut;
  const has=v&&v.run_id;
  $('vcutnone').style.display=has?'none':'';
  $('vcutknob').style.display=has?'':'none';
  if(!has)return;
  $('vcut_energy').value=v.energy_default;
  $('s_vcut_flags').textContent=(v.flags||[]).length;
  updateVcutStats(v.resolved||[]);
}

function updateVcutStats(resolved){
  $('s_vcut_n').textContent=resolved.length;
}

async function refetchVcutResolve(energy){
  const seq=++vcutFetchSeq;
  try{
    const r=await fetch(`${API}/vcut_resolve/${DATA.file_id}?energy=${energy}`);
    if(!r.ok||seq!==vcutFetchSeq)return;
    const j=await r.json();
    if(seq!==vcutFetchSeq)return;
    DATA.vcut.resolved=j.resolved;
    updateVcutStats(j.resolved);
    redraw();
  }catch(e){/* best-effort debug overlay -- a failed re-derive just leaves the last curve on screen */}
}

$('vcut_energy').addEventListener('input',()=>{
  $('vcut_energy_v').textContent=$('vcut_energy').value;
  if(DATA&&DATA.vcut&&DATA.vcut.run_id)refetchVcutResolve($('vcut_energy').value);
});

function buildLegend(){
  const hasSeam=DATA.seam&&DATA.seam.S&&DATA.seam.S.length;
  const hasVcut=DATA.vcut&&DATA.vcut.run_id;
  $('leg').innerHTML=TRACKS.map(t=>`<span><i style="background:${t.color}"></i>${t.label}</span>`).join('')
    +`<span><i style="background:#ff7a1a"></i>CUT SCORE</span>`
    +((DATA.cpd_boundaries_ms&&DATA.cpd_boundaries_ms.length)?`<span><i style="background:#ecc94b"></i>CPD boundaries</span>`:'')
    +(hasSeam?(`<span><i style="background:#fff;height:2px"></i><b>SEAM S(t)</b></span>`
      +SEAM_TRACKS.map(t=>`<span><i style="background:${t.color}"></i>${t.label}</span>`).join('')
      +`<span><i style="background:#d6bcfa"></i>beat/onset (feeds audio)</span>`):'')
    +(hasVcut?(`<span><i style="background:#9f7aea"></i>vcut moment flag (◆ shape)</span>`
      +`<span><i style="background:#48bb78"></i>vcut resolved cut</span>`
      +Object.entries(VCUT_TAG_COLORS).map(([tag,c])=>`<span><i style="background:${c}"></i>${tag}</span>`).join('')):'');
}

function sampleAt(arr,hop,ms){if(!arr||!arr.length)return 0;let i=Math.round(ms/hop);i=Math.max(0,Math.min(arr.length-1,i));return arr[i]||0;}
function normArr(a){if(!a||!a.length)return a;const mx=Math.max(...a);return mx>0?a.map(v=>v/mx):a;}
function movavg(a,w){if(w<=1)return a;const o=[],h=w>>1;for(let i=0;i<a.length;i++){let s=0,c=0;for(let j=i-h;j<=i+h;j++){if(j>=0&&j<a.length){s+=a[j];c++;}}o.push(s/c);}return o;}

function computeScore(){
  const m=DATA.motion, hop=m.hop_ms, N=(m.action_energy||[]).length;
  const wv={action:+$('w_action').value,fd:+$('w_fd').value,rms:+$('w_rms').value,
            sharp:+$('w_sharp').value,stab:+$('w_stab').value,coh:+$('w_coh').value};
  const actKey=$('actsrc').value;
  let act=m[actKey]||[]; if(actKey==='action_energy_raw')act=normArr(act.slice());
  const fd=m.frame_diff||[], stab=m.camera_stability||[], coh=m.camera_coherence||[], blur=m.blur||[];
  const blend=$('blend').value;
  const score=[];
  for(let i=0;i<N;i++){
    const ms=i*hop;
    const rms=sampleAt(DATA.audio.rms,DATA.audio.hop_ms||hop,ms);
    let cw=wv.action+wv.fd+wv.rms; if(cw<=0)cw=1;
    const content=(wv.action*(act[i]||0)+wv.fd*(fd[i]||0)+wv.rms*rms)/cw;
    let qw=wv.sharp+wv.stab+wv.coh; if(qw<=0)qw=1;
    const quality=(wv.sharp*(1-(blur[i]||0))+wv.stab*(stab[i]||0)+wv.coh*(coh[i]||0))/qw;
    let s;
    if(blend==='product')s=content*quality;
    else if(blend==='min')s=Math.min(content,quality);
    else if(blend==='avg')s=(content+quality)/2;
    else s=content;
    score.push(s);
  }
  return {score:movavg(score,+$('smooth').value),hop,N};
}

function scoreToCuts(score,hop,thr,mindur){
  const cuts=[];let st=-1;
  for(let i=0;i<score.length;i++){
    if(score[i]>=thr && st<0)st=i;
    else if(score[i]<thr && st>=0){cuts.push([st*hop,i*hop]);st=-1;}
  }
  if(st>=0)cuts.push([st*hop,score.length*hop]);
  return cuts.filter(c=>c[1]-c[0]>=mindur);
}

function redraw(){
  if(!DATA)return;
  ['w_action','w_fd','w_rms','w_sharp','w_stab','w_coh','thr','smooth','mindur'].forEach(k=>{$(k+'_v').textContent=$(k).value;});
  const cv=$('cv'),ctx=cv.getContext('2d');
  const W=cv.clientWidth,H=cv.height; cv.width=W;
  ctx.clearRect(0,0,W,H);
  const dur=DATA.duration_ms||1, top=10, botAxis=H-70, plotH=botAxis-top;
  const x=ms=>ms/dur*W, y=v=>botAxis-Math.max(0,Math.min(1,v))*plotH;
  // grid
  ctx.strokeStyle='#26262b';ctx.lineWidth=1;
  for(let s=0;s<=dur/1000;s++){const px=x(s*1000);ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,botAxis);ctx.stroke();
    ctx.fillStyle='#55555c';ctx.fillText(s+'s',px+2,botAxis+12);}
  ctx.strokeStyle='#3a3a42';ctx.beginPath();ctx.moveTo(0,y(0));ctx.lineTo(W,y(0));ctx.stroke();
  // signal tracks
  const m=DATA.motion;
  TRACKS.forEach(t=>{
    let arr,hop;
    if(t.src==='motion'){arr=m[t.key];hop=m.hop_ms;if(t.rawnorm)arr=normArr((arr||[]).slice());}
    else{arr=DATA.audio.rms;hop=DATA.audio.hop_ms||m.hop_ms;}
    if(!arr||!arr.length)return;
    ctx.strokeStyle=t.color;ctx.lineWidth=1.2;ctx.globalAlpha=.85;ctx.beginPath();
    for(let i=0;i<arr.length;i++){const px=x(i*hop),py=y(arr[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
    ctx.stroke();ctx.globalAlpha=1;
  });
  // cut score + threshold + kept regions
  const {score,hop}=computeScore();
  const thr=+$('thr').value, mindur=+$('mindur').value;
  const cuts=scoreToCuts(score,hop,thr,mindur);
  cuts.forEach(c=>{ctx.fillStyle='rgba(255,122,26,0.14)';ctx.fillRect(x(c[0]),top,x(c[1])-x(c[0]),plotH);});
  ctx.strokeStyle='#ff7a1a';ctx.lineWidth=2;ctx.beginPath();
  for(let i=0;i<score.length;i++){const px=x(i*hop),py=y(score[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
  ctx.stroke();
  ctx.strokeStyle='#ff7a1a';ctx.setLineDash([4,4]);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,y(thr));ctx.lineTo(W,y(thr));ctx.stroke();ctx.setLineDash([]);
  // event ticks
  const tick=(list,color,hop2)=>{ctx.strokeStyle=color;ctx.lineWidth=1;(list||[]).forEach(ts=>{const px=x(ts);ctx.beginPath();ctx.moveTo(px,botAxis-plotH*0.12);ctx.lineTo(px,botAxis);ctx.stroke();});};
  tick(m.action_points,'#38a169');
  tick(DATA.scene.shot_points,'#4299e1');
  tick(DATA.scene.composition_points,'#9f7aea');
  tick((m.transition_points||[]).map(p=>p.ts_ms),'#e53e3e');
  // seam_function.plan.md §7: thin S(t) component tracks, the beat/onset
  // ticks that feed audio(t), then the bold S(t) curve itself on top --
  // never shaded/thresholded (the seam layer makes no cut decision).
  const seam=DATA.seam;
  if(seam&&seam.S&&seam.S.length){
    const shop=seam.hop_ms;
    SEAM_TRACKS.forEach(t=>{
      const arr=seam[t.key]; if(!arr||!arr.length)return;
      ctx.strokeStyle=t.color;ctx.lineWidth=1;ctx.globalAlpha=.6;ctx.beginPath();
      for(let i=0;i<arr.length;i++){const px=x(i*shop),py=y(arr[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
      ctx.stroke();ctx.globalAlpha=1;
    });
    tick((seam.beats_ms||[]).concat(seam.onsets_ms||[]),'#d6bcfa');
    ctx.strokeStyle='#ffffff';ctx.lineWidth=2.5;ctx.beginPath();
    for(let i=0;i<seam.S.length;i++){const px=x(i*shop),py=y(seam.S[i]);i?ctx.lineTo(px,py):ctx.moveTo(px,py);}
    ctx.stroke();
  }
  // existing segmenter cuts (grey bars below axis)
  (DATA.cuts||[]).forEach(c=>{ctx.fillStyle='#4a4a52';ctx.fillRect(x(c.in_ms),botAxis+16,x(c.out_ms)-x(c.in_ms),8);});
  // CPD model's predicted boundaries (cpd_boundary_segmenter.plan.md Phase G.2),
  // a third row below the segmenter's -- thin yellow ticks, one per boundary.
  (DATA.cpd_boundaries_ms||[]).forEach(ms=>{ctx.fillStyle='#ecc94b';ctx.fillRect(x(ms)-1,botAxis+28,2,8);});
  // vcut_moment_energy.plan.md section 13.1: Pass 1's raw moment FLAGS
  // (small diamond markers in the main plot area, colored by shape, so you
  // can see every planted flag regardless of energy) + the RESOLVED cuts
  // (a fourth row below CPD) + a peak marker per resolved cut, colored by
  // tag -- never shaded across the whole plot like the CUT SCORE curve,
  // since these ARE the actual decided boundaries, not a continuous field.
  if(DATA.vcut&&DATA.vcut.run_id){
    (DATA.vcut.flags||[]).forEach(f=>{
      const px=x(f.t_ms),py=top+10;
      ctx.fillStyle=VCUT_TAG_COLORS[f.shape]||'#9f7aea';
      ctx.beginPath();ctx.moveTo(px,py-5);ctx.lineTo(px+5,py);ctx.lineTo(px,py+5);ctx.lineTo(px-5,py);ctx.closePath();ctx.fill();
    });
    (DATA.vcut.resolved||[]).forEach(c=>{
      ctx.fillStyle='#48bb78';ctx.fillRect(x(c.in_ms),botAxis+40,Math.max(1,x(c.out_ms)-x(c.in_ms)),8);
      ctx.fillStyle=VCUT_TAG_COLORS[c.tag]||'#9f7aea';
      const px=x(c.peak_ms);ctx.beginPath();ctx.arc(px,botAxis+44,3,0,Math.PI*2);ctx.fill();
    });
  }
  // playhead
  const phx=x(playhead*1000);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(phx,top);ctx.lineTo(phx,botAxis);ctx.stroke();
  // stats
  const kept=cuts.reduce((a,c)=>a+(c[1]-c[0]),0);
  $('s_ncuts').textContent=cuts.length;
  $('s_kept').textContent=(kept/1000).toFixed(1)+'s';
  $('s_cov').textContent=(kept/dur*100).toFixed(0)+'%';
}

['w_action','w_fd','w_rms','w_sharp','w_stab','w_coh','thr','smooth','mindur','actsrc','blend'].forEach(k=>{
  $(k).addEventListener('input',redraw);
});
$('vid').addEventListener('timeupdate',()=>{playhead=$('vid').currentTime;redraw();});
window.addEventListener('resize',redraw);
loadProjects();
</script>
</body></html>
"""
