"""End-to-end verification of engine looks via the live API on Reel 5.

Robust to concurrent frontend writes (retries 409 stale_base_version) and to a
slow PUT (long timeout). Applies two different engine looks, polls
grade-status to done, GETs the thread, reads resolved.video_layers[*].grade,
confirms the resolved grade DIFFERS between picks and is non-identity, then
restores the thread's original look.
"""
import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
T = "3bfe3db3-0dce-4fc6-bc97-42fb4ec08bad"


def req(method, path, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_thread():
    return req("GET", f"/api/edit/threads/{T}")


def put_look(look, note, retries=6):
    """PUT the look, retrying on 409 stale_base_version (concurrent writes)."""
    for attempt in range(retries):
        th = get_thread()
        doc = th["document"]
        ver = th["document_version"]
        body = {"base_version": ver, "timeline": doc.get("timeline", []),
                "operations": doc.get("operations", []), "look": look}
        try:
            put = req("PUT", f"/api/edit/threads/{T}/document", body)
            print(f"[{note}] PUT look={json.dumps(look)} base_version={ver} -> version {put['version']}")
            return put["version"]
        except urllib.error.HTTPError as e:
            if e.code == 409:
                print(f"[{note}] 409 stale base_version (attempt {attempt+1}); refetching...")
                time.sleep(0.5)
                continue
            raise
    raise RuntimeError(f"[{note}] gave up after {retries} 409 retries")


def poll_done(note):
    for i in range(120):
        st = req("GET", f"/api/edit/threads/{T}/grade-status")
        if i % 6 == 0 or st["state"] in ("done", "error"):
            print(f"    t={i*0.6:.1f}s state={st['state']} progress={st.get('progress')} done={st.get('done')}/{st.get('total')}")
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.6)
    return {"state": "timeout"}


def read_grades(expect_look):
    """GET the thread; confirm the active look is still the one we set (no
    concurrent overwrite), then return per-layer resolved grades."""
    th = get_thread()
    doc = th["document"]
    active = doc.get("look")
    layers = (doc.get("resolved") or {}).get("video_layers") or []
    grades = [lyr.get("grade") or {} for lyr in layers]
    return active, grades


def summarize(grades):
    return [{
        "hash": (g.get("grade_hash") or "")[:12],
        "look_engine": g.get("look_engine"),
        "cdl_sat": round((g.get("cdl") or {}).get("sat", 1.0), 3),
        "creative_lut_ref": g.get("creative_lut_ref"),
    } for g in grades]


def apply_and_read(look, note):
    put_look(look, note)
    poll_done(note)
    active, grades = read_grades(look)
    if active != look:
        # concurrent write changed the look; re-assert once
        print(f"[{note}] active look drifted to {json.dumps(active)}; re-applying...")
        put_look(look, note + "-reassert")
        poll_done(note + "-reassert")
        active, grades = read_grades(look)
    return active, grades


def main():
    orig = get_thread()
    orig_look = orig["document"].get("look")
    print(f"ORIGINAL look: {json.dumps(orig_look)}\n")

    look_a = {"mode": "engine", "look_id": "punchy_vibrant"}
    look_b = {"mode": "engine", "look_id": "moody_cinematic"}

    active_a, grades_a = apply_and_read(look_a, "LOOK A punchy_vibrant")
    print(f"  active look A: {json.dumps(active_a)}")
    print("  resolved grades A:")
    for s in summarize(grades_a):
        print("   ", json.dumps(s))
    print()

    active_b, grades_b = apply_and_read(look_b, "LOOK B moody_cinematic")
    print(f"  active look B: {json.dumps(active_b)}")
    print("  resolved grades B:")
    for s in summarize(grades_b):
        print("   ", json.dumps(s))
    print()

    ga0 = grades_a[0] if grades_a else {}
    gb0 = grades_b[0] if grades_b else {}
    le_a, le_b = ga0.get("look_engine"), gb0.get("look_engine")
    ha, hb = ga0.get("grade_hash"), gb0.get("grade_hash")

    print("==== VERDICT ====")
    print(f"look_engine A (first layer): {json.dumps(le_a)}")
    print(f"look_engine B (first layer): {json.dumps(le_b)}")
    print(f"grade_hash A: {ha}")
    print(f"grade_hash B: {hb}")
    both_set = bool(le_a) and bool(le_b)
    differ = (le_a != le_b) and (ha != hb)
    all_hashes_differ = (
        all(a.get("grade_hash") != b.get("grade_hash") for a, b in zip(grades_a, grades_b))
        if grades_a and grades_b and len(grades_a) == len(grades_b) else None
    )
    print(f"BOTH non-identity engine looks: {both_set}")
    print(f"DIFFER (first-layer look_engine + hash): {differ}")
    print(f"ALL per-shot hashes differ A vs B: {all_hashes_differ}")
    ok = both_set and differ
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")

    reverted = put_look(orig_look, "REVERT")
    poll_done("REVERT")
    active_r, _ = read_grades(orig_look)
    print(f"\nreverted look -> version {reverted}; active look now: {json.dumps(active_r)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
