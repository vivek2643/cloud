# COLOR GRADING — STRATEGY & BUILD PLAN

Companion to `master_strategy.md`. Captures the ideation for the automated
color-grading system: the philosophy, the moat, the locked decisions, the
architecture, and a scannable build checklist.

Status: **planning doc — nothing here is implemented yet.**

---

## 1. Philosophy: "Manual taste, automatic craft"

We do **not** auto-pick the creative look (that's subjective — the user owns it).
We **do** automate everything that makes a look land like a professional grade:
correction, shot-to-shot consistency, subject-aware metering, and the cinematic
arc — all deterministically, guided by context nobody else has.

> Grade the **story**, not the frame.

The user chooses *which* look via three modes. The system makes that look
correct, consistent, and cinematic across the whole edit, and exports it so a
pro can keep editing.

---

## 2. The moat — where we differ and where we're best (honest)

| Capability | Us | CapCut / filters | Resolve (manual) | Colourlab Ai (auto) | Human colorist |
|---|---|---|---|---|---|
| Fully automatic, no manual reference | ✅ | ✅ (dumb) | ❌ | ✅ | ❌ |
| **Understands footage content** (graded? true neutral? same scene?) | ✅ only us | ❌ | ❌ | ❌ pixels | ✅ |
| **Subject-anchored WB/exposure** | ✅ | ❌ | ⚠️ manual | ⚠️ generic face | ✅ |
| **Semantic auto shot-matching** (auto-groups scene) | ✅ only us | ❌ | ⚠️ manual/pixel | ⚠️ pixel, no grouping | ✅ |
| **Story/intent-driven look + arc** | ✅ only us | ❌ | ❌ | ❌ | ✅ if they bother |
| **Asks the user when unsure** | ✅ only us | ❌ | ❌ | ❌ | ✅ |
| **One signal → grade + reframe + subject** | ✅ only us | ❌ | ❌ | ❌ | ❌ |
| Crisp local/spatial grade (power windows) | ⚠️ soft only | ❌ | ✅ best | ⚠️ limited | ✅ best |
| Pro round-trip export (CDL/LUT) | ✅ | ❌ | ✅ | ✅ | ✅ |
| Creative taste **ceiling** | ⚠️ medium | ❌ | (operator) | ⚠️ medium | ✅ highest |
| Fix physically broken lighting (relight) | ❌ | ❌ | ❌ | ❌ | ❌ |

**Where we're genuinely best:** context-aware grading (VLM knows the footage) +
subject-anchored correction + story/intent arc + ask-the-user + one-signal
integration. This *combination* exists nowhere.

**Where we only tie (copyable):** the color math, LUT/reference/CDL export,
basic auto-matching (Colourlab is close).

**Where we lose (brutal):** taste ceiling vs a senior colorist; crisp local work
vs Resolve power windows; we can be *confidently wrong* on ambiguous footage;
we cannot fix physically broken lighting.

**Bottom line:** we don't raise the *ceiling* of what's possible — we raise the
**floor**. Better than the ~90% who don't grade well or lack time; the only
automated tool that grades with *understanding* and *asks* when unsure.

**Honest caveat:** two headline advantages — **auto-scene-grouping** and
**`color_stats`** — are *plans, not capabilities* yet. The moat is only real
once they're built.

---

## 3. Locked decisions

- Audience: **prosumer** (great default + heavy steering + pro export).
- Ambition: **full narrative arc**, but **invisible** (user sets base look + one
  intensity dial; system choreographs per-beat).
- **No auto look-selection.** User picks the look via three modes.
- Three input modes: **default gallery (parametric)**, **reference image**, **LUT drop**.
- Keep the **correct + match** layer underneath the chosen look (deterministic).
- Defaults are **parametric recipes we author** (not scraped LUTs — licensing + consistency).
- **CDL is the internal spine**; look travels as a separable LUT; non-CDL nuance bakes into the LUT.
- Add L1 **`color_stats`** (measured pixels) — the honest foundation.
- Bring **VLM + LLM** back in: VLM = perception, deterministic = execution, LLM = human boundary.
- LLM role: **ask when unsure + parse NL steering + explain + rank looks** (never auto-applies).
- **Drop the broken-footage flag** — but pair with **graceful, never-worse** degradation.
- Subject placement: **ranked ROI boxes + salience + composition cues** (not a single circle); **static for now** (no temporal tracking yet).

---

## 4. The grade object (CDL-native spine)

```
grade {
  cdl:          { slope[3], offset[3], power[3], sat }   // per clip — correct + match + arc; round-trips losslessly
  creative_lut: <optional .cube>                          // the look; non-CDL nuance (skin-protect, curves) baked in
  working_space: "rec709" | "logc" | ...                  // stamped so preview == Resolve
}
```

- CDL carries correction, matching, and the whole arc (arc = a different CDL per clip → round-trips).
- The look (preset / reference / raw LUT) resolves into `creative_lut`.
- Working space must be stamped or preview ≠ downstream tool (the #1 CDL round-trip bug).

---

## 5. The grade stack (fixed order)

```
Measure → Correct (semantic-gated) → Match (group consistency) → Look → Arc → NL trims → export
```

---

## 6. The three input modes → one target look

| Mode | Resolves to | Arc behavior |
|---|---|---|
| **Default gallery (parametric)** | recipe id + dials | full arc — modulate dials per beat |
| **Reference image** | fit to dials/CDL (skin-aware transfer) | full arc — modulate fitted dials |
| **User `.cube` upload** | black-box LUT + param trims | arc via LUT-mix + CDL trims per beat |

Default gallery: ~10–12 **parametric recipes we author**, grounded in the grading
canon (Warm Film, Clean Modern, Cinematic Teal, Moody, Faded/Matte, Golden,
Punch, Vintage, Pastel, Mono, …). Small & excellent > large & mediocre. Can be
seeded via our own reference-transfer on rights-cleared references. VLM ranks
"suggested for this footage" (orders, never auto-applies).

---

## 7. VLM (L2) additions — trimmed to what only a VLM can do

Bar: add a field only if (a) pixels can't do it, (b) VLM is reliable at it,
(c) it doesn't force per-region grading that breaks the global CDL model.

| Field | Why | Guardrail |
|---|---|---|
| `already_graded` | don't double-grade | pixels handle log/flat; VLM only judges "intentional grade present?" |
| `white_reference` | anchor WB to a true neutral (auto-WB's holy grail) | VLM proposes region + object; **deterministic verifies it's actually neutral** |

Everything else is done deterministically (`color_stats`: log/flat, clipping,
cast, signature) or already exists (`persons` skin, `setting`/`look` for
grouping). Dropped as redundant/model-breaking: `color_cast`, sky/foliage
`memory_colors`, `dynamic_range_issues`, a separate `scene_continuity_key`.

---

## 8. Subject placement (shared with reframing)

Model placement as a **ranked list of ROI boxes + salience + composition cues**
(`horizon_y`, facing/gaze, headroom, frame_type) — NOT a single center/circle
(under-expressive; fails on multi-subject, scenic, action).

- **Reframing:** deterministic solver picks the crop for any target aspect by
  maximizing preserved salience under composition rules. When content can't all
  fit → choose **prioritize / fit+pad / layout** (not blind center-crop).
- **Grading synergy (same signal):** subject-anchored WB/exposure (meter off the
  face, not the frame), subject-priority matching, soft depth grade, and
  attention vignette (the circle's sweet spot).
- VLM spatial output is loose → good for metering/soft gradients; **crisp masks
  need a real segmentation model**, not a VLM box.
- Temporal tracking parked; static per-clip for now.

---

## 9. LLM (L3) role — the human boundary

- **Ask-user loop** when genuinely uncertain (few high-leverage questions;
  relights the stubbed `awaiting_user` status).
- **Parse NL steering** ("warmer", "less teal", "fix skin") → dial/CDL edits.
- **Explain the grade** (trust + steerability).
- **Rank/suggest looks** (never auto-applies).

LLM never emits raw color numbers — it chooses intent + dials; deterministic code
owns the geometry (same pattern as `framing.py`).

---

## 10. Pipeline integration

```
Director → Editor → resolve_document → render
                         │
        grade (user choice) stored on Edit Document
                         │
   deterministic resolver: color_stats → correct+match CDL + look + per-beat arc delta
                         │  (LLM only for ask / steer / explain)
   render: per-clip CDL + lut3d in working_space (proxy == export)
                         │
   export: .cdl/.ccc + .cube, referenced in XML/EDL
```

- Grade is **user-set metadata + deterministic resolver** — no heavy new LLM pass.
- Schema: `grade` block per timeline item + sequence-level look; new L1
  `color_stats` table; versioned via append-only `edit_documents`.
- Re-grade triggers on timeline change (regroup/re-arc); decoupled from the cut.

---

## 11. Honest limits (do not over-promise)

**Grading CAN** (we do well): exposure, white balance, contrast, shot-to-shot
brightness/color matching, soft directional shaping (sun example), attention
vignette.

**Grading CANNOT** (physics, not effort): recover blown highlights, rescue
crushed noisy shadows, truly relight (move/remove shadows, fix mixed-color-temp),
add light never captured. That's **relighting** — a separate generative capability,
out of scope.

Since we drop the broken-footage flag: **never make it worse** (don't lift
crushed→noise, don't push clipped further). Do best-effort silently.

---

## 12. Build checklist

Engine — D: deterministic · V: VLM/L2 · L: LLM/L3

### 12.1 Foundation — measurement
| Task | Engine | Notes |
|---|---|---|
| L1 `color_stats` stage | D | histograms, black/white points, luma, Lab chroma, WB estimate, clipping, skin sample. **Everything depends on this.** |
| `color_stats` DB table | D | per-clip signature |

### 12.2 Perception — VLM (L2)
| Task | Engine | Notes |
|---|---|---|
| `already_graded` field | V | "intentional stylized grade present?" → don't double-grade |
| `white_reference` field | V | neutral object + region; deterministic verifies |
| ROI / salience + composition cues | V | ranked boxes + salience + horizon_y, facing, headroom, frame_type |
| Reuse existing | — | `setting.location`, `look.*`, `persons.skin_tone/frame_region/best_face_ms` |

### 12.3 Correct layer (global, CDL)
| Task | Engine | Notes |
|---|---|---|
| Auto exposure normalize | D | within captured range |
| Auto white balance | D | subject/face-anchored + verified `white_reference` |
| Contrast / black-white point | D | |
| Log/flat detect + input transform | D | don't grade in wrong space |
| Semantic gating | D+V | keep golden hour, protect skin, skip already-graded |
| Graceful-degrade guardrail | D | never-worse (crushed→noise, clipped) |

### 12.4 Match layer (consistency)
| Task | Engine | Notes |
|---|---|---|
| Scene grouping | D | join `setting`/`look` + `color_stats`. **Not built — the "auto-group" advantage** |
| Anchor selection per group | D | best-exposed/hero; reuse `hero_cuts`/take_quality |
| Match within group → anchor | D | conservative |
| Skin-priority weighting | D+V | consistent faces across cuts |
| Master look across groups | D | unify identity, keep day/night differences |

### 12.5 Look layer (3 modes)
| Task | Engine | Notes |
|---|---|---|
| Parametric recipe engine | D | recipe → CDL + LUT |
| Generate ~10–12 default recipes | D | author from canon + taste-validate |
| Look browser UI | D+V | live thumbnails on smart frame; VLM ranking |
| Reference-image → parametric transfer | D | skin-aware; match-strength dial |
| LUT upload → apply | D | wrap with trims; input color space |

### 12.6 Arc layer (invisible)
| Task | Engine | Notes |
|---|---|---|
| Per-beat dial deltas | D | fixed table keyed to `beats[].purpose` |
| Global intensity dial | D | user's one control; scales arc amplitude |

### 12.7 Subject placement / reframing
| Task | Engine | Notes |
|---|---|---|
| ROI representation | V | ranked boxes + salience + composition cues |
| Reframing solver | D | maximize salience under composition rules |
| Not-fit strategy | D | prioritize / fit+pad / layout |
| Pad style | D | blurred-bg or letterbox |
| Temporal tracking | — | parked |

### 12.8 LLM (L3)
| Task | Engine | Notes |
|---|---|---|
| Ask-user loop | L | relight `awaiting_user`; few high-leverage Qs |
| Parse NL steering | L | "warmer", "less teal" → dial/CDL |
| Explain the grade | L | trust + steering |
| Rank/suggest looks | L | never auto-applies |

### 12.9 Grade representation + render + export
| Task | Engine | Notes |
|---|---|---|
| Grade object (CDL + LUT + working_space) | D | CDL round-trips; extras bake into LUT |
| `grade` block on Edit Document | D | per-clip + sequence-level; versioned |
| Render grading stage | D | CDL + `lut3d`; **proxy = export math** |
| Preview thumbnail cache | D | per look × frame |
| Re-grade triggers | D | timeline change → regroup/re-arc; decoupled from cut |
| Export bundle | D | `.cdl`/`.ccc` + `.cube` + ref in XML/EDL; editable-vs-baked toggle |

### 12.10 UI
| Task | Engine | Notes |
|---|---|---|
| Grade panel | D | fills `project-lenses.tsx` "Colour grading — Coming soon" |
| Before/after toggle | D | on `CompositePreview` |
| Controls | D | look picker + intensity dial + reference/LUT drop + NL steering box |

### 12.11 Local/spatial (Tier 2 — ambitious, later)
| Task | Engine | Notes |
|---|---|---|
| Directional split / graduated sky / highlight bloom / depth pop | D+V | **soft only**; breaks clean CDL → bakes to LUT; crisp masks need segmentation |

---

## 13. Suggested build order

1. `color_stats` (§12.1)
2. Correct layer (§12.3)
3. Render grading stage (§12.9) — prove the CDL+LUT loop with parity
4. VLM additions (§12.2)
5. Match layer (§12.4) — the auto-group advantage
6. Look layer + default gallery (§12.5)
7. Arc (§12.6)
8. Subject placement / reframing (§12.7)
9. LLM ask/steer/explain (§12.8)
10. UI (§12.10)
11. Local/spatial Tier 2 (§12.11) — last

Ship value as early as step 3–6 (correction + consistency + a default look) before
the arc and local work land.
