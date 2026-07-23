# Reliable Caption Styles MVP

## Goal

Replace the current broad caption gallery with five reliable choices:

1. **Standard** — always first, deterministic, and editable.
2. **AI Pick 1**
3. **AI Pick 2**
4. **AI Pick 3**
5. **AI Pick 4**

Every choice applies one consistent style to the entire video. Regeneration changes only the four unapplied AI suggestions. Per-cut style changes are out of scope.

## Product contract

### Allowed values

| Category | Values |
|---|---|
| Colour | White `#FFFFFF`; Vibrant Yellow `#FFEB3B`; Cyan `#00E5FF`; Charcoal `#1A1A1A` |
| Outline | Off; On |
| Shadow | Off; On |
| Position | Lower Third; Dead Center; Upper Third; Bottom Dynamic |
| Animation | Active Reader; Pop/Bounce; Smooth Fade Up; Sequential Reveal |
| Font | Montserrat ExtraBold; Anton; Jost Heavy; Inter SemiBold |
| Case | Original; Sentence; UPPERCASE; lowercase |
| Size | Small; Regular; Large; Extra Large |

Do not expose animation intensity, tracking, outline width, shadow parameters, line limits, beat sync, or arbitrary colours in the MVP.

### Permanent Standard

The Standard card is hand-authored and is never regenerated:

- Inter SemiBold
- White
- Lower Third
- Smooth Fade Up
- Original case
- Regular size
- Outline off
- Subtle shadow on

Users can customize the applied Standard using the same controls as every other card. Resetting restores these values.

### Font licensing

Use distributable open-source fonts:

- Montserrat ExtraBold — modern workhorse
- Anton — punchy creator font
- Jost Heavy — Futura-like geometric font
- Inter SemiBold — neutral standard

Bundle the real files for both preview and export. Do not rely on system fallback fonts.

## Implementation

### 1. Simplify the style schema and catalog

Update `backend/app/services/l3/captions/styles.py`.

- Expose only the approved four-value catalogs.
- Reduce `STANDARDS` to the permanent Standard.
- Add `outline_enabled` and `shadow_enabled` booleans.
- Add a four-value font-size enum.
- Expand case handling to Original, Sentence, UPPERCASE, and lowercase.
- Replace public placement values with:
  - `lower_third`
  - `center`
  - `top`
  - `bottom_dynamic`
- Replace public animation values with:
  - `active_reader`
  - `pop`
  - `fade_up`
  - `sequential_reveal`
- Continue storing the complete selected style in `document.captions.base_style`.
- Keep legacy style parsing internally so previously saved documents still render.

Use these size mappings consistently in preview and export:

| Size | Frame-height percentage |
|---|---:|
| Small | 3.6% |
| Regular | 4.5% |
| Large | 5.5% |
| Extra Large | 6.5% |

### 2. Implement the four fixed positions

Update `backend/app/services/l3/captions/placement.py`.

| Position | Approximate vertical center |
|---|---:|
| Lower Third | 80–82% |
| Dead Center | 50% |
| Upper Third | 12–15% |
| Bottom Dynamic | 70–73% |

- Apply existing aspect-specific safe margins.
- Keep the selected position fixed throughout the video.
- Do not use `dynamic`, `speaker`, or per-cut `caption_zones` for these styles.
- Suggestion ranking may inspect subject information when choosing a global position.
- Fall back to Lower Third when analysis is unavailable.

### 3. Implement deterministic colours, outline, and shadow

Update:

- `backend/app/services/l3/captions/colour.py`
- `backend/app/services/l3/captions/ass_export.py`
- `frontend/src/components/preview/caption-overlay.tsx`

Rules:

- Outline is off unless the selected style explicitly enables it.
- Shadow is off unless explicitly enabled.
- Use one fixed restrained outline width.
- Use one fixed subtle shadow treatment.
- White, Yellow, and Cyan receive a dark outline/shadow when enabled.
- Charcoal receives a light outline/shadow when enabled.
- Remove dynamic palette, grade-matching, and high-contrast colour sources from the public MVP catalog.
- Ensure CSS preview and ASS export render the toggles consistently.

### 4. Implement the four animations

Update:

- `frontend/src/lib/resolve-captions.ts`
- `frontend/src/components/preview/caption-overlay.tsx`
- `backend/app/services/l3/captions/ass_export.py`

Definitions:

1. **Active Reader** — the full caption is visible; the currently spoken word changes colour or brightness.
2. **Pop/Bounce** — the emphasized word starts around 80%, peaks at no more than 105%, and settles at 100%.
3. **Smooth Fade Up** — the complete caption moves approximately 10–15 reference pixels while fading in.
4. **Sequential Reveal** — words appear individually according to their word timestamps.

Requirements:

- Every entry transition completes within 200ms.
- Remove beat synchronization from the MVP suggestion path.
- Preview and ASS export must use the same timing semantics.

### 5. Bundle fonts for preview and export

Add licensed assets under:

- `frontend/public/fonts/`
- `backend/app/services/render/caption_fonts/`

Update frontend `@font-face` declarations in `frontend/src/app/globals.css`.

Requirements:

- Use WOFF2 in the frontend and TTF/OTF for libass.
- Match font family names exactly across CSS and ASS.
- Include the appropriate OFL license files.
- Detect and report missing backend caption fonts during development or tests.
- Never silently display one font in preview and export another.

### 6. Replace suggestion archetypes with constrained ranking

Rewrite `backend/app/services/l3/captions/suggest.py` to return exactly four suggestions.

Use existing signals only:

- Aspect ratio
- Content type
- Dominant energy
- Speaker count
- Speech pace
- Musical/non-musical status
- Shot-size and subject-box distribution
- Average footage brightness/palette
- Timeline structure

Process:

1. Construct combinations from the approved values.
2. Reject incompatible combinations.
3. Score valid combinations against the edit signals.
4. Select the highest-scoring combination.
5. Select three additional combinations with a diversity penalty.
6. Return four distinct complete style bundles with short rationales.

Hard constraints:

- Do not return Charcoal for predominantly dark footage without protection.
- Do not return Yellow or Cyan against similar footage without protection.
- Avoid Center for face-heavy talking-head footage.
- Avoid Upper Third when subjects consistently occupy the upper frame.
- Do not use Sequential Reveal when word timestamps are incomplete.
- Avoid Pop/Bounce for calm corporate or documentary content.
- Do not combine Extra Large with long lines.
- Do not combine lowercase with Anton.

Diversity constraints:

- Do not repeat the same font more than twice.
- Do not repeat the same animation more than twice.
- Include at least one restrained choice.
- Include at most two high-energy choices.
- Avoid visually near-identical suggestions.

Regeneration:

- Continue using `reshuffle_seed`.
- Preserve hard constraints.
- Change secondary valid choices through deterministic score perturbation.
- Never modify the selected/applied style.
- Never regenerate Standard.

### 7. Make the suggestions endpoint fail safely

Update `backend/app/routers/captions.py`.

- Return one Standard and exactly four suggestions.
- Fetch optional analysis signals independently.
- Missing colour, audio, or cut analysis must not fail the endpoint.
- Generate four reasonable defaults when optional signals are absent.
- Continue returning `representative_frame` and `sample_words`.
- Only missing/unauthorized edit documents should prevent a response.

Target response shape:

```json
{
  "standard": {},
  "suggestions": [{}, {}, {}, {}],
  "representative_frame": {},
  "sample_words": []
}
```

### 8. Replace the frontend gallery

Update `frontend/src/components/captions-view.tsx`.

- Remove separate Suggested and Standards galleries.
- Render one five-card gallery:
  1. Standard
  2. AI Pick 1
  3. AI Pick 2
  4. AI Pick 3
  5. AI Pick 4
- Use the same representative video frame and sample words for all cards.
- Display each AI suggestion's short rationale.
- Keep Regenerate beside the AI suggestions.
- Do not automatically apply any generated suggestion.
- Preserve the currently applied style while regeneration is in progress.

After selection, expose only:

- Font
- Colour
- Position
- Animation
- Case
- Size
- Outline toggle
- Shadow toggle
- Turn captions off
- Reset to selected card

### 9. Update frontend types

Update `frontend/src/lib/api.ts`.

- Replace legacy public animation, placement, and colour unions.
- Add the size enum and four case values.
- Add outline/shadow booleans.
- Replace `Record<string, unknown>` overrides with a typed caption-overrides interface.
- Model the new Standard-plus-four API response.
- Retain legacy values only where needed to read old documents.

### 10. Preserve old saved documents

Normalize legacy snapshots while reading/resolving; do not rewrite historical versions.

Suggested mappings:

| Legacy value | MVP equivalent |
|---|---|
| `inter_tight` | Inter |
| `poppins_extrabold` | Montserrat |
| `anton` | Anton |
| Other legacy fonts | Inter |
| `fade` | Smooth Fade Up |
| `karaoke` | Active Reader |
| `slide` | Smooth Fade Up |
| `dynamic` / `speaker` | Lower Third |
| Dynamic colour source | Nearest fixed palette colour |

## Testing

Add dedicated backend caption tests and frontend component/logic tests.

### Catalog and API

- Exactly four public values exist per category.
- Exactly one Standard exists.
- Standard has outline disabled.
- Suggestions endpoint always returns Standard plus four suggestions.
- Missing optional analysis still returns four suggestions.

### Ranking

- Same edit and seed produce identical results.
- Regeneration produces valid alternatives.
- No duplicate suggestions.
- Every field belongs to the approved catalog.
- Hard compatibility constraints are enforced.

### Rendering parity

- Outline off produces zero ASS outline width.
- Shadow off produces zero ASS shadow.
- All four fonts are available to libass.
- Preview and export use matching font sizes and positions.
- All animation entry transitions finish within 200ms.
- Active Reader and Sequential Reveal follow word timestamps.
- Preview and export use matching colours and animation behavior.

### Frontend

- Exactly five cards render.
- Standard is always first.
- Regenerate changes only the four AI cards.
- Regeneration does not change the applied style.
- Every customization control saves and restores correctly.

## Acceptance criteria

- The gallery always shows exactly five choices.
- Standard is always first and never AI-generated.
- Four suggestions use only approved catalog values.
- A selected style remains consistent throughout the video.
- Regeneration never changes the applied style.
- Outline is off by default.
- Preview and export use the same font, colour, position, size, and animation.
- Missing optional analysis never prevents suggestions.
- Legacy dynamic styles do not appear in the new gallery.
- All five choices can be adjusted through the same simple controls.

## Out of scope

- Per-cut style changes
- Free-form AI typography or CSS
- New face-tracking or visual-analysis models
- Manual subtitle text/timing editing
- Custom font uploads
- Arbitrary colour picker
- Beat synchronization
- Speaker-specific styling
- Dynamic palette or grade matching
