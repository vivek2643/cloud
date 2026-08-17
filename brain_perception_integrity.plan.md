# Brain PERCEPTION INTEGRITY — one granularity, no cached derivations, one definition per concept, and a status on every sense

Status: proposal, READ-ONLY handoff. **Implement from another chat. Do NOT
implement from this doc; do NOT commit from this doc.**

**Scope.** Four architecture-level changes to what the brain PERCEIVES, in
`backend/app/services/l3/` (`observe.py`, `feel.py`, `footage_map.py`, `act.py`,
`arrange.py`, `cutrecord_map.py`):

- **§1 — span granularity at source.** Three different "what does this span
  say?" readers with three different containment rules and two different source
  tables feed four senses. `review` reports two of them side by side, in one
  dict, disagreeing.
- **§2 — kill derived-field staleness generically.** `seg["content"]` is a
  cached derivation of a moment label that drifts on `trim`, is duplicated on
  `retime`, and is concatenated on `heal`. It is also *not what its name says*,
  and `feel.py` computes the program's speaking pace from it.
- **§3 — reconcile the senses that disagree.** One definition per concept:
  program clock, cut position, program median loudness, and the `resolved`
  snapshot.
- **§4 — the sense status envelope.** Every sense returns `observed` / `empty` /
  `unavailable`, so "nothing there" and "couldn't see" stop being the same
  answer — for the brain AND for gate policy.

Plus **§5** migration/back-compat, **§6** verification, **§7** the complete
enumeration of drift instances found, and **§8** what is deliberately excluded.

**Sequencing (recorded identically in all three plans).**

1. `junk_span_suppression.plan.md` — independent, does not touch L3.
2. `brain_gate_integrity.plan.md` §1 (evaluate/commit split) — must precede any
   new gate stage.
3. **`brain_perception_integrity.plan.md` (this file)** — independent of the
   accountability parts; unblocks them, since
   `brain_accountability_architecture.plan.md` PARTS 2, 3 and 4 all consume
   perception.
4. `brain_gate_integrity.plan.md` §2 (coverage gate).
5. `brain_accountability_architecture.plan.md` PART 2, then PART 3, then
   PARTS 4/5 (PART 5 recommended earlier — see its amendment §A.5).
6. Beat binding + the adversarial quote check, last (amendment §A.6).

**Cross-references.** §4 is consumed by `brain_gate_integrity.plan.md` §1.6
(an `unavailable` sense must BLOCK a mandatory stage, not satisfy it) and §2.3
(a coverage rule must not be satisfied by a degraded sense) — ship §4 with
them or the status field is inert. §1 is a prerequisite for the adversarial
quote check in `brain_accountability_architecture.plan.md` amendment §A.6: a
quote check against wrong words verifies nothing.

---

## 0. Verified current state, and four corrections to the brief

Verified against HEAD, commit `3337c31`. Everything below was read, not
inferred. Where the commissioning brief does not match HEAD, it is corrected
plainly rather than papered over — three of the four corrections make the
underlying defect *worse* than described, and one makes a claimed defect
currently unreachable.

### 0.1 CORRECTION 1 (major) — `seg["content"]` is not the played text; it is the VLM label

The brief says: *"`act.py:595-596` — `trim` writes `seg["in_ms"], seg["out_ms"]`
and leaves `seg["content"]` untouched … So after any trim, the brain's primary
sense describes the pre-trim span. Worse, `feel.py:166-169` derives word count
from that same stale string → `pace` → `energy`."*

The staleness is real and the causal chain is real. **What `content` contains is
not.** Traced end to end:

```96:99:backend/app/services/l3/act.py
            "axis": "speech" if rc.channel == "said" else "any",
            "beat_id": None,
            "content": rc.label,
            "rationale": rc.reason or None,
```

`rc.label` comes from `_MapIndex.resolve`:

```179:184:backend/app/services/l3/arrange.py
        return ResolvedCut(
            file_id=m["file_id"], src_in_ms=int(v["in_ms"]),
            src_out_ms=int(v["out_ms"]), keep_spans=_norm_keep_spans(v.get("keep_spans")),
            channel=m.get("channel"), label=m.get("gist") or "",
            track=p.track, from_ms=p.from_ms, reason=p.reason,
            ref=p.ref, level=v.get("level", level),
```

and `m["gist"]` is the vision model's short label:

```323:326:backend/app/services/l3/footage_map.py
            "gist": cut.get("label") or "",
            # beat_transcript.plan.md: verbatim dialogue_segments transcript
            # over this speech beat's own span -- '' for a non-speech beat.
            "said_text": said_text,
```

**So `seg["content"]` is a paraphrase like `"The speaker touches his chin while
thinking"`, never the spoken words.** Three consequences, each worse than
staleness:

1. `read_state.cuts[].text` (`observe.py:540-542`, emitted at `observe.py:565`)
   is a 60-char truncation of a VLM paraphrase, sitting in a field named `text`,
   beside `review.items[].played_text` which IS the transcript. Two senses, two
   incompatible meanings for "what this cut says". **This, not the staleness, is
   why the brain in thread `34bb9967-97aa-4818-9f21-01145156a9bb` trimmed, saw
   `text` unchanged, concluded the trim had failed, trimmed again, and burned
   ~4 turns.** A trim of 800 ms would not change a VLM label even if the field
   were refreshed on every write — the field cannot express the change the brain
   was looking for.
2. `feel.CutFeel.pace_wps` is documented as spoken words per second:

```77:79:backend/app/services/l3/feel.py
    words: int
    pace_wps: float          # spoken words per second (0 for silent/video cuts)
    is_speech: bool
```

   and is computed as label-words per second:

```166:169:backend/app/services/l3/feel.py
        content = (seg.get("content") or "").strip()
        words = len(content.split()) if content else 0
        is_speech = seg.get("axis") == "speech"
        pace = round(words / (dur / 1000.0), 2) if (is_speech and words and dur) else 0.0
```

   **`pace_wps` is not a measurement of anything.** It flows into
   `_score_energy` (`feel.py:204-206`), `energy`, `_low_energy_runs`
   (`feel.py:219-224`), and from there into `diagnose`'s low-energy finding
   (`observe.py:1222-1224`), the mirror's `dead-air` flag
   (`observe.py:1670-1671`), `read_state["feel"]`'s narration `"~N words/s when
   talking"` (`feel.py:110`, surfaced at `observe.py:631`), and
   `read_state["feel_detail"]` (`observe.py:632`). Four downstream signals, as
   the brief says — but corrupted at the source, not merely stale.
3. `act.place_span` writes the caller's `content` argument, defaulting to `""`
   (`act.py:142`, `act.py:172`). A `place_span`-placed speech span therefore has
   `words = 0` → `pace_wps = 0.0` → maximum `short`-only energy, regardless of
   how densely it is spoken.

**The correct word-count reader already exists in a sibling module** and is
already used by the mirror: `cutrecord_map.speech_words_in_span(file_id, in_ms,
out_ms)` returns `(word_count, has_speech, is_musical)` off the file's own
`transcripts` words (`cutrecord_map.py:528-549`), and the mirror calls it at
`observe.py:1659-1660`. `feel.py` cannot call it directly — `feel` is
deliberately PURE with no DB (`feel.py:11-14`), and that property is worth
keeping — so §2.4 threads the counts in from `observe`, which already does the
DB reads once (`observe.build_context`).

### 0.2 CORRECTION 2 (minor, line numbers) — the overlap filter is at `footage_map.py:1517`

The brief cites `footage_map.py:1516-1519` as the root. The function is defined
at `:1508`; the overlap filter is line `:1517`:

```1508:1521:backend/app/services/l3/footage_map.py
def _said_text_for_span(file_id: str, in_ms: int, out_ms: int) -> str:
    """Verbatim transcript for a SPEECH beat's own span (no padding, unlike
    `_span_detail`'s Tier-1 window) -- the words actually spoken, joined into
    one string for the resident beat line. Speaker labels are prefixed only
    when the beat spans more than one speaker (post speaker-change-split,
    almost always exactly one) -- otherwise just the words, since repeating
    a lone speaker's name on every beat would be noise. '' when this file has
    no transcript, or none of it overlaps the span (never a fabricated line)."""
    in_span = sorted(
        (s for s in _sentences_for_file(file_id) if _overlap_ms(*_sentence_span(s), in_ms, out_ms) > 0),
        key=lambda s: _sentence_span(s)[0],
    )
    if not in_span:
        return ""
```

Note the docstring's own claim: *"the words actually spoken"*. It admits any
sentence that touches the span, whole. The claim and the code disagree, which is
how four callers came to trust it.

### 0.3 CORRECTION 3 (major) — there is a FIFTH caller, and the disagreement has THREE axes not one

The brief names four callers of `_said_text_for_span`. There are four
(`observe.py:1516`, `observe.py:1592`, `observe.py:1662`,
`footage_map.py:291`), and a **fifth caller of the same sentence-overlap
pattern, inlined**:

```1288:1296:backend/app/services/l3/observe.py
    file_id = seg.get("file_id") or ""
    in_ms, out_ms = int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
    sentences = sorted(
        (s for s in footage_map._sentences_for_file(file_id)
         if footage_map._overlap_ms(*footage_map._sentence_span(s), in_ms, out_ms) > 0),
        key=lambda s: footage_map._sentence_span(s)[0],
    )
    if not sentences:
        return []
```

That is `observe._head_tail_flags` (`observe.py:1283`), which produces
`review`'s `guidance`-category flags — including the *"~X.Xs of dead air before
the first word"* / *"after the last word"* findings (`observe.py:1333-1340`),
which are computed from `_sentence_span(sentences[0])` and
`_sentence_span(sentences[-1])`. **Because the first and last sentences may
extend outside the cut, `first_s - in_ms` can be NEGATIVE and `out_ms - last_e`
can be negative**, so the dead-air flags silently fail to fire on exactly the
cuts whose edges sit mid-sentence — the case they exist for. A copy of the bug
that is invisible from the shared-helper call graph.

Second, and more important: the brief describes `played_text` and `words` as
*"the two disagree"*. They disagree on **three independent axes**, from **two
different database tables**:

| | `played_text` (`observe.py:1516`) | `words` (`observe.py:1520`) | `mirror` has-speech (`observe.py:1659`) |
|---|---|---|---|
| helper | `footage_map._said_text_for_span` | `observe._word_offsets_for_seg` → `captions_timing.words_in_source_window` | `cutrecord_map.speech_words_in_span` |
| source table | `dialogue_segments` (`footage_map.py:1401`) | `transcripts` (via `captions_resolver.fetch_transcripts`, `observe.py:482`) | `transcripts` (`cutrecord_map.py:498`) |
| granularity | whole SENTENCES | WORDS | WORDS |
| containment | overlap > 0 → sentence admitted WHOLE (`footage_map.py:1517`) | word must be FULLY INSIDE (`captions/timing.py:73`) | word OVERLAPS (`cutrecord_map.py:542`) |
| fillers | KEPT | DROPPED (`timing.py:70-71`) | DROPPED (`cutrecord_map.py:506`) |

So `review.items[i]["played_text"]` and `review.items[i]["words"]` — adjacent
keys in one dict, both documented in one docstring (`observe.py:1495-1498`) —
can disagree about the content of the same span for three separate reasons, and
neither is reconcilable with the other by inspection. `words` is *closer* to
correct, as the brief says, but it is not simply correct: it drops a word that
straddles the cut edge rather than reporting that the cut severs it, and it
drops fillers, so it cannot answer "does this cut open on an 'um'" — which is a
question `_head_tail_flags` (`observe.py:1322-1329`) is separately trying to
answer off the sentence reader.

### 0.4 CORRECTION 4 — the `total_ms` divergence is REAL IN CODE but NOT REACHABLE at HEAD

The brief says: *"`read_state.total_ms` comes from `feel.simulate` which SKIPS
degenerate segments (`feel.py:163-164`) while `review.total_ms` sums all of them
(`observe.py:1538`) — two senses report two different program lengths."*

The code difference is exactly as described:

```160:165:backend/app/services/l3/feel.py
    for i, seg in enumerate(timeline):
        src_in_ms, src_out_ms = int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
        dur = max(0, src_out_ms - src_in_ms)
        if dur <= 0:
            continue
        total += dur
```

versus `review`'s `dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))`
(`observe.py:1514`) and `prog += dur` (`observe.py:1538`), unclamped.

**But the two totals can only differ when a segment has `out_ms < in_ms`, and
`out_ms <= in_ms` is rejected at every ingress:**

- the HTTP save path: `_sanitize_timeline` raises 422
  (`backend/app/routers/edit_threads.py:86-87`);
- `act.trim`: `EditReject` (`act.py:587-591`, and `act.py:605-609` for ops);
- `act.place_span`: `EditReject` (`act.py:160-161`);
- `act._segments_from_cut`: skipped (`act.py:89-90`);
- `act.move`, `act.set_audio`, `act.retime`, `act.tighten` never write a span
  narrower than the map's own (`act.py:1020-1027`, `act.py:918-936`).

For a segment with `out_ms == in_ms` the totals agree (both add 0), and per the
reachability rule this plan is held to, **a divergence that cannot be provoked
on production-shaped data is not a shipped bug and must not be counted as
one.** §3.1 still unifies the definition — four independent program clocks in
one module is a class regardless — but it does so with an INVARIANT assertion,
not a gate, and its verification is honest about being a latent-class fix. The
brief's framing that these are "two senses reporting two different program
lengths" today is not supportable; what IS supportable is that they are two
definitions of one concept, and the next verb that produces a degenerate span
turns that into four wrong answers at once.

**What IS live and reachable in the same family** is `read_state`'s *position*
numbering versus `feel`'s (§3.2) — only in the degenerate case, so also latent —
and the `loudness_rel` divergence (§3.3) and the `resolved` snapshot (§3.4),
both of which are reachable today and one of which is worse than "stale".

---

## 1. §1 — SPAN GRANULARITY AT SOURCE

### 1.1 The class, stated once

> **"What does this span say?" must have exactly ONE answer, computed once, at
> one granularity, with one containment rule — and every sense that reports
> words must report a projection of that one answer.**

### 1.2 The false-genericity trap, named first

**The trap: write a `_text_from_words()` helper and call it at the `review`
site.** A shared function with one caller is still a point fix. It will look
generic in review — a new module-level helper, a docstring, a test — and it
changes nothing about the three other readers.

The real test of this section is not "is there a helper". It is:

1. `review.played_text`, `read_transcript`, and the mirror's `gist` become
   **IMPOSSIBLE to disagree** — not "are made consistent", but derive from one
   call whose result is passed to all three, so a divergence is not expressible;
2. `review.items[i]["played_text"]` and `review.items[i]["words"]` are the same
   data in two shapes (a joined string and a list of timed words), so the string
   is `" ".join(w["text"] for w in words)` by construction;
3. `read_state.cuts[].text` stops surviving a trim (§2).

A second trap specific to this section: fixing the containment rule
(`overlap > 0` → `fully inside`) in `_said_text_for_span` and stopping there.
That silently changes the **Beat Index** for every project, which is a
presentation decision made by accident — see §1.4, where it is made
deliberately and differently.

### 1.3 One reader: `l3/spoken.py`

A new module, so `footage_map.py` (1581 lines) does not grow another
responsibility and `observe.py` does not import a private helper across a module
boundary (which it does today at `observe.py:1291`).

```python
"""The ONE reader of "what is spoken over a source span".

Every sense that reports words to the brain projects THIS module's result.
Before it, three readers disagreed on granularity (sentences vs words),
containment (overlap vs fully-inside) and fillers (kept vs dropped), and
`observe.review` reported two of them side by side in one dict.

PURE-ish: DB reads are @lru_cache'd per file exactly as
footage_map._sentences_for_file and cutrecord_map._words_for_file already are
(both tables are immutable once L1 writes them). No LLM, no network.
"""

@dataclass(frozen=True)
class SpokenWord:
    text: str
    src_start_ms: int
    src_end_ms: int
    is_filler: bool
    speaker: Optional[str]
    # True when the word's own audio extends past the span's edge -- the cut
    # SEVERS this word. Kept rather than dropped (captions/timing.py:73 drops
    # it) because "this cut cuts a word in half" is a fact the brain must be
    # able to see; a consumer that wants clean words filters on it.
    clipped: bool


@dataclass(frozen=True)
class Spoken:
    """What is spoken over ONE source span, at word granularity, with the
    surrounding sentence context kept SEPARATE rather than mixed in."""
    status: str                     # STATUS_OBSERVED | STATUS_EMPTY | STATUS_UNAVAILABLE (§4)
    reason: str                     # '' unless status is unavailable
    words: Tuple[SpokenWord, ...]   # every word overlapping the span, in time order
    # The sentences the span TOUCHES, verbatim and unclipped -- context, never
    # presented as what plays. This is what the Beat Index legitimately wants
    # (§1.4) and what `review.played_text` illegitimately reported.
    context_sentences: Tuple[Dict[str, Any], ...]

    def text(self, *, fillers: bool = True, clipped: bool = True) -> str:
        """The words that ACTUALLY PLAY over the span, joined. This is the ONE
        source of every spoken-text string the brain reads."""

    def word_count(self, *, fillers: bool = False) -> int:
        """Spoken-word count over the span -- what feel.py's pace_wps has always
        claimed to measure (feel.py:78) and never did (§0.1)."""

    def context_text(self) -> str:
        """The touched sentences, whole. NAMED so no caller can mistake it for
        what plays -- the old _said_text_for_span returned exactly this under a
        name and docstring that promised the opposite."""


def spoken_for_span(file_id: str, in_ms: int, out_ms: int) -> Spoken:
    """The one call. Word source: `transcripts.segments[].words` (the same rows
    cutrecord_map._words_for_file reads, cutrecord_map.py:487-508) -- word-level
    timings are the finest truth L1 stores, so the coarser sentence grid is
    derived context, never the primary. Sentence source: `dialogue_segments`
    (footage_map._sentences_for_file, footage_map.py:1388-1410).

    CONTAINMENT: a word is IN the span when its own audio overlaps it
    (`end_ms > in_ms and start_ms < out_ms`) -- the same predicate
    cutrecord_map.speech_words_in_span already uses (cutrecord_map.py:542), so
    "does this span carry speech" and "what does it say" agree by construction.
    A word whose audio crosses an edge is kept with clipped=True rather than
    dropped, so severing a word is VISIBLE instead of silently disappearing.

    FILLERS are kept and FLAGGED, not dropped. Dropping them at the source is
    what made observe._head_tail_flags reach for the sentence reader to answer
    "does this cut open on an 'um'" (observe.py:1322-1329) -- the fix is one
    reader with a flag, not two readers with two policies.
    """
```

Note the deliberate ordering of authority: **word timings are primary, sentences
are derived context.** The reverse (sentences primary, words derived) is not
available — you cannot recover word boundaries from a sentence row — and it is
the choice that produced the bug.

Two existing readers become thin wrappers rather than being deleted, because
both have callers outside the brain's read path and both are already correct for
what they do:

- `cutrecord_map.speech_words_in_span` (`cutrecord_map.py:528`) — reimplement as
  `spoken_for_span(...)` + `word_count`, preserving its exact
  `(int, bool, bool)` return so `_audio_mute_for` (`cutrecord_map.py:591`) and
  the mirror (`observe.py:1659`) are untouched. It already uses the chosen
  containment rule, so this is a de-duplication, not a behaviour change.
- `captions_timing.words_in_source_window` (`captions/timing.py:60`) — **left
  ALONE.** Its fully-inside/drop-fillers policy is correct for CAPTIONS (a
  half-word caption reads worse than an early cut — its own docstring,
  `timing.py:65-66`) and captions are not a brain sense. §8 records this.

`footage_map._said_text_for_span` is **deleted**, not wrapped. A helper whose
name and docstring promise the words that play, while returning touched
sentences, cannot be kept as an alias — the next caller reads the name. Its
four call sites are rewritten in §1.4. `_sentences_for_file` and `_sentence_span`
stay (used by `snap_speech_spans_to_sentences`, `footage_map.py:1425`, and
`_span_detail`, `footage_map.py:1535`) and become `spoken.py`'s sentence source.

### 1.4 The four callers, decided one at a time

**(a) `observe.review` — `played_text` AND `words` from ONE call.** This is the
requirement "derive them from one call so they cannot disagree by
construction":

```python
    for i, seg in enumerate(timeline):
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        is_speech = seg.get("axis") == "speech"
        # ONE read; played_text and words are two SHAPES of it, so the two keys
        # in this dict cannot describe different content (they did: sentences-
        # that-overlap vs words-fully-inside, from two different tables, one
        # keeping fillers and one dropping them -- brain_perception_integrity
        # .plan.md §0.3).
        sp = spoken.spoken_for_span(seg.get("file_id") or "",
                                    int(seg.get("in_ms", 0)),
                                    int(seg.get("out_ms", 0))) if is_speech else spoken.NONE
        items.append({
            "idx": i + 1, "id": seg.get("seg_id"), "ref": seg.get("ref"),
            "played_ms": dur,
            "played_text": sp.text(),
            "words": [{"text": w.text,
                       "prog_start_ms": prog + (w.src_start_ms - int(seg.get("in_ms", 0))),
                       "prog_end_ms": prog + (w.src_end_ms - int(seg.get("in_ms", 0))),
                       "clipped": w.clipped} for w in sp.words],
            "speech_status": sp.status,           # §4
        })
```

Three side effects worth naming:

- `_word_offsets_for_seg` (`observe.py:470-490`) currently calls
  `captions_resolver.fetch_transcripts([file_id])` **once per segment, inside
  `review`'s loop** — an N+1 read on every `review` call, and `review` is
  computed UNCONDITIONALLY on every gate evaluation (`tools.py:1042`). Routing
  through `spoken_for_span`'s per-file `lru_cache` makes it one read per file
  per process. This is a performance improvement, not a cost.
- The program-time mapping moves inline. Keep `captions_timing._to_program`'s
  arithmetic identical (`captions/timing.py:79-82`) so caption timings and
  `review`'s word offsets stay in lockstep; a test should assert they agree on a
  shared fixture.
- `read_state`'s `seg_id` word-offset detail (`observe.py:693-703`) uses the same
  helper and must be migrated in the same change, or `read_state.word_offsets`
  and `review.words` diverge on filler policy — replacing one disagreement with
  another.

**(b) `observe.read_transcript` (`observe.py:1592`)** — `sp.text()`. Its whole
purpose is *"the exact words … to check wording before placing a cut"*
(`observe.py:1580-1583`). Today it returns overlapping sentences, so the brain
checking wording before a cut reads words the cut will not contain. Straight
substitution. Keep the deliberate axis-ungated read (`observe.py:1577-1580`) —
incidental speech under a picture cut must stay visible.

**(c) The mirror's per-cut `gist` (`observe.py:1662`)** — `sp.text()`, then
`_bounded_gist`. This is the highest-frequency surface in the system: injected
after EVERY tool call at `tools.py:1480`. **This caller is why patching only
`review` makes things actively worse**: `review` would deny a tail the mirror
kept showing, on every single step, and the mirror wins on frequency. The two
must change together in one commit; there is no partial landing of §1.

One additional defect at this site, worth fixing while it is open:

```1613:1617:backend/app/services/l3/observe.py
def _bounded_gist(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _MIRROR_GIST_HEAD_CHARS + _MIRROR_GIST_TAIL_CHARS + 1:
        return text
    return f"{text[:_MIRROR_GIST_HEAD_CHARS].rstrip()}…{text[-_MIRROR_GIST_TAIL_CHARS:].lstrip()}"
```

The result reads as a quotation (it is rendered inside literal double quotes at
`observe.py:1703`) but is a head+tail splice. Once §1 lands, the two ends are
genuinely from the span — but they are still not contiguous. Render the elision
visibly and label the field: `speech(audible): "first words …" … "last words"`
or keep the `…` and add `(head…tail)` to the mirror's own legend. This matters
because `brain_accountability_architecture.plan.md` amendment §A.6 makes the
brain QUOTE played words to justify a beat; a spliced "quote" from the mirror
would be checkable against nothing.

**(d) `footage_map.build_clip_tree`'s `said_text` (`footage_map.py:291`) —
DECISION: this caller KEEPS sentence granularity, and the field is RENAMED.**

Defended, because the brief asks:

*Keep sentences here.* The Beat Index is a CATALOG of candidate material the
brain has not placed yet. Its job is to help the brain decide whether a moment
is worth placing, and for a speech moment the surrounding sentence is frequently
the point — a 1.2 s moment whose words are "…and that's the whole thesis" is
only evaluable with the sentence it sits in. The moment's own boundaries were
set by the segmenter from motion/audio/scene signals
(`converse.py:294-302`), not by meaning, so clipping the catalog entry to
those boundaries would hide the meaning the boundaries happen to bisect. This is
also the one caller reading a span the brain has NOT committed to, so
"what would actually play" is not yet a question.

*But rename it, and add the played words beside it.* The field is currently
`said_text` and is rendered by `_moment_line` (`footage_map.py:1136`) with no
indication that it may exceed the beat. Two changes:

1. rename the moment key to `context_text` (matching `Spoken.context_text()`),
   and
2. add `played_text` from `sp.text()` beside it, rendered only when it DIFFERS
   from the context — so the Beat Index shows both the sentence and the subset
   that would actually play, and the brain can see the difference instead of
   inferring it.

**This second half is what makes the decision defensible rather than an
exemption.** Keeping sentence granularity while calling it "what is said" is the
original bug; keeping it under an honest name, beside the honest subset, is a
presentation choice. If the extra token cost is judged too high for a resident
prompt block, render `played_text` only when the context sentence extends more
than a threshold beyond the beat — but the KEY must exist in the dict either
way, so `inspect_cut` and any future consumer see it.

`observe.build_context`'s `meta_by_ref` carries the moment dicts through
(`observe.py:176-182`), so anything reading `moment["said_text"]` must be
migrated. Grep before editing: at HEAD the readers are `footage_map:1136` and
the `beat_transcript.plan.md` / `speech_label_and_shot_continuity_fix.plan.md`
references. `_moment_line`'s rendered output is asserted in
`scripts/test_footage_map.py` (multiple `_sentences_for_file` monkeypatches at
`:295`, `:336`, `:367`, `:403`, `:923-971`), all of which must be updated
deliberately, not mechanically — each one encodes an expectation about what the
brain reads.

**(e) The fifth caller, `observe._head_tail_flags` (`observe.py:1283-1341`)** —
rewritten onto `spoken_for_span`. This fixes a latent bug of its own: the
dead-air flags compare the cut's edge against the first/last *sentence* span
(`observe.py:1331-1340`), which can lie outside the cut, so `first_s - in_ms`
goes negative and the flag never fires precisely when the cut opens mid-line.
With words, `first_word.src_start_ms - in_ms` is the real silent lead-in. The
filler check (`observe.py:1322-1329`) uses `_is_filler_sentence`
(`observe.py:1278-1280`), a local `_FILLER_WORDS` set (`observe.py:1262-1265`)
that duplicates L1's own `is_filler` word flag — replace it with
`w.is_filler`, and delete `_FILLER_WORDS` and `_is_filler_sentence`. **Two
filler vocabularies is the same class as two granularities**, and this one is a
hardcoded English word list next to a per-word L1 signal.

### 1.5 What §1 explicitly does NOT change

- `captions_timing.words_in_source_window` and everything under
  `l3/captions/` — different consumer, different correct policy (§8).
- `footage_map.snap_speech_spans_to_sentences` (`footage_map.py:1425`) — it
  snaps cut EDGES to sentence boundaries at placement time, which is a
  deliberate editorial rule (`act._snap_cut_to_sentences`, `act.py:116-131`),
  not a perception read. Note the interaction: because placement snaps speech
  cuts to sentence edges, a freshly placed speech cut usually has
  `played_text == context_text`, and §1's whole effect appears only after a
  `trim` or a `retime` re-slice. **That is exactly the case the brain got wrong**
  — and it means a fixture built only from fresh placements will show no
  difference and prove nothing. §6 V1 accounts for this.
- `footage_map._span_detail` (`footage_map.py:1535`) — the Tier-1 padded window
  is INTENTIONALLY wider than the span (`_DETAIL_PAD_MS = 1200`,
  `footage_map.py:1532`) and says so. Correct as-is.

---

## 2. §2 — KILL DERIVED-FIELD STALENESS GENERICALLY

### 2.1 The class, stated once

> **A field derived from other fields must not be STORED. Derive it on read, or
> the writer of every future verb must remember to refresh it — and one of them
> will not.**

`seg["content"]` is the instance. The class is a cache with no invalidation and
no owner.

### 2.2 DECISION: `seg["content"]` is ELIMINATED as a stored field

Recommended, with the cost weighed.

**Three independent producers already disagree about what it holds**, which is
the strongest argument that no writer can own it:

1. `act._segments_from_cut` (`act.py:98`) — the moment's VLM label.
2. `act.place_span` (`act.py:172`) — whatever the caller passed; `""` by
   default.
3. `arrange.heal_adjacent_cuts` (`arrange.py:291-293`) — **concatenates** the
   two merged segments' contents:

```291:293:backend/app/services/l3/arrange.py
            if s.get("content") and s["content"] != prev.get("content"):
                prev["content"] = f"{(prev.get('content') or '').strip()} "\
                                  f"{s['content'].strip()}".strip()
```

   So a healed segment's `content` is two labels joined, its word count roughly
   doubles, and `feel`'s `pace_wps` for that cut roughly doubles with it. And
   `heal_adjacent_cuts` runs on EVERY turn via `observe.resolve_doc`
   (`observe.py:317-318`) and on every human save
   (`edit_threads.py:218`) — this is not a corner.

4. And a fourth, silent duplicator: `act.retime`'s speech branch copies the
   whole segment dict onto every re-sliced piece:

```1025:1031:backend/app/services/l3/act.py
        for j, (a, b) in enumerate(kept):
            slc = {k: v for k, v in seg.items() if k not in ("speed", "pace_level")}
            slc["in_ms"], slc["out_ms"] = int(a), int(b)
            if frac > 0:
                slc["pace_level"] = pace
```

   `content` is carried onto all N slices, so one cut re-sliced into 3 pieces
   contributes 3× its label's word count to the program. `retime(pace="faster")`
   on a speech cut is a routine call.

**The change.** Delete `content` from the segment shape written by
`act._segments_from_cut` (`act.py:98`), `act.place_span` (`act.py:172`), and
the merge branch in `arrange.heal_adjacent_cuts` (`arrange.py:291-293`).
Replace every read with a derivation:

| reader | today | after |
|---|---|---|
| `read_state.cuts[].text` (`observe.py:540-542`, `:565`) | 60-char truncation of `seg["content"]` | **two keys**: `text` = `_bounded(sp.text())` (what actually plays, for a speech cut; `""` otherwise) and `label` = `meta.get("gist")` (the VLM paraphrase, honestly named). Both derived on read; neither stored. |
| `feel.CutFeel.words` / `pace_wps` (`feel.py:166-169`) | `len(content.split())` | a word count PASSED IN — §2.4 |
| `arrange` merged `content` | concatenation | nothing to merge |
| `arrange._label` (`arrange.py:321-323`) | truncates a passed string | check its callers; if any reads `seg["content"]`, migrate it |

Note `read_state` gaining a `label` key is not a compromise — it is the
separation the fix is for. The brain legitimately wants both "what does the
camera show here" (the VLM paraphrase) and "what words play here" (the
transcript). Today it gets one field that is the first, named as the second.
`footage_map` already models this correctly with distinct `gist` and `said_text`
keys (`footage_map.py:323-326`); `act`/`observe` collapsed them.

**Cost, weighed honestly.**

- *Cost:* `read_state` gains a per-speech-segment `spoken_for_span` call. Bounded
  by `lru_cache` per file and by timeline length (typically < 20 segments over
  < 6 files); `read_state` already calls `feel.simulate`, `feel.join_continuity`,
  `layers.resolve` (`observe.py:647`) and `footage_map.piece_breakdown` per
  segment. `review` already does per-segment transcript reads *without* the
  cache (`observe.py:482`), so §1 makes the net read count go DOWN.
- *Cost:* segments become slightly less self-describing in a stored jsonb row —
  a human reading `edit_documents` no longer sees a label inline. Mitigated: the
  `ref` is on the segment (`act.py:105`) and resolves to the moment; and
  `document["resolved"]` still carries `arrange._label`-style descriptions for
  the render path.
- *Benefit:* the field cannot be stale, cannot be duplicated by a re-slice,
  cannot be concatenated by a merge, and no future verb author has to know it
  exists. **A cache that can go stale versus no cache at all — with four
  producers that already disagree, there is no third option worth the
  ambiguity.**

**The alternative, rejected:** keep `content` and refresh it in `trim` (and
`retime`, and `heal`, and the next verb). That is a compensating write per verb,
it is what the brief calls the band-aid, and it does not fix the field being the
wrong kind of thing (§0.1) — a refreshed VLM label is still not the played
words, so the brain still sees no change after a trim and still burns turns.

**A partial elimination is also rejected and must be named**, because it is the
tempting middle: keeping `content` for `place_span` (where the caller supplied
it deliberately) while deriving it elsewhere. That yields a field which is
sometimes derived and sometimes stored, i.e. a field whose staleness depends on
provenance — strictly worse to reason about than either extreme. If a
caller-supplied note is wanted, `seg["rationale"]` already exists for exactly
that (`act.py:99`, `act.py:173`) and is never presented as text-that-plays.

### 2.3 `seg["level"]` — the second cached derivation (FIXED, smaller)

`seg["level"]` is written at placement from the resolved variant
(`act.py:106` ← `arrange.py:184`, `v.get("level", level)`) and is NOT updated by
`act.trim` (`act.py:595`). `affordances` computes the retake menu from it:

```1797:1800:backend/app/services/l3/observe.py
        variants = list((meta.get("variants") or {}).keys())
        cur = seg.get("level")
        tighter = [L for L in _LEVELS if L in variants and _idx(L) > _idx(cur)]
        wider = [L for L in _LEVELS if L in variants and _idx(L) < _idx(cur)]
```

So after a trim the cut's span matches no variant, yet `level` still claims one,
and `can_tighten_to` / `can_widen_to` are computed against a level the cut is no
longer at — offering "widen to balanced" on a cut already hand-trimmed narrower
than `tight`. **Fix:** on `trim` of a main-line segment, set
`seg["level"] = "custom"` (a token `_LEVELS` does not contain, so `_idx` falls
back to `"balanced"` at `observe.py:1910-1914` — which is why `_idx`'s fallback
must ALSO change, to return `None`/sentinel so `affordances` can render *"hand-
trimmed; retake at any level to discard the trim"* rather than silently guessing
balanced). `act.tighten` already re-resolves the whole segment
(`act.py:922-934`) and so correctly resets `level`; nothing else needs
touching.

Deliberately NOT eliminated: `level` cannot be derived on read (the variant a
cut came from is history, not a function of the current span), so it is a
genuine stored fact whose value must be *invalidated* on write. That is the
right shape for a non-derivable field, and stating the distinction is the point:
**derive what is derivable; invalidate what is not.**

### 2.4 `feel.py` gets real word counts without losing purity

`feel` must stay pure — no DB, safe to call every turn (`feel.py:11-14`), and it
is called five times per gate evaluation (`observe.py:502`, `:1210`, `:1552`,
`:1636`, plus `predict`). So the counts are threaded in:

```python
def simulate(
    timeline: List[dict],
    meta_by_ref: Optional[Dict[str, dict]] = None,
    descriptors: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    words_by_seg: Optional[Dict[str, int]] = None,
) -> FeelReport:
    """...
    ``words_by_seg`` maps seg_id -> the SPOKEN word count over that segment's
    span (l3.spoken.spoken_for_span(...).word_count()). Passed in because feel
    is PURE and the caller (observe.build_context / observe's senses) already
    does the DB reads once. Omitted -> pace_wps is None, NOT 0.0: "we did not
    measure the pace" and "this cut has no speech" are different facts and must
    not render identically (§4).
    """
```

and `pace_wps: Optional[float]`, with every consumer updated:

- `_score_energy` (`feel.py:201-208`) — a `None` pace means the pace term is
  ABSENT, so energy falls back to the shortness term alone. That is already the
  code path for a non-speech cut (`feel.py:207-208`); the difference is that it
  must no longer be reachable by *accident*.
- `FeelReport.avg_pace` (`feel.py:97-100`) — skip `None`s (already skips
  `<= 0`), and return `None` rather than `0.0` when nothing was measured, so
  `narrate()`'s `"~N words/s when talking"` (`feel.py:110`) is omitted instead
  of claiming 0.
- `FeelReport.to_dict()` (`feel.py:130-142`) — carry the `None` through.
- Add a `words_status` to `FeelReport` (§4) so `read_state["feel_detail"]` can
  say the pace was not measured.

Every caller of `feel.simulate` in `observe.py` passes `words_by_seg`. Build it
once per sense from the timeline via `spoken.spoken_for_span`; because the
underlying reads are `lru_cache`d per file, the five `simulate` calls in one
gate evaluation share the reads.

**The signal that changes behaviour.** `_low_energy_runs` (`feel.py:219-224`)
currently ranks cuts by an energy built from label-word density. With real word
counts, the mirror's `dead-air` flag (`observe.py:1670-1671`) and `diagnose`'s
low-energy finding (`observe.py:1222-1224`) start pointing at cuts that are
actually slow. **This will change which cuts are flagged on existing threads,
and that is the intended effect, not a regression** — but it means §6 must
include a before/after comparison on the real fixture rather than only
assertions, so the change is inspected rather than assumed.

---

## 3. §3 — RECONCILE THE SENSES THAT DISAGREE

### 3.1 One program clock (LATENT class — honest labelling)

Four definitions of program duration exist:

| definition | site | rule |
|---|---|---|
| **AUTHORITATIVE (the render)** | `layers.spine_spans` (`layers.py:388-396`) | `max(0, out-in)` per segment, accumulated |
| `observe._seg_ms` | `observe.py:1072-1073` | `max(0, out-in)` — agrees |
| `feel.simulate.total_ms` | `feel.py:162-165` | `max(0, …)` and **skips** `dur <= 0` — agrees numerically, differs in POSITION side effects (§3.2) |
| `review.total_ms` | `observe.py:1514`, `:1538` | `out-in`, **unclamped** — disagrees when `out < in` |

and four independent program-time accumulators, three of which are unclamped:
`read_state` (`observe.py:538`, `:610`), `review` (`observe.py:1514`, `:1538`),
`audio_state` (`observe.py:988`, `:997`), `affordances` (`observe.py:1802`,
`:1816`).

**Per §0.4 this cannot be provoked at HEAD.** So:

- **Change:** one helper, `layers.prog_span_ms(seg) -> int` (or reuse
  `observe._seg_ms`, moved to `layers.py` beside `spine_spans` so the render's
  own definition is the one everyone imports), used by all four accumulators and
  by `review`. Delete the local arithmetic at `observe.py:538`, `:988`, `:1514`,
  `:1802`.
- **Do NOT add a gate, a flag, or a warning for this.** There is no failing
  input to detect. Adding one would be a check that can never fire — the exact
  pattern this repo has shipped four times.
- **Add an INVARIANT assertion instead** (§6 V6): a unit test asserting that all
  four accumulators and `layers.spine_spans` agree on a timeline that
  *deliberately contains* an out-of-band segment constructed directly in the
  test. That test is honest: it does not claim the input is reachable, it claims
  the definitions are one.

### 3.2 One cut position (LATENT, same root)

`feel.simulate` renumbers positions after skipping degenerate segments
(`pos=len(cuts) + 1`, `feel.py:172`), while `read_state` (`observe.py:553`),
`review` (`observe.py:1524`), `mirror` (`observe.py:1644`) and `affordances`
(`observe.py:1804`) all use the raw timeline index `i + 1`. The mirror mixes
both in ONE row:

```1637:1651:backend/app/services/l3/observe.py
    joins = {j["to_pos"]: j for j in feel.join_continuity(report.cuts, report.descriptors)}
    low_energy_positions = {
        p for lo, hi in feel._low_energy_runs(report.cuts) for p in range(lo, hi + 1)
    }

    segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(timeline):
        pos = i + 1
        ...
        j = joins.get(pos)
```

`joins` is keyed by feel positions; `pos` is a timeline index. They coincide only
while no segment is skipped. `diagnose`'s anchors — *"cuts 4-6"*
(`observe.py:1213`, `:1220`, `:1223`) — are feel positions presented to the
brain as cut numbers it will look up in `read_state`.

**Change:** `feel.simulate` stops skipping. Emit a `CutFeel` for every timeline
segment, with `pos = i + 1` matching the timeline index, and let `dur_ms = 0`
carry the degeneracy as data. Then every "cuts N-M" anchor in the system refers
to the same N. `_fast_runs`/`_lingers`/`_low_energy_runs`/`_same_speaker_runs`
need a `dur_ms > 0` guard where they would otherwise treat a zero-length cut as
a fast cut (`feel.py:212`) — which is the one real behaviour change and must be
stated in the docstring. Same latency caveat as §3.1: assert the invariant
(§6 V6), do not add a runtime check.

### 3.3 One program median loudness (REACHABLE — fix properly)

Two medians, two populations, both presented to the brain:

```523:533:backend/app/services/l3/observe.py
    lufs_by_file = {
        fid: af["integrated_lufs"] for fid, af in ctx.audio_features.items()
        if af.get("integrated_lufs") is not None
    }
    _timeline_lufs = sorted(
        lufs_by_file[seg["file_id"]] for seg in timeline
        if seg.get("file_id") in lufs_by_file
    )
    lufs_median = (
        _timeline_lufs[len(_timeline_lufs) // 2] if _timeline_lufs else None
    )
```

→ `read_state.cuts[].loudness_rel` (`observe.py:570-572`). Population:
main-line segment files only; **one entry per SEGMENT**, so a file placed five
times counts five times and skews its own median; **no gain applied**.

```1427:1435:backend/app/services/l3/observe.py
    levels: List[Tuple[Any, float]] = []
    for a in resolved.audio_layers:
        af = audio_features.get(a.source_file_id)
        if af and af.get("integrated_lufs") is not None and a.gain_db > -60:
            levels.append((a, float(af["integrated_lufs"]) + a.gain_db))
    if len(levels) < 2:
        return []
    values = sorted(v for _, v in levels)
    median = values[len(values) // 2]
```

→ `_loudness_imbalance_flags`, which reaches the brain as a `review` `craft`
flag (`observe.py:1544`) and through `_stage_flags` (`tools.py:1110`).
Population: ALL resolved audio layers including music beds and replaced audio;
gain APPLIED; near-silent layers excluded.

**Reachable on any document with a music bed**, which is most of them. The brain
reads `loudness_rel: -3.2` on a cut in `read_state`, then a flag saying that same
layer sits `+4.1dB` off the program median, and both are true of different
medians. There is no way to reconcile them from the outside.

**Change:** one function, one definition:

```python
def program_loudness(resolved: "layers.ResolvedTimeline",
                     audio_features: Dict[str, dict]) -> ProgramLoudness:
    """The program's loudness reference, computed ONCE from the RESOLVED
    timeline: per audible audio layer, its source integrated LUFS + its gain,
    median over layers (not over segments -- a file placed five times is five
    layers of the program, and that IS the mix the listener hears, whereas
    counting a file five times in a per-FILE median was neither).

    Returns {median_lufs, per_layer, status} with status=STATUS_UNAVAILABLE and
    median_lufs=None when fewer than 2 layers carry a measured LUFS -- so
    "balanced" and "unmeasurable" are distinguishable (§4). Both read_state's
    per-cut loudness_rel and diagnose/review's imbalance flags project THIS.
    """
```

`read_state.cuts[].loudness_rel` becomes that layer's own value minus
`median_lufs`, or the key is OMITTED with a `loudness_status` when
`unavailable`. `_loudness_imbalance_flags` keeps its `_LOUDNESS_IMBALANCE_LUFS`
threshold (`observe.py:1407`) and reads the same median.

Note the deliberate choice: the resolved-layer population WINS over the
main-line-segment population. It is the mix the listener hears, it includes the
beds the brain places, and `read_state` already resolves the timeline 130 lines
later anyway (`observe.py:646-647`) so the data is free. **Preserving both
populations "because each is right for its caller" is the disagreement, not a
resolution of it.**

### 3.4 The `resolved` snapshot — NOT stale, ABSENT (REACHABLE, and worse than described)

```508:514:backend/app/services/l3/observe.py
    # Grade summaries (color_grading.plan.md SS10 "explain the grade") read
    # from the LAST resolve's snapshot -- same staleness as everything else
    # read_state reports, and avoids a full re-resolve just to describe it.
    grade_by_layer_id = {
        v.get("layer_id"): v.get("grade")
        for v in ((document.get("resolved") or {}).get("video_layers") or [])
    }
```

The comment says "same staleness as everything else". **It is not staleness.**
`act._clone` DELETES the key on every verb:

```70:72:backend/app/services/l3/act.py
    if document.get("layout_regions"):
        doc["layout_regions"] = [dict(r) for r in document["layout_regions"]]
    doc.pop("resolved", None)
    return doc
```

and `document["resolved"]` is only rewritten by `observe.resolve_doc`
(`observe.py:335`), which is called **once per user turn, after the loop
finishes**, at `converse.py:700`. Therefore: **from the first edit verb of a
turn onward, `grade_by_layer_id` is `{}` and `read_state` silently omits
`cut["grade"]` for every cut, on every call, for the rest of the turn.** The
brain cannot see its own grades while it is editing. It is not reading an old
value; it is reading no value, and nothing tells it so.

Meanwhile the same function resolves the z-stack FRESH 130 lines later, with an
explanatory comment about exactly this hazard:

```641:647:backend/app/services/l3/observe.py
    # edso_pacing_audit_timing.plan.md item 2: report the z-stack (coverage
    # layers + layout), not just the spine `cuts` above, so this on-demand
    # look matches what the Program Map already shows -- resolved FRESH here
    # (not read off `document["resolved"]`, which is only as fresh as the
    # LAST full turn's resolve_doc, stale for a tool call mid-loop).
    try:
        resolved = layers.resolve(document, ctx.durations)
```

**Change:** move the fresh `layers.resolve` to the TOP of `read_state`, before
the `cuts` loop, and read grades off it. One resolve per `read_state` call
instead of one resolve plus one dead dict lookup. `resolve_doc`'s own resolve
passes `ctx.color_stats` and a `grade_lookup` (`observe.py:335-337`) that
`read_state`'s call omits (`observe.py:647`) — reconcile these deliberately: if
`read_state` needs grade summaries it needs the same `grade_lookup`, so factor
the resolve arguments into one helper both call, or accept and DOCUMENT that
`read_state` reports the resolver's default grade. **Do not leave two
`layers.resolve` call sites in the same module with different arguments and no
note about why.**

Delete the misleading comment at `observe.py:508-510`. A comment asserting a
property the code does not have is how this survived.

---

## 4. §4 — THE SENSE STATUS ENVELOPE

### 4.1 The class, stated once

> **A sense must never return the shape of an honest empty answer when it could
> not see. `observed` / `empty` / `unavailable` are three different answers and
> only two of them mean "there is nothing there".**

### 4.2 The false-genericity trap, named first

**The trap: add a `status` field that every caller ignores.** A `status` key on
nine sense payloads, a passing unit test per key, zero behaviour change — and
the field is a fifth "value set, nothing consumes it" in a codebase
`junk_span_suppression.plan.md` §2.3 already indicts for having three.

The bar: **§4 does not ship unless both consequences below ship with it.** They
are the reason the field exists.

### 4.3 Consequence A — the brain can tell absence from emptiness

The worst instance, verified:

```92:94:backend/app/services/l3/feel.py
    # having to thread it through separately. {} (not None) when the caller
    # passed nothing -- join_continuity then fails open to "cut" for every
    # non-contiguous seam (no false "seamless"/"jump-cut" without pixels).
```

```348:351:backend/app/services/l3/feel.py
            hA = _seam_phash(descriptors, prev.file_id, prev.src_out_ms, nearest_ms)
            hB = _seam_phash(descriptors, cur.file_id, cur.src_in_ms, nearest_ms)
            if hA is None or hB is None:
                kind = "cut"
```

(The brief cites `feel.py:89-94` — that is the comment; the code is
`feel.py:350-351`.) With no frame descriptors, every non-contiguous seam grades
`"cut"`, and `"this edit has no jump-cuts"` renders identically to `"no pixels
were loaded"`. `ctx.frame_descriptors` is `{}` whenever the lookup fails
(`observe.py:163-169`, a try/except that logs and continues) and for any file
with no `frame_descriptors` row — and `junk_span_suppression.plan.md` §2.4
records that descriptors exist for **114 files**, i.e. not all of them. So this
is provokable on real data, not hypothetically.

**Change.** `join_continuity`'s verdict gains a third value alongside
`seamless`/`jump-cut`/`cut`: `unknown`, with `pic_status` on the entry. The
render side must show it: `mirror`'s `join:` token (`observe.py:1694-1698`),
`read_state.cuts[].join` (`observe.py:597-608`), `diagnose`'s jump-cut finding
(`observe.py:1218-1221`), `review`'s continuity flags
(`observe.py:1553-1557`), and `narrate()` (`feel.py:122-126`). And the mirror
should carry ONE header line when the whole project has no pixels — *"joins
UNGRADED: no frame samples loaded for this project, so seams are unclassified"*
— rather than N per-seam `unknown` tokens, because the failure is per-project
not per-seam.

The same treatment for the other reachable cases (full list in §7):
`read_transcript` returning `{"text": "", "segments": []}` for both "no speech
placed" and "the transcript load failed" (`observe.py:1599`, downstream of
`footage_map._sentences_for_file`'s `return ()` at `footage_map.py:1406`);
`read_state`'s swallowed layer resolve (`observe.py:679-680`); `review`'s three
internal try/excepts (`observe.py:1519-1522`, `:1530-1537`, `:1540-1546`).

**Spec how it surfaces in the RENDERED TEXT, not just the dict.** This is
explicit because a status the brain never reads is the trap in §4.2 wearing a
dict key. Concretely:

- `observe.render_mirror` (`observe.py:1680-1707`) — a `MIRROR (…)` header line
  gains a suffix naming any sense that is `unavailable` this turn, e.g.
  `MIRROR (the edit as it now stands -- read before your next move; JOINS
  UNGRADED: no frame samples for this project):`.
- `read_state` — a top-level `"unavailable": ["grades: no resolved snapshot",
  "loudness: fewer than 2 measured layers"]` list, and the affected per-cut keys
  OMITTED rather than defaulted, so the brain sees a gap plus a reason instead
  of a plausible number.
- `converse._LOOP_SYSTEM` — one sentence in the senses paragraph
  (`converse.py:187-189`), descriptive in the house style: *a sense can report
  that it could not see something rather than that there was nothing there;
  treat "unavailable" as a gap in your own perception, not as an absence in the
  edit, and say so to the user rather than asserting the thing is absent.* This
  is the sentence that would have prevented the thread-`34bb9967`
  *"the music bed didn't fit"* claim about an asset never examined.

### 4.4 Consequence B — POLICY acts on `unavailable`

This is the half that makes the field load-bearing, and it is where §4 meets
the gate. Today:

```1041:1048:backend/app/services/l3/tools.py
    try:
        review_out = observe.review(working, ctx, user_ask=user_ask)
    except Exception:
        logger.exception("_gate_ctx: review failed (continuing without it)")
        review_out = None
    return _GateCtx(working=working, ctx=ctx, state=state, steps=steps,
                    user_ask=user_ask, advisory_skip=advisory_skip,
                    findings=findings, review_out=review_out)
```

```1075:1078:backend/app/services/l3/tools.py
    ask_flags = [f for f in (gc.review_out or {}).get("flags", [])
                 if f.get("category") == "ask"]
    if not ask_flags or gc.state["intent_surfaced"]:
        return None
```

`_stage_intent` is MANDATORY. A `review` exception empties it silently. **That
is the same bypass shape PART 1 was written to close, arriving via the DATA path
instead of the control path** — and `_gate_ctx`'s own docstring
(`tools.py:1008-1013`) says a mandatory stage may never depend on an
optimization for an advisory one, while the handler four lines below lets a
mandatory stage depend on a sense having succeeded.

The rule, and it belongs in `tools.py` next to the ladder so it is one place:

> **A MANDATORY gate stage may not be SATISFIED by a sense whose status is
> `unavailable`.** It must convert the silent pass into a stated uncertainty the
> brain has to answer.

`brain_gate_integrity.plan.md` §1.6 specifies the exact stage-side change
(the `_unavailable(gc.review_out)` branch in `_stage_intent`, the
`blocked_by_unavailable` trace field, and why it states uncertainty rather than
hard-blocking forever — a `review` outage must not make every edit
unfinishable). **This plan owns the status shape; that plan owns the policy.
They must land in one commit or the field is inert.** Ditto the coverage gate's
`_speech_text_was_observed` predicate (`brain_gate_integrity.plan.md` §2.3),
which must not be satisfiable by a `read_transcript` call that returned
`unavailable`.

Note the deliberate asymmetry with the house fail-open rule: everywhere else in
this codebase, fail-open means "continue as if there was nothing". For a
MANDATORY invariant it must mean "say you could not check". Both are fail-OPEN
in the sense that neither aborts the edit; they differ in what the brain is
told, which is the entire point.

### 4.5 The shape, once

In a new `l3/sense.py` (or at the top of `observe.py` — but a separate module
keeps `spoken.py`, `observe.py` and `tools.py` from each defining their own):

```python
STATUS_OBSERVED = "observed"        # the sense ran and there is something there
STATUS_EMPTY = "empty"             # the sense ran and there is genuinely nothing
STATUS_UNAVAILABLE = "unavailable" # the sense could NOT run -- `reason` says why


def unavailable(reason: str) -> Dict[str, Any]:
    """The ONLY value a fail-open handler may return in place of a payload. A
    handler that returns the empty PAYLOAD shape instead is the bug this module
    exists to make unwritable: it makes "nothing is there" and "I could not
    look" the same answer, for the brain AND for gate policy."""


def is_unavailable(payload: Any) -> bool: ...
def why(payload: Any) -> str: ...
```

Every sense payload gains a top-level `"status"` and, when relevant, a per-field
`"<field>_status"`. Additive: existing keys keep their names and types, so every
current consumer and every existing test keeps working — which is exactly why
§4.2's trap is so easy to fall into and why §4.3/§4.4 are non-optional.

### 4.6 The complete enumeration of indistinguishable sites

All verified against HEAD. A site qualifies when the handler's error return is
**shape-identical** to an honest empty result, so no consumer — brain or policy
— can tell them apart. Grouped by chain, because the fix is per-chain: the
status is attached where the payload is BUILT, and every site in a chain then
carries it.

**Chain 1 — context loaders (`observe.build_context`, once per turn).** These are
not sense payloads, which is why they are easy to miss, and one of them weakens
a MANDATORY gate stage without raising anything downstream.

| site | handler | error shape ≡ honest shape | consumer that cannot tell |
|---|---|---|---|
| run resolve | `observe.py:126-127` | `eff_run = None` ≡ no run for these files | the whole map + seam reads |
| durations | `observe.py:131-135` | `{}` ≡ files of unknown duration | **`validate`'s over-span check is SKIPPED → MANDATORY `structural` silently checks less.** Owned by `brain_gate_integrity.plan.md` §1.8.5 V12 |
| color_stats | `observe.py:136-141` | `{}` ≡ never measured | grade reads |
| audio_assets | `observe.py:142-146` | `[]` ≡ **no audio uploaded at all** | `audio_state`, and the coverage gate's `audio` subject (`brain_gate_integrity.plan.md` §2.3) — a load failure would make the coverage rule report "nothing to look at" |
| audio_features | `observe.py:147-155` | `{}` ≡ never analysed | beat grid, loudness, `_hint_beatsync` |
| scene_taxonomy | `observe.py:156-162` | `None` ≡ no taxonomy for this run | scene reads |
| frame_descriptors | `observe.py:163-169` | `{}` ≡ no rows for these files | every pHash seam verdict this turn |
| audio-asset transcripts | `observe.py:268-269` | asset without `transcript` ≡ music/SFX with no speech | the voiceover block in the system prompt — a VO script the brain should sync to becomes invisible |

**Chain 2 — pHash / continuity.** `feel.py:350-351` folds a missing pHash into
`kind: "cut"`, identical to an honest content-change cut; `feel.py:94`/`:185`/
`:332` default descriptors to `{}`; `frame_descriptors.py:122-138` writes no row
on decode failure; `frame_descriptors.py:200-212` returns `None` beyond
`_JOIN_NEAREST_MS`. **The `"cut"` fallback itself STAYS** — it is a documented
anti-false-alarm rule and inventing `jump-cut`/`seamless` without pixels is
worse. What changes is that the seam carries `pic_status: "unavailable"` with a
reason, so `read_state`/`review`/the mirror can say "seam not graded" instead of
implying a clean cut.

**Chain 3 — transcript / speech text.** `footage_map.py:1404-1408` (`()` on DB
error ≡ empty sentence array), `:1383-1384` (`{}` on corrupt JSON),
`:1520-1521` (`""` ≡ silent span), `observe.py:1592-1599`
(`{"text": "", "segments": []}` ≡ no speech placed),
`cutrecord_map.py:500-504` (`()` ≡ never transcribed), `:544-546`
(`(0, False, False)` ≡ no overlapping words), `:577-579` (all-False boundary
flags ≡ no words), `:522-524` (`is_musical=False` ≡ analysed non-musical).
**This chain is the highest-severity one in the set**: a DB blip makes a talking
head look silent to every sense simultaneously, and the mirror then renders a
speech cut as if it carried no words. §1.3's `Spoken` dataclass carries
`status`/`reason` for the whole chain in one place, which is the structural
reason §1 is sequenced before §4's consumers.

**Chain 4 — `review` / `read_state` internals.**
`observe.py:1521-1522` (`words = []`), `:1532-1533` (head/tail flags absent ≡
clean edges), `:1536-1537` (off-cam flag absent ≡ speaker on camera),
`:1545-1546` (`layers.resolve` throw → overlay-fit, audio-gap AND
loudness-imbalance flags all absent ≡ a timeline with none of those problems),
`:679-680` (`video_stack`/`audio_layers`/`audio_gaps` keys omitted ≡ no overlays
and no gaps), `:702-703` (`word_offsets` omitted). `:1545-1546` is the one to
weight: a single exception silently removes three whole flag families, and
`review`'s flag list is what the ADVISORY `flags` stage projects.

**Chain 5 — gate inputs.** `tools.py:1038-1040` (`findings = []` ≡ zero
editorial findings → MANDATORY `length`), `tools.py:1043-1045`
(`review_out = None` ≡ review with zero flags → MANDATORY `intent`),
`tools.py:1187-1193` (stage function raises ≡ stage clean → **six** MANDATORY
stages), `tools.py:952-954` (`_plan_conformance` → `None` ≡ nothing to conform
to → MANDATORY `conformance` and, downstream, `surface`), `tools.py:915-916` and
`:888-890` (hints dropped). **All of Chain 5 is owned by
`brain_gate_integrity.plan.md` §1.6 and §1.8**, not by this plan: this plan
supplies the status, that plan supplies the policy. Listed here so the two
enumerations are known to be the same set viewed from both ends.

**Chain 6 — prompt / index assembly.** `converse.py:584-585` drops the ENTIRE
`BEAT INDEX` section from the system prompt on any `assemble_map` failure —
the brain begins the turn with no material listed and no way to know material
exists. `footage_map.py:1333-1335` drops the `CAST` line. `observe.py:388-390`
returns `[]` seam points ≡ no run. `observe.py:295-300` returns empty curves for
`inspect_cut` ≡ never analysed.

Chain 6 has the precedent that settles the design argument, twelve lines above
the site itself, in the same function:

```576:579:backend/app/services/l3/converse.py
        if len(text) > _INDEX_CHAR_CAP:
            text = (text[:_INDEX_CHAR_CAP] +
                    "\n[TRUNCATED: the beat index exceeded its budget here -- beats "
                    "after this point are MISSING above.]")
```

When the index is truncated the code **tells the brain, in the prompt, that
material is missing**. When the index fails to build it tells it nothing. The
target state for every site in this section is the one this line already
demonstrates: absence is stated in the payload the brain reads, not left for it
to infer. Cite this line in the change so the reviewer sees the pattern is
established, not invented here.

**Deliberate keeps (do NOT convert these).** `feel.join_continuity`'s `"cut"`
verdict without pixels; `nearest_phash`'s distance gate; `speech_words_in_span`
and `speech_boundary_flags` on the genuinely-no-words path; `snap_span_to_seams`
/ `snap_to_beats` no-op with nothing to snap to; `cut_dicts_for_files` and
`load_descriptors` omitting absent files; `validate` returning `[]` on a clean
document; `_dispatch`'s `EditReject` vs `Exception` split (`tools.py:747-754`,
already distinguishable — `applied: false` + reason vs `error`); `converse`'s
top-level catch (`converse.py:693-695`, already user-visible). Each is either a
true negative or an honest "not present", and converting them would manufacture
the false alarms §4.2 warns about.

**Totals.** 32 indistinguishable sites across six chains; 9 deliberate keeps.
Of the 32, **9 are fixed by the `Spoken`/`program_loudness`/resolve work in
§§1–3 as a side effect** (the chain gets a status because the payload does), 17
are the direct subject of §4, and 6 are Chain 5 and land with
`brain_gate_integrity.plan.md`. A change that adds `status` to fewer than the 17
has not finished §4; a change that adds it to the 9 keeps has misread it.

---

## 5. §5 — MIGRATION AND BACK-COMPAT

**No database migration.** Every change is to derived read-side projections, to
the in-memory segment shape, and to jsonb documents that already tolerate
unknown/absent keys.

**In-flight documents and stored `content` fields.** `edit_documents` rows are
versioned jsonb (`store.save_document`), and old versions are immutable history
that must keep rendering. So:

1. **Never write `content` again** (§2.2 deletes the three writers).
2. **Never read `content` again.** Every reader is migrated in the same commit
   (`observe.py:540`, `feel.py:166`, `arrange.py:291`, plus any
   `arrange._label` caller). Grep for `["content"]` and `.get("content")`
   across `backend/app` and `frontend/src` before landing — the frontend must
   not be rendering it either.
3. **Leave existing `content` values in stored documents untouched.** Do not
   write a backfill and do not strip the key. It becomes inert data in
   historical rows. A stripping migration would rewrite immutable history to
   remove a field nothing reads — all risk, no benefit.
4. **A document saved before this change and loaded after it behaves
   identically**, because the new readers derive from `file_id` + `in_ms` +
   `out_ms` + `ref`, all of which every stored segment already carries
   (`act.py:91-112` has written them since placement existed). This is the
   property that makes elimination cheaper than refresh: there is nothing to
   migrate because the derivation's inputs were always stored.

**One genuine incompatibility to handle:** a stored segment placed by
`place_span` with no `ref` and no `file_id`-resolvable transcript will now render
`text: ""` where it previously rendered the caller's note. If that note matters,
it is in `seg["rationale"]` (`act.py:173`) — surface `rationale` in
`read_state.cuts[]` rather than resurrecting `content`.

**Renamed moment key (§1.4d).** `moment["said_text"]` → `moment["context_text"]`
+ `moment["played_text"]`. The moment tree is CACHED
(`footage_map.get_trees`, and `_sentences_for_file`'s `lru_cache`) but is
rebuilt from `cut_records` rows, not persisted as a tree — verify this before
landing (`footage_map.build_clip_tree` is called per request path). If any cache
persists the tree across a deploy, the reader must tolerate both keys for one
release. **Say which it is in the implementing PR; do not assume.**

**Fail-open direction preserved.** Every handler that today logs and continues
still logs and continues. The only change is WHAT it returns
(`sense.unavailable(reason)` instead of the empty payload shape), so no error
path becomes a raise and no edit becomes unfinishable. The one place where the
consequence tightens is a MANDATORY gate stage (§4.4), and it tightens into a
question, not a block.

---

## 6. §6 — VERIFICATION, WITH REVERT SIGNATURES

**Every fixture below must be observed RED against HEAD before the change
lands.** Four features have shipped in this repo with passing tests against code
that could never execute (`vcut/resolve.py:512`'s dead-air gate unreachable
because `has_peak` defaults True at `resolve.py:173`; the `junk` filter because
`vcut/store.py:135` hardcodes `junk=False`; the `SND:silence` render because
`vcut/store.py:81` hardcodes `natural_sound=True`; and the gate ladder's
last-turn path). A test that has never been seen failing against the old code
proves nothing.

**Shared fixture.** Reuse the capture that `junk_span_suppression.plan.md` §3.5
specifies — ingest run `24894c8a-9daa-4aa3-aea2-1f3b60c93478`, edit thread
`34bb9967-97aa-4818-9f21-01145156a9bb` — extended with (a) the `dialogue_segments`
sentence rows and (b) the `transcripts` word rows for the three files, so §1's
sentence-vs-word divergence is measurable on real data. **Do not capture a
second fixture.**

**V1 — the divergence exists on real data, then does not.** Over the fixture's
placed speech segments, assert against HEAD that at least one segment has
`_said_text_for_span(...) != " ".join(w["text"] for w in _word_offsets_for_seg(...))`,
and record the actual pair in the test as a comment. **Then apply a `trim` (via
`act.trim`, ±800 ms) to that segment before asserting** — because
`act._snap_cut_to_sentences` (`act.py:116-131`) snaps freshly placed speech cuts
to sentence edges, so an untrimmed fixture may show NO divergence and the test
would pass vacuously against HEAD. After the change, assert
`review["items"][i]["played_text"] == " ".join(w["text"] for w in
review["items"][i]["words"])` for **every** item, and that the same string
appears in `read_transcript`'s per-segment text and (bounded) in the mirror's
`gist`.

*Revert signature:* restore `_said_text_for_span` at the `review` site and the
equality assertion fails on the trimmed segment. Restore it at the mirror site
only (the partial fix the brief warns about) and the "same string in the mirror"
assertion fails — **this is the assertion that forbids the partial landing.**

**V2 — `read_state.cuts[].text` stops surviving a trim.** Place a speech cut,
snapshot `read_state()["cuts"][0]["text"]`, `act.trim(delta_in_ms=+900)`,
re-read. Assert the two differ, and that the new value equals the played words
over the new span. Additionally assert the new `label` key holds the VLM gist
and is UNCHANGED by the trim (it describes the moment, not the span — that is
correct and must not be "fixed").

*Revert signature:* HEAD returns an identical `text` before and after, which is
the exact observation that made the brain trim twice and burn four turns.
Restoring `seg["content"]` as the source makes the inequality assertion fail.
Note this test also fails against the *band-aid* fix (refreshing `content` in
`trim` from `rc.label`), because a re-derived label is still identical after a
900 ms trim — so it distinguishes the real fix from the tempting one.

**V3 — `pace_wps` measures speech.** A synthetic timeline with two speech
segments of equal duration, one covering 30 transcript words and one covering 3,
both carrying the SAME `content` label. Assert their `pace_wps` differ by ~10×.
Then, on the fixture, assert `feel.simulate(...).avg_pace` differs from HEAD's
value and that `read_state()["feel"]` no longer contains a `words/s` claim when
`words_by_seg` is omitted.

*Revert signature:* HEAD reports IDENTICAL `pace_wps` for both segments (same
label, same duration), so the ~10× assertion fails. This is the assertion that
proves `pace_wps` was measuring nothing.

**V4 — `retime` and `heal` stop inflating the word count.** (a) A speech cut
re-sliced by `act.retime(pace="faster")` into N slices: assert the sum of
`CutFeel.words` across the slices equals the spoken words in the union of the
kept spans, not N× the label. (b) Two same-source adjacent segments run through
`arrange.heal_adjacent_cuts`: assert the merged segment's word count equals the
words over the merged span. *Revert signature:* HEAD's (a) multiplies the label
count by N (`act.py:1026` copies the dict) and (b) concatenates two labels
(`arrange.py:291-293`), so both sums come out wrong by a factor. **Both of these
are reachable by a single routine verb call, which is what makes them proof
rather than illustration.**

**V5 — reachability of `unavailable` on a real degraded path.** THE
anti-dead-code assertion for §4. Three sub-cases, each provoked without
mocking a raise, because a mocked raise proves the handler works and not that
the path exists:

- **(a) no pixels.** Load the fixture context, then remove the
  `frame_descriptors` entries for its files (a file with no descriptor row is a
  real state — `junk_span_suppression.plan.md` §2.4 records descriptors for 114
  files, not all). Assert `feel.join_continuity(...)` returns `kind == "unknown"`
  with `pic_status == STATUS_UNAVAILABLE` for the non-contiguous seams, that
  `mirror_text` contains the ungraded-joins notice, and that `diagnose` emits
  NO jump-cut finding **and** `read_state["unavailable"]` names it.
- **(b) no transcript.** A fixture file with no `dialogue_segments`/`transcripts`
  row. Assert `read_transcript` returns `status == STATUS_UNAVAILABLE` with a
  reason, distinct from a document with no speech segments placed, which must
  return `status == STATUS_EMPTY`. **Assert the two payloads are not equal** —
  at HEAD they are byte-identical (`{"text": "", "segments": []}`), which is the
  defect.
- **(c) no measured loudness.** A context whose `audio_features` has fewer than
  two `integrated_lufs` values. Assert `read_state.cuts[]` OMITS `loudness_rel`
  and `read_state["unavailable"]` names loudness, rather than reporting `0.0`
  differences.

*Revert signature (all three):* revert the handler to returning the empty
payload shape and each "the two are not equal" / "the notice is present"
assertion fails. **These are the assertions that make §4 more than a field
nobody reads**, and (b) is the one that would have caught the observed
18-reasoning-step blindness to an empty `read_transcript`.

**V6 — one definition per concept, asserted as an invariant.** Table-driven,
over a timeline constructed IN THE TEST to include an out-of-band segment
(§0.4/§3.1 — deliberately not claimed to be reachable through any verb):
`layers.spine_spans(...)[1]`, `feel.simulate(...).total_ms`,
`observe._seg_span_total(...)`, `review(...)["total_ms"]`, and the final `prog`
of `read_state` / `audio_state` / `affordances` all agree. Plus: `feel`'s
`CutFeel.pos` equals `timeline` index + 1 for every segment, and every
`diagnose` anchor's cut number resolves to a `read_state.cuts[].pos`.
*Revert signature:* restore any one local `out - in` (e.g. `observe.py:1514`) and
the equality fails. **State in the test docstring that the input is
constructed, not reachable** — an honest latent-class invariant, not a fake bug
report.

**V7 — one program median loudness.** On a fixture with a music bed placed:
assert `read_state.cuts[i]["loudness_rel"]` and the corresponding
`_loudness_imbalance_flags` entry are computed against the same median, by
asserting `read_state.cuts[i]["loudness_rel"]` equals that layer's
`loudness_lufs + gain_db - median` from the same `program_loudness` call.
*Revert signature:* at HEAD the two medians differ whenever a bed is present
(different population, gain applied on one side only), so the equality fails.
**RED against HEAD on any document with a bed**, which is the common case.

**V8 — grades are visible mid-loop.** A session-level test: place a cut, call
`set_grade`, then call `read_state` — all within one `run_edit_loop`. Assert
`read_state.cuts[0]["grade"]` is present. *Revert signature:* at HEAD
`act._clone` has popped `resolved` (`act.py:71`) and `resolve_doc` has not run
(it runs at `converse.py:700`, after the loop), so `grade_by_layer_id` is `{}`
and the key is absent. **This test is RED against HEAD for every grade, on every
turn, which is why it belongs at session level and not as a unit test of
`read_state`** — a unit test can hand `read_state` a document with `resolved`
present and pass while the live path never has one.

**V9 — the Beat Index decision is deliberate, asserted on the rendered string.**
Assert `footage_map.assemble_map(...)["text"]` for a fixture moment whose
containing sentence extends beyond the beat: the line contains the context
sentence AND a distinct played-words token, and the moment dict has both
`context_text` and `played_text`. *Revert signature:* collapsing them back to a
single `said_text` fails the two-keys assertion. This is the one assertion that
records §1.4(d) as a decision rather than an oversight, and it asserts on the
string the model actually receives, not on a dict.

**V8b — a PENDING grade is visible mid-loop and labelled as pending.** Same
setup as V8, extended for §7c: assert `read_state.cuts[0]["grade"]` is present
AND `cuts[0]["grade_status"] == "pending"` before the grade job has run, and
that it reads `"measured"` on a fixture whose `resolved_grades` row exists.
*Revert signature:* an implementation that only fixes the `resolved`-snapshot
half of §3.4 (V8) still shows NO grade here, because `layers.resolve` reads
`grade_lookup` alone (`layers.py:619`) and the job is enqueued after the loop
(`converse.py:704-705`). **V8 passes and V8b fails for that partial fix, which
is the only way to tell the two apart.**

**V11 — `inspect_cut(seg_id=…)` follows the trim.** Place a cut, assert
`inspect_cut(seg_id=…)["span_ms"]` equals its duration; `act.trim(delta_in_ms=
+500)`; assert `span_ms` shrank by 500 and `cut_span_ms` did not. Then assert
`inspect_cut(ref=…)["span_ms"]` is UNCHANGED across the trim — the material
question has a material answer. *Revert signature:* at HEAD both calls return
the full cut's span, so the "shrank by 500" assertion fails; an over-correction
that also changes the `ref` path fails the third assertion.

**V12 — `audio_source` reflects the mechanism that actually routes audio.**
Place a cut, assert `read_state.cuts[0]["audio_source"] == "own"`; call
`act.replace_audio` pointing at a second file; assert `audio_source` now names
the replacement and that `layers.resolve`'s corresponding `AudioLayer`
`source_file_id` equals the same file. Separately, place a bed op with
`audio_kind="replace"` over an untouched cut and assert that cut reports
`replaced_by_bed: true` while `audio_source` stays `"own"`. *Revert signature:*
at HEAD the first pair disagrees (`read_state` says `"own"`, the resolve says
another file) and the second case reports `audio_source: "replaced"` for a cut
whose own audio was never rerouted — one token covering two facts, which is the
`≡` this test breaks.

**V13 — `channel` and `axis` are separate fields with separate vocabularies.**
Place the same material twice, once by `ref` and once by `place_span`. Assert
both cuts carry an `axis` in `{speech, any}`; assert the ref-placed cut carries
`channel` in `{said, done, shown}`; assert the span-placed cut OMITS `channel`
and carries a status saying there is no map row for it. *Revert signature:* at
HEAD the span-placed cut reports `channel: "any"` — a token from the wrong
vocabulary in the field named after the other one — so the "omits `channel`"
assertion fails.

**V10 — no reader of `content` survives.** A source-level guard, in the spirit of
`brain_accountability_architecture.plan.md` PART 5's
`test_no_raw_text_block_injection_in_the_loop`: grep `backend/app/services/l3/`
and `backend/app/routers/` for `["content"]` / `.get("content")` on a timeline
segment and fail if any remain. *Revert signature:* it fails the moment someone
reintroduces the cache. This is what stops §2 from being undone by a future PR
that "just needs a label on the segment".

**Run order.** V1, V3, V5(b), V7, V8, V8b, V11, V12 and V13 must be captured and
observed FAILING against `HEAD` before implementation begins. V2 and V4 must
additionally be observed failing against the *band-aid* implementation (refresh
`content` in `trim`), and V8b against the partial §3.4 (V8 only), because
passing against HEAD alone does not distinguish those fixes from the real ones.

---

## 7. §7 — THE COMPLETE ENUMERATION OF DRIFT INSTANCES

All verified against HEAD. **Class A** = a cached derivation allowed to drift.
**Class B** = two definitions of one concept. **Class C** = a field whose name
promises something other than its value. `fixed` / `deferred` / `left` is this
plan's disposition for each.

| # | class | instance | anchors | reachable at HEAD? | disposition |
|---|---|---|---|---|---|
| 1 | A+C | `seg["content"]`: written as the VLM label, never refreshed on `trim` | `act.py:98`, `act.py:172`, `act.py:595`; read `observe.py:540-542`, `feel.py:166-169` | **yes, observed** (thread `34bb9967`, ~4 turns burned) | **FIXED** §2.2 (eliminated) |
| 2 | A | `retime` speech re-slice copies `content` onto all N slices → N× word count | `act.py:1025-1027` | **yes**, one verb call | **FIXED** §2.2 |
| 3 | A | `heal_adjacent_cuts` CONCATENATES `content` on merge → ~2× word count | `arrange.py:291-293`; runs every turn `observe.py:317-318` and every human save `edit_threads.py:218` | **yes**, every turn | **FIXED** §2.2 |
| 4 | C | `pace_wps` documented "spoken words per second", computes label-words per second | `feel.py:78` vs `feel.py:166-169` | **yes, always** | **FIXED** §2.4 |
| 5 | C | `read_state.cuts[].text` is a VLM paraphrase, beside `review.played_text` which is transcript | `observe.py:565` vs `observe.py:1525` | **yes, always** | **FIXED** §2.2 (split into `text` + `label`) |
| 6 | C | `place_span` speech cut gets `content=""` → `pace_wps = 0.0` for real speech | `act.py:142`, `act.py:172`, `feel.py:169` | **yes** | **FIXED** §2.4 |
| 7 | A | `seg["level"]` not invalidated by `trim` → `affordances` offers retakes from a level the cut left | `act.py:106`, `arrange.py:184`, `act.py:595`; read `observe.py:1797-1800`, `_idx` `observe.py:1910-1914` | **yes** | **FIXED** §2.3 (invalidate, not derive) |
| 8 | B | `played_text` vs `words`: sentences-overlapping vs words-fully-inside, two tables, two filler policies | `observe.py:1516` / `footage_map.py:1517` / `footage_map.py:1401` vs `observe.py:1520` / `captions/timing.py:70-73` / `observe.py:482` | **yes, always**, same dict | **FIXED** §1.3–1.4a |
| 9 | B | a THIRD speech-presence predicate: word-overlap, used by the mirror | `cutrecord_map.py:540-543` vs the two above | **yes** | **FIXED** §1.3 (becomes a wrapper) |
| 10 | B | `_head_tail_flags` inlines the sentence-overlap pattern; dead-air deltas can go negative so the flag never fires on mid-sentence edges | `observe.py:1290-1294`, `:1331-1340` | **yes** | **FIXED** §1.4e |
| 11 | B | two filler vocabularies: `observe._FILLER_WORDS` hardcoded list vs L1's per-word `is_filler` | `observe.py:1262-1265`, `:1278-1280` vs `cutrecord_map.py:506`, `captions/timing.py:70` | **yes** | **FIXED** §1.4e (delete the list) |
| 12 | B | `review.played_text` is `""` for a `shown` cut carrying incidental speech while `read_transcript` and the mirror both report it | `observe.py:1515` vs `observe.py:1592`, `observe.py:1659-1662` | **yes**, the `incidental` flag case `observe.py:1664-1665` | **LEFT deliberately** — `review` reads the ASSEMBLED spine per axis, `read_transcript` is explicitly ungated (`observe.py:1577-1580`). §1 makes both word-accurate; the axis difference is a documented editorial distinction, not a drift. Named here so it is not "fixed" by accident. |
| 13 | B | program clock: 4 definitions (`layers.spine_spans` authoritative; `_seg_ms`; `feel` skip-degenerate; `review` unclamped) + 4 unclamped accumulators | `layers.py:393`, `observe.py:1072`, `feel.py:162-165`, `observe.py:538`/`988`/`1514`/`1802` | **NO** — `out <= in` rejected at every ingress (`edit_threads.py:86`, `act.py:587`, `act.py:160`, `act.py:89`) | **FIXED as a latent class** §3.1: unify to one helper + invariant test (V6). **No gate, no runtime check** — there is no input to detect. |
| 14 | B | cut POSITION: `feel` renumbers after skipping; every other sense uses the raw index; the mirror mixes both in one row | `feel.py:172` vs `observe.py:553`/`1524`/`1644`/`1804`; mirror `observe.py:1637` + `:1644` | **NO** (same root as #13) | **FIXED as a latent class** §3.2 (`feel` stops skipping) + V6 |
| 15 | B | `cut_count` twice in ONE payload: `len(timeline)` and `len(feel.cuts)` | `observe.py:626` vs `feel.py:133` via `observe.py:632` | **NO** (same root) | **FIXED** by §3.2 (they become equal by construction) |
| 16 | B | program median loudness: per-SEGMENT files, no gain, main line only vs per-LAYER incl. beds, gain applied | `observe.py:523-533` (→ `observe.py:570-572`) vs `observe.py:1427-1435` (→ `observe.py:1544`, `tools.py:1110`) | **YES**, any doc with a bed | **FIXED** §3.3 (one `program_loudness`) |
| 17 | B | grades read off `document["resolved"]` while the z-stack resolves fresh, in ONE function | `observe.py:511-514` vs `observe.py:646-647` | **YES** | **FIXED** §3.4 |
| 18 | A | `document["resolved"]` is DELETED by every verb and only rewritten once per turn after the loop → grades silently absent mid-loop, and the comment calls it "staleness" | `act.py:71`, `observe.py:335`, `converse.py:700`, comment `observe.py:508-510` | **YES, always after the first edit verb** | **FIXED** §3.4 (worse than the brief described) |
| 19 | B | two `layers.resolve` call sites in one module with different arguments (`ctx.color_stats` + `grade_lookup` vs neither) and no note why | `observe.py:335-337` vs `observe.py:647` | **YES** | **FIXED** §3.4 (reconcile or document) |
| 20 | C | mirror `gist` is a head+tail SPLICE rendered inside quotation marks | `observe.py:1613-1617`, rendered `observe.py:1703` | **YES** | **FIXED** §1.4c (render the elision; label the field) |
| 21 | C | `inspect_cut`'s min-max-normalized audio curve renders digital silence as "quiet but present" | `observe.py:798-799` | **YES** | **DEFERRED** → owned by `junk_span_suppression.plan.md` §3.3 item 3 (adds absolute `rms_dbfs_median`). Cross-referenced, not duplicated. |
| 22 | B | `_qual` renders both channels' `total_quality * 100` on incompatible ranges (video 0–1, speech 0–4.5) | `footage_map.py:953-955`, prompt `converse.py:309` | **YES** | **DEFERRED** → `junk_span_suppression.plan.md` §3.3 item 4 |
| 23 | B | `_overlap_ms` defined three times identically | `observe.py:1399-1400`, `footage_map.py:159-160`, `identity/bind_asd.py:50` | no drift today | **LEFT** — identical definitions, no consumer disagreement. Noted so a future change to one is recognised as a divergence. Not worth a refactor commit of its own. |
| 24 | B | `channels` computed three times with cosmetically different rules that currently agree | `observe.py:612-616`, `observe.py:1000-1004`, `observe.py:1851-1855` | equal today | **LEFT** — same values; consolidating is churn. Named so nobody "discovers" it as a bug. |
| 25 | A | `seg["rationale"]` / `seg["warnings"]` written once at placement, never updated | `act.py:99`, `act.py:103` | yes, but not read as a fact about the span | **LEFT** — `rationale` is a record of INTENT AT PLACEMENT, which is correctly immutable. §5 proposes surfacing it in `read_state` as such. |
| 26 | A | `seg["speed"]` / `pace_level` recorded but not applied to export length | `act.py:1006`, surfaced `observe.py:583-585` | yes | **LEFT — correctly handled already.** `read_state` states `"recorded; not yet applied to the export length"`. An acknowledged, surfaced drift is the pattern the rest of this plan is trying to reach, not an instance of the bug. |
| 27 | A+B | `inspect_cut(seg_id=…)` describes the MAP CUT's full span, never the PLACED segment's span — so after any `trim`/`tighten`/`retime`, the deepest look the brain has reports curves for a window the cut no longer occupies, with no indication | resolve `observe.py:893-897`, span `observe.py:904-908` (`meta["src_in_ms"]/["src_out_ms"]`), source of that span `footage_map.py:384-388` | **YES**, one `trim` | **FIXED** §2.2 — see 7a below. The `ref` path keeps the full-cut window (deliberate, documented at `footage_map.py:384-386`); only the `seg_id` path changes. |
| 28 | B+C | `read_state.cuts[].audio_source` is computed from `place_audio` ops tagged `audio_kind="replace"` and NEVER from `seg["audio_override"]`, which is the mechanism that actually reroutes audio (`layers.resolve` priority 1) — so after `replace_audio` the brain reads `audio_source: "own"` for a cut whose sound comes from another file | writer `act.py:485`, real reader `layers.py:635-638`, `read_state`'s unrelated definition `observe.py:516-522` + `observe.py:544-551` | **YES**, one `replace_audio` | **FIXED** §3.4 — see 7b below |
| 29 | A | `act.set_grade` writes `seg["grade"]`, which is consumed ONLY by the async grade job's resolver, enqueued AFTER the loop — so for the remainder of the turn the grade is invisible to every sense, by two independent mechanisms at once | writer `act.py:806-807`; `layers.resolve` reads only `grade_lookup` `layers.py:619`; the inline override is read at `grade/resolver.py:292` inside the job; job enqueued at `converse.py:704-705`, documented async at `edit_threads.py:162` | **YES, always** | **FIXED** §3.4 — see 7c below |
| 30 | B | a FIFTH unclamped program accumulator, in `_beat_grid` — musical onsets are mapped to program time with its own `prog += dur` | `observe.py:1058-1064` | **NO** (same ingress guards as #13) | **FIXED** by §3.1's single helper. Listed because §3.1's acceptance test must cover five accumulators, not four. |
| 31 | C | `read_state.cuts[].channel` is `meta["channel"] or seg["axis"]` — two vocabularies in one field: `said`/`done`/`shown` for a ref-placed cut, `speech`/`any` for a raw-span cut, and the `axis` collapse has already destroyed the `done` vs `shown` distinction | `observe.py:561`; `axis` written `act.py:96` and `act.py:170`; `channel` carried `footage_map.py:299` | **YES**, any `place_span` | **FIXED** §2.2 — see 7d below |

Counts: 31 instances. **25 fixed** by this plan, **2 deferred** to
`junk_span_suppression.plan.md` (cross-referenced, not duplicated), **4
deliberately left** with the reason stated. Of the 25 fixed, **18 are reachable
at HEAD** and 7 (#13, #14, #15, #30 and their sub-parts) are latent-class
unifications explicitly labelled as such.

### 7a. #27 — `inspect_cut`'s span, resolved

```904:908:backend/app/services/l3/observe.py
    try:
        s, e = int(meta.get("src_in_ms")), int(meta.get("src_out_ms"))
    except (TypeError, ValueError):
        return {"error": f"ref {ref!r} has no resolvable span"}
    span_ms = e - s
```

`meta` is the map row (`ctx.meta_by_ref[ref]`), and its `src_in_ms`/`src_out_ms`
are the FULL cut's span, chosen deliberately:

```384:388:backend/app/services/l3/footage_map.py
            # The TRUE cut span (not anchor["in_ms"]/["out_ms"], which is the
            # balanced-variant window) -- Mechanism B's inspect_cut sense
            # (observe.py) windows to the FULL cut, not one zoom level of it.
            "src_in_ms": cut.get("src_in_ms"),
            "src_out_ms": cut.get("src_out_ms"),
```

That is right for `inspect_cut(ref=…)`, which asks about *material*. It is wrong
for `inspect_cut(seg_id=…)`, which asks about *this cut in my edit*: the seg_id
is resolved to a ref at `observe.py:893-897` and the segment's own
`in_ms`/`out_ms` are then discarded. **Target state:** when the call resolved
through `seg_id`, window to `int(seg["in_ms"]), int(seg["out_ms"])` and report
both — `"span_ms"` for the placed window plus `"cut_span_ms"` for the full cut —
so the brain can see that it is looking at a trimmed portion. Keep the `ref`
path byte-identical. Reachability and revert signature: §6 V11.

### 7b. #28 — `audio_source`, resolved

`read_state`'s own comment states what its `replace_windows` list means:

```516:522:backend/app/services/l3/observe.py
    # Audio facts (audio_brain.plan.md 1c). `replace_windows`: program spans a
    # place_audio op tagged audio_kind="replace" covers -- a brain-authored fact
    # about ITS OWN past decision, not a code-enforced mute of the cut beneath.
    replace_windows = [
        (int(o.get("from_ms", 0)), int(o.get("to_ms", 0)))
        for o in ops if o.get("type") == "place_audio" and o.get("audio_kind") == "replace"
    ]
```

So `audio_source: "replaced"` means *"a bed the brain tagged as a replacement
plays over this cut"* — while the mechanism that actually changes where a cut's
audio comes from is `seg["audio_override"]`, read first by `layers.resolve`
(`layers.py:635-638`) and never read by `read_state`. One label, two unrelated
mechanisms, and the load-bearing one is invisible. **Target state:** since §3.4
already makes `read_state` derive its audio facts from the fresh resolve rather
than re-deriving them from `operations`, take `audio_source` from the resolved
`AudioLayer`'s actual source (`"own"` when the layer's file is the segment's own
file, `"replaced-source"` naming the other file when it is not, `"muted"`,
`"group-authoritative"`), and keep the bed-overlay fact as a SEPARATE key
(`"replaced_by_bed": true`) rather than collapsing both into one token. Two
facts, two fields. Asserted by §6 V12.

### 7c. #29 — `set_grade`'s invisible write, resolved

```805:807:backend/app/services/l3/act.py
    for t in targets:
        existing = Grade.from_dict(t.get("grade"))
        t["grade"] = compose(existing, delta, 1.0).to_dict()
```

```619:619:backend/app/services/l3/layers.py
            grade=grade_lookup.get(seg["seg_id"]) or identity_grade_json(WORKING_SPACE_V1),
```

`layers.resolve` reads grades ONLY from `grade_lookup`, which `resolve_doc`
fills from `fetch_latest_grades` (`observe.py:330-337`) — the persisted output
of the background job, which `converse` enqueues *after* the loop:

```704:705:backend/app/services/l3/converse.py
            from app.services.l3.grade.job import maybe_enqueue
            maybe_enqueue(thread_id, result.document)
```

The inline `seg["grade"]` IS honoured — but at `grade/resolver.py:292`, inside
that job. So the brain's own grade cannot appear in any sense before the turn
ends, and #18 independently guarantees `read_state` omits `cut["grade"]` for the
rest of the turn regardless. Two mechanisms, one symptom, and the brain's only
recourse today is to re-issue `set_grade` and hope.

**Target state (do not "fix" this by running the grade job inline — it is a
measurement job over rendered frames and belongs async).** §3.4's fresh resolve
gains a third source, ordered exactly as the resolver already orders them:
`grade_lookup.get(seg_id)` (persisted, measured) → else `seg.get("grade")` (the
brain's own pending override) → else identity. `read_state` then reports
`cut["grade"]` with a companion `grade_status` of `"measured"` or `"pending"`,
where `"pending"` states in words that the value is the brain's own request and
has not been through the grade job yet. This is §4's envelope applied to a
field, which is why it lands with §4 and not before it. Reachability and revert
signature: §6 V8b, which is specifically constructed to fail against the
partial version of §3.4 that fixes only the `resolved`-snapshot half.

### 7d. #31 — `channel`, resolved

```96:96:backend/app/services/l3/act.py
            "axis": "speech" if rc.channel == "said" else "any",
```

The map's three-way channel (`said` / `done` / `shown`) is collapsed to a
two-way axis at placement, and `read_state` then emits whichever of the two
vocabularies happens to be available for that segment:

```561:561:backend/app/services/l3/observe.py
            "channel": meta.get("channel") or seg.get("axis"),
```

A ref-placed cut reports `shown`; the same material placed by `place_span`
reports `any`. **Target state:** `read_state` reports the map channel as
`channel` (omitted, with a status, when there is no map row) and the placement
axis as a separate `axis` field, never one field holding either. `axis` keeps
its two tokens — it is a real editorial concept used by the spine logic — but it
stops impersonating `channel`. This is the same one-name-one-definition rule as
§2.2's `text` / `label` split and lands in the same edit. Asserted by §6 V13.

---

## 8. §8 — EXPLICITLY NOT IN THIS PLAN

- **The captions word reader.** `captions_timing.words_in_source_window`
  (`captions/timing.py:60-76`) keeps its fully-inside / drop-fillers policy. It
  is correct for captions (its own docstring says why) and captions are not a
  brain sense. §1.3 is careful to route `observe`'s use of it through
  `spoken.py` while leaving the captions pipeline alone.
- **Sentence snapping at placement.**
  `footage_map.snap_speech_spans_to_sentences` (`footage_map.py:1425`) and
  `act._snap_cut_to_sentences` (`act.py:116`) are an editorial rule about where
  cuts should START and END, not a perception read. Unchanged.
- **`_span_detail`'s padded window** (`footage_map.py:1535`, `_DETAIL_PAD_MS`)
  — intentionally wider than the span and documented as such.
- **Span usability grading, the `junk` producer, `SND:silence`, video duplicate
  detection, `_qual`'s scale, `inspect_cut`'s absolute dB, and the false
  `_PROVENANCE` guarantee at `converse.py:284-287`.** All owned by
  `junk_span_suppression.plan.md`. This plan does not touch `vcut/` or `l1/`.
- **The gate.** The evaluate/commit split, the coverage stage, and the
  `mirror_text` wrap are `brain_gate_integrity.plan.md`. §4.4's *policy*
  statement lives there; §4's *shape* lives here. Specifically: **Chain 5 of
  §4.6** (the six fail-opens that let a MANDATORY stage pass — `findings`,
  `review_out`, the stage-function exception, `_plan_conformance`, and the two
  hint swallows) is owned by that plan's §1.6 and §1.8, including the
  `ctx.durations = {}` case that weakens `validate` without raising anything.
  This plan gives those values a status; that plan decides what the gate does
  when it sees one. Landing either alone leaves a half-fix: a status nobody
  reads, or a policy with nothing to read.
- **Signal `kind` tokens and severity escalation.**
  `brain_accountability_architecture.plan.md` PART 3. §2.4 changes which cuts
  `_low_energy_runs` flags; it does NOT change how those flags are ranked or
  whether they block. Landing this plan first makes PART 3's severity policy
  operate on signals that measure something, which is why the sequencing puts
  perception before PARTS 2–4.
- **Commitments, reversals, `edit_moved`, the trace funnel.** PARTS 2, 4, 5.
- **Beat binding and the adversarial quote check.** Last in sequence;
  `brain_accountability_architecture.plan.md` amendment §A.6. Noted twice
  because §1 is its hard prerequisite: a quote check against
  sentences-that-merely-overlap verifies nothing.
