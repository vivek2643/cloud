<!--
  GUIDANCE DOC -- Edso's reference for GUESSING under incomplete perception.

  Edso can't see or hear. It works off the transcript and each cut's information
  (label, summary, channel=said/done/shown, on-screen text, energy, camera move,
  quality scores, take role, continuity). When that leaves a gap -- what to put on screen next,
  which angle to show -- this doc says how to make the best GUESS. It is not a
  rulebook and not per-format cookbook: a few general principles cover almost
  every requirement, and the podcast section is one worked example that uses
  the first two. General craft (move vocabulary, pacing, continuity/junk meaning) lives
  in the system prompt, not here.

  How it's used: injected into Edso's system prompt (cached) and consulted in
  the planning step. edso_think_act_check.plan.md change 2: these are BINDING
  DEFAULTS now, not a soft "lean toward" -- follow them unless the user's ask
  or a clear material reality calls for otherwise (see _guidance_block's own
  framing text and _LOOP_SYSTEM's PRECEDENCE line, both in converse.py).
-->

# Guidance: how to guess

## 1. Read the words and pictures as one
The words and the pictures are a single reading, not a ranking — take each beat
from everything at once and lean on whichever carries the meaning at that moment.

When the words lead (talk-driven material — vlog, tutorial, talking-head): read
them as intent. When a line names or implies something to be seen ("look at
this", "here's the setup", "so I grab the…", "and then it does X"), the next
thing on screen is probably a video cut OF that. Scan the cut information —
labels/summaries, channel (`done` = an action performed, `shown` = something
displayed), any on-screen `text:`, the `camera` move — for the cut that matches
the guess, and place it under or right after the line. If nothing matches, stay
on the speaker. Follow the transcript order; guess only the pictures that ride
alongside it.

When the words DON'T lead (little or no speech — b-roll, montage, action): the
pictures carry the meaning. Read the visual signals instead — each cut's energy
(`nrg`), whether shots continue or break (welds / `cut:`), any on-screen
`text:`, its `camera` move — to decide what matters and what follows what. There
is NO house style to fall back on: take the shape and the pace from the user's
goal and the material in front of you, never from a default "look". If music
drives the piece, let its beat set the pace. In mixed material (reels, most
social) the lead flips moment to moment — read each beat for whatever is carrying
it right then. When timing a punch-in or choosing where to hold within a cut,
lean on its `peak:` (when present) — the cut's own strongest instant.

## 2. Outlooks are alternate angles — pick per beat
An OUTLOOK is the same content shot from a different angle (not a retake, so it
has no "winner"). When a beat has outlooks, choose the angle that best serves
the moment and switch angles on the beat, never mid-thought. When someone is
speaking, the default is the angle where the SPEAKER is on camera — cut to who's
talking; holding on a listener or reaction shot is a deliberate choice, not the
fallback. So if a beat plays the speaker OFF camera while an on-camera angle of
that same moment exists, that's usually a miss — switch to the on-camera angle
unless you meant the reaction. This is how any multicam material is assembled
(interviews included, which are otherwise just talk-driven like a podcast).

## 3. Fitting a clip to a target window
Two situations are the SAME generic operation: landing an overlay on a specific
line, and cutting a shot to a beat. Both just fit a clip to a target program
window `[A,B]`. To do it: get `[A,B]` from whatever sense exposes it, then
adjust the clip's length with `trim` or pace (`retime` for video's playback
speed, or speech's dead-air trim). Don't compute the exact result — read the
actual length back from `read_state`/the Program Map (or `review`, once
placed) and adjust again if it's off; that loop is exact, blind arithmetic
isn't.

## 4. Select for the video's purpose
Most videos exist to serve a higher purpose for their audience — a teaser to
spark curiosity, a tutorial to help viewers learn, a documentary to inform and
keep them engaged. When the material is larger than the ask allows, let that
purpose guide what to keep. Read the purpose from the user's ask and the
material; when it's unclear and the selection hinges on it, ask rather than
assume.

A moment may hold several beats — take it whole or take its key beat(s),
guided by your length budget and the video's purpose.

## Podcast / multicam (worked example — uses both principles)
Conversation filmed from several fixed angles. The alternate angles of one
speaker are outlooks (principle 2), and the transcript drives who to show
(principle 1).

Lean toward: keep the CURRENT speaker on screen, choosing the angle with the
highest total_quality (this already biases to the on-camera close-up). Hold an
angle for the whole thought rather than cutting on every pause — let delivery,
not the clock, drive the change. On a speaker change, cut to the new speaker's
best angle at their first word.

Rapid back-and-forth (fast turn-taking, overlaps, reactions): lean to the
widest shot that holds both people so the exchange reads without whip-cutting.
There's no fixed trigger for "rapid" — judge it from the turn pattern on the
go; if it's a close call and the choice matters, ask the user rather than
guess. When no true wide angle exists (all cameras are single-person), you have
two good moves for a lively exchange: stay on whoever is speaking and cut on
each turn, OR use `split_screen` to show both single-person angles at once — the
speaker on one side and the other person's outlook (their angle for the same
beat) on the other, so the back-and-forth reads without whip-cutting. Reach for
the split when turns come too fast to cut cleanly. Cutaways/B-roll only to cover
a real disfluency gap, not for variety.
