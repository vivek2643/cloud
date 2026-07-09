<!--
  GUIDANCE DOC -- Edso's reference for GUESSING under incomplete perception.

  Edso can't see or hear. It works off the transcript and each cut's information
  (label, summary, channel=said/done/shown, camera move, quality scores,
  take role, continuity). When that leaves a gap -- what to put on screen next,
  which angle to show -- this doc says how to make the best GUESS. It is not a
  rulebook and not per-format cookbook: two general principles cover almost
  every requirement, and the podcast section is one worked example that uses
  both. General craft (move vocabulary, pacing, continuity/junk meaning) lives
  in the system prompt, not here.

  How it's used: injected into Edso's system prompt (cached) and consulted in
  the planning step. Everything here is a "lean toward", to be BLENDED or
  OVERRIDDEN when the material or the user says otherwise.
-->

# Guidance: how to guess

## 1. Predict what's next from the transcript
The spoken words are the plan. Read them as intent: when a line names or implies
something to be seen ("look at this", "here's the setup", "so I grab the…", "and
then it does X"), the next thing on screen is probably a video cut OF that. Scan
the cut information — labels/summaries, channel (`done` = an action performed,
`shown` = something displayed), the `camera` move — for the cut that matches the
guess, and place it under or right after the line. If nothing matches, stay on
the speaker. This is how to arrange B-roll, demos, and actions for vlogs,
tutorials, talking-head, and anything where speech leads and the picture should
follow along. Strictly follow the transcript order; guess only the pictures that
ride alongside it.

## 2. Outlooks are alternate angles — pick per beat
An OUTLOOK is the same content shot from a different angle (not a retake, so it
has no "winner"). When a beat has outlooks, choose the angle that best serves
the moment — show whoever or whatever is relevant — and switch angles on the
beat, never mid-thought. This is how any multicam material is assembled
(interviews included, which are otherwise just talk-driven like a podcast).

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
