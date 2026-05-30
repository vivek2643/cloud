# Prompts

Every prompt sent to Claude lives in this directory as a plain Markdown/text
file. Editing a file here takes effect on the **next** request — the loader
re-reads the file on every call (cheap, ~0.1 ms).

| File | Where it's used | Substitutions |
| --- | --- | --- |
| `query_parser.md` | L3 — turns the user's natural-language edit prompt into a structured JSON query | _(none — sent as system prompt)_ |
| `narrative_stage.md` | L2 Stage D — analyzes 3 keyframes of a shot to produce `{description, role, valence}` | `{transcript}` |

## Editing rules

1. Keep the **JSON schema sections verbatim** — the response parser depends on
   exact field names. If you rename a field, also update
   `query_parser._normalize` or `narrative_stage._coerce`.
2. Substitution placeholders use Python's `str.format` braces, e.g.
   `{transcript}`. Literal `{` / `}` characters must be doubled (`{{`/`}}`).
3. Lines starting with `<!--` are comments in Markdown but **are sent to
   Claude** — strip them if you want.
