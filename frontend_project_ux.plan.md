# Project UX pass — kill the dead buttons, scope the chrome to the project, ship the upload link

Status: NOT STARTED. Written 2026-08-26 for implementation on `brain-rework`.

Five changes. Four are small and self-contained. **One (Stage 2, the upload
link) is a genuinely new feature** with a migration, public endpoints, and a new
route — do not let its size surprise you mid-way.

Everything below was verified against the running code on `brain-rework`. The
line numbers are real. Do not re-derive the ground truth section.

---

## Branch and starting state

Work on `brain-rework`, which is currently identical to `origin/main`
(`aa42fba`). The archived brain experiments live on `local-dev-isolation` and
are **not** wanted here.

Local dev is already running and correctly wired: API on `:8000`, frontend on
`:3000`, worker bound to the `dev-*` queues. Local shares production's Postgres
(`public` schema) and R2 bucket, and isolates only the job queue via
`QUEUE_PREFIX=dev`. See `local_dev_queue_isolation.plan.md` — that model is
settled, do not re-litigate it or reintroduce `DB_SCHEMA=dev`.

Frontend conventions are non-negotiable and documented in
`.cursor/skills/frontend-design/SKILL.md`. Read it before writing any UI. The
short version: colors only via `var(--token)` (never hex), orange (`--accent`)
at most a couple of places per view, Tailwind for layout and CSS vars for color,
`cn()` for conditional classes, no `tailwind.config.js`.

---

## Ground truth (verified 2026-08-26 — do not re-derive)

### The two buttons are dead

`frontend/src/app/(drive)/drive/folder/[folderId]/page.tsx`

* Line 10 imports `Upload, Link2, Share2`.
* Lines **73–80**: the `Share` button. No `onClick`. Renders and does nothing.
* Lines **81–88**: the `Upload Link` button. No `onClick`. Same.

There is **no** share/link/token concept anywhere in the backend. Stage 2 builds
it from zero.

### Export's backend works — the break is in the UI

Proven end to end on 2026-08-26 against the local API: a `kind:"srt"` export for
thread `9120080a-053e-45ea-a82f-3e86d505a22a` returned HTTP 200 `queued`, the
local `dev-export` worker picked up `build_export[1218]`, and it finished in
**1.4s** with `status:"done"` and a valid presigned `output_url`. The router
(`app/routers/exports.py`) and the queue path are healthy. **Do not go looking
for a backend export bug — there isn't one.**

The actual failure is `frontend/src/components/export-view.tsx`:

* Line 40 reads `threadId` from `useEditDocStore`.
* That store's `threadId` is set **only** by `seed(...)`, and `seed` has exactly
  one caller in the entire frontend: `ai-edit-panel.tsx:238`.
* So if you open the Export lens from the sidebar without having opened the AI
  edit panel first, `threadId` is `null` and line 107 renders the "Start an edit
  first" empty state. There is no way for Export to find a thread on its own.

Second, smaller defect in the same file:

* Line 84: `if (!token || !threadId || busy) return;` — a missing auth token makes
  the Export button a **silent no-op**, with no error shown.
* Line 64: the same silent return inside `poll()`.
* This is gratuitous, because `request()` in `lib/api.ts:12` only attaches the
  header `if (token)` — a tokenless call is sent anyway and the backend accepts
  it (see the auth note below).

`store.list_threads` (`app/services/l3/store.py:73`) returns `id, title, status,
created_at, file_count, max_version` — it does **not** return `file_ids`, so the
frontend currently cannot filter threads to a project.

### Chrome is rendered unconditionally

`frontend/src/app/(drive)/layout.tsx`, in `DriveShell`:

* Line **73**: `<Sidebar />` — renders on every `(drive)` route including `/drive`.
* Line **80**: `<AiEditPanel />` — same.

`Sidebar` (`components/sidebar.tsx:15-22`) is purely project-scoped: Media,
Cuts, Captions, Export. None of it means anything at the project list.

`aiPanelOpen` in `stores/drive-store.ts:68-71` gates **both** the panel body
(`ai-edit-panel.tsx:396`) and the timeline portal into `#ai-editor-dock`
(`ai-edit-panel.tsx:591`). So a single `closeAiPanel()` closes the editor and
the timeline together. `syncPanelOpen`/`closeSyncPanel` sit alongside it at
lines 73-76.

### There is no "recently opened" data

`Folder` (`lib/api.ts:24-31`) is `id, user_id, name, parent_id, created_at,
updated_at`. `updated_at` tracks modification, not opening. Per the decision
below, recents are **client-side only** — no migration, no backend change.

### Auth is currently bypassed

`app/auth.py:20-21`: when `settings.dev_user_id` is set, `get_current_user_id`
returns it unconditionally and skips all token parsing. It is set, which is why
the entire database has exactly one user
(`00000000-0000-0000-0000-000000000001`).

Two consequences for Stage 2. First, you **cannot meaningfully test the
"different user" half of the upload-link flow locally** — every request resolves
to the same dev user. Test the token path itself (valid / expired / revoked /
exhausted), not the identity separation. Second, the public endpoints must
derive `user_id` and `folder_id` **from the link row only**, never from a
client-supplied field — that property is what makes the feature correct once
real auth is switched back on.

Unrelated but worth knowing since you'll be in this file: `auth.py:30-33` decodes
the JWT with `verify_signature: False`. That is a real production issue, it is
**out of scope here**, and Stage 2 must not lean on JWT validation for safety.

### Next migration number

`backend/migrations/` is at `055_frame_descriptors.sql`. Yours is **`056`**.

---

## Decisions already made (do not re-open)

| Question | Decision |
|---|---|
| Who can use an upload link | **Fully anonymous** — no account, no login |
| Link settings offered | **Expiry, max files, revoke.** Nothing else |
| Recents storage | **Browser localStorage** — no backend, no migration |
| Home page keeps | **Search bar and New Project**, plus Projects and Recents |
| Export symptom | Unverified by the user; treat the `threadId` gap above as the cause |

---

## STAGE 1 — Remove Share

`drive/folder/[folderId]/page.tsx`. Delete lines 73–80 and drop `Share2` from
the line 10 import. Leave `Upload` and `Link2` — Stage 2 needs `Link2`.

That is the whole stage. Ship it on its own so the diff is unambiguous.

---

## STAGE 2 — The upload link

The only large stage. Land the backend and frontend together; a generated link
that 404s is worse than no button.

### 2.1 Migration `056_upload_links.sql`

```sql
create table if not exists upload_links (
  id          uuid primary key default uuid_generate_v4(),
  token       text not null unique,
  folder_id   uuid not null references folders(id) on delete cascade,
  user_id     uuid not null,
  expires_at  timestamptz,          -- null = never expires
  max_files   int,                  -- null = unlimited
  used_count  int  not null default 0,
  revoked     bool not null default false,
  created_at  timestamptz not null default now()
);
create index if not exists upload_links_token_idx  on upload_links (token);
create index if not exists upload_links_folder_idx on upload_links (folder_id);
```

Follow the existing migrations' style. `user_id` is the **project owner**, copied
from the folder at creation time — uploaded files are owned by them, not by the
anonymous uploader.

Generate `token` with `secrets.token_urlsafe(32)`. It is a bearer credential:
anyone holding it can write into the project, so it must be unguessable.

### 2.2 Refactor the upload core before adding endpoints

**Do this first, and do not skip it.** `app/routers/upload.py` holds the whole
upload lifecycle — `presign` (56), `multipart/create` (183),
`multipart/complete` (226), `multipart/abort` (248), the analysis-proxy pair
(271, 290), and `complete` (325). Note `presign` builds
`r2_key = f"raw/{user_id}/{file_id}/{filename}"` at line 70.

The public flow needs **all** of these, not just `presign`: video files over
100 MiB go down the multipart path (`upload-zone.tsx:26`), and L1 only starts
once the client analysis proxies land (see the comment at `upload.py:45-50`). A
public flow that skips proxies uploads files that are never analyzed.

So extract the body of each handler into a plain function taking explicit
`(user_id, folder_id, ...)` and have the authenticated router call it. Then the
public router calls the same functions with values read off the link row. Two
parallel copies of this logic will drift, and the drift will be silent.

### 2.3 New router `app/routers/upload_links.py`

Owner-facing (all `Depends(get_current_user_id)`, all verifying the folder
belongs to the caller):

* `POST   /api/folders/{folder_id}/upload-links` — body `{expires_in_hours?: int,
  max_files?: int}`. Creates and returns the row plus the full shareable URL.
* `GET    /api/folders/{folder_id}/upload-links` — list, for the manage view.
* `DELETE /api/upload-links/{link_id}` — sets `revoked = true`. Do not hard-delete;
  a revoked link should be able to say "this link was turned off."

Public (**no auth dependency at all**):

* `GET  /api/public/upload-links/{token}` — validate, return only
  `{project_name, expires_at, remaining}`. Never leak `user_id`, `folder_id`, or
  anything about other files in the project.
* `POST /api/public/upload-links/{token}/presign`
* `POST /api/public/upload-links/{token}/multipart/create`
* `POST /api/public/upload-links/{token}/multipart/complete`
* `POST /api/public/upload-links/{token}/multipart/abort`
* `POST /api/public/upload-links/{token}/files/{file_id}/analysis-proxies/presign`
* `POST /api/public/upload-links/{token}/files/{file_id}/analysis-proxies/complete`
* `POST /api/public/upload-links/{token}/files/{file_id}/complete`

Shared guard, used by every public route:

```
_resolve_link(token) -> row
    unknown token         -> 404
    revoked               -> 410
    expires_at passed     -> 410
    used_count >= max_files -> 410
```

Two rules that matter:

* **Increment `used_count` at `complete`, not at `presign`.** An abandoned or
  failed presign must not burn the quota. Check the cap on the way in, count on
  the way out.
* The `{file_id}` public routes must verify that file actually belongs to this
  link's folder before acting. Otherwise a link for project A becomes a write
  primitive against any file id the caller can guess.

Register the router in `app/main.py` next to the others.

### 2.4 Frontend — the generator dialog

New `components/upload-link-dialog.tsx`, modeled on the existing
`create-folder-dialog.tsx` so it matches. Contents: an expiry select (24 hours /
7 days / never), a max-files number input (blank = unlimited), a Generate
button, the resulting URL in a read-only field with a copy button, and a list of
existing links with a Revoke action.

Wire it to the existing `Upload Link` button (`page.tsx:81-88`) — add the
`onClick`, keep the markup.

### 2.5 Frontend — the public page

New route **outside** the `(drive)` group, because that group's `DriveGuard`
redirects unauthenticated visitors to `/login` and `DriveShell` wraps everything
in navbar + sidebar + panels:

```
frontend/src/app/(public)/layout.tsx      -- bare shell, no auth, no chrome
frontend/src/app/(public)/u/[token]/page.tsx
```

The page: call `GET /api/public/upload-links/{token}` on mount; on 404/410 show a
plain "This link is no longer active" state; otherwise show the project name, a
dropzone, per-file progress, and a done state. **Nothing else** — no navigation,
no project contents, no way to reach the rest of the app.

For the upload itself, reuse the machinery from 2.2 rather than duplicating it.
`useUploadFiles` (`upload-zone.tsx:80`) is currently bound to `useDriveStore`
(for `currentFolderId` and the progress list) and `useAuthStore` (for the token).
Parameterize it over an API binding plus progress callbacks, then have both the
authenticated zone and the public page supply their own. The proxy generation
(`generateProxies`, `lib/proxy-gen.ts`) must run in the public path too, or
uploads through a link never get analyzed.

---

## STAGE 3 — Make Export reachable

Two independent fixes in `components/export-view.tsx`, plus one small backend
addition.

**3.1 Let Export find a thread.** Add `file_ids` to the `select` and the returned
dict in `store.list_threads` (`app/services/l3/store.py:73-87`). Then in
`ExportView`, when `threadId` is `null`, fetch `GET /api/edit/threads`, keep the
threads whose `file_ids` intersect the current project's files, and select the
most recent (the query is already `order by t.updated_at desc`). If more than one
matches, show a small picker; if none, keep today's empty state, which is then
telling the truth.

**3.2 Stop the silent no-op.** Drop `!token` from the guard at line 84 and from
`poll()` at line 64. `request()` already omits the header when there's no token,
and the backend accepts it. If a call does fail, surface the error through the
existing `setError` path rather than returning silently.

---

## STAGE 4 — Home page shows only what belongs to it

**4.1 Hide the project chrome.** In `layout.tsx`'s `DriveShell`, read
`usePathname()` and treat `/drive` as home. Render `<Sidebar />` only when not
home. Keep the `#ai-editor-dock` div mounted unconditionally — `AiEditPanel`
portals into it by id, and removing it from the tree risks a null portal target
during transitions.

**4.2 Recents.** Add `lib/recent-projects.ts` with a tiny localStorage helper —
read, and a `record(id, name)` that de-dupes by id, unshifts, and caps at ~8
entries. Call `record()` from the folder page's existing mount effect
(`folder/[folderId]/page.tsx:44-48`), where the breadcrumb already resolves the
project name.

On the home page (`drive/page.tsx`), render a **Recents** section above
**Projects**, ordered most-recent first, reusing the existing folder card
presentation from `DriveContent` rather than inventing a second card style.
Reconcile against the loaded folder list and drop entries whose folder no longer
exists — a deleted project must not linger in recents.

Keep the search bar (lines 66–92) and New Project (lines 110–117) exactly as they
are.

---

## STAGE 5 — Close the editor when leaving for home

Same `usePathname()` already added in 4.1. In `DriveShell`, on transition to
`/drive`, call `closeAiPanel()` and `closeSyncPanel()`.

Put it in the layout, not in `drive/page.tsx` — the layout is where the panels
are mounted, and 4.1 already computes `isHome` there, so both behaviors read from
one source. Guard the effect on the pathname so it fires on the transition, not
on every render.

---

## Traps

**Do not put the public upload page inside `(drive)`.** `DriveGuard` will bounce
your anonymous visitor to `/login` and `DriveShell` will wrap the page in the
full app chrome. This is the single most likely way to get Stage 2 wrong.

**Do not trust any client-supplied `user_id` or `folder_id` on a public
endpoint.** Both come from the link row. Because `DEV_USER_ID` currently makes
every request resolve to the same user, a mistake here will look like it works
locally and become a cross-tenant write the moment real auth is restored.

**Do not fork the upload logic.** Multipart, analysis proxies, and the L1 kick
are all load-bearing. Two copies will drift and the failure mode is "files
uploaded through a link are silently never analyzed."

**Do not hunt for an export backend bug.** It was measured working. If export
still misbehaves after Stage 3, capture the browser network tab before touching
Python.

**Do not add a migration for recents.** The decision is localStorage. A
`last_opened_at` column is a different, larger feature.

**Do not hardcode colors.** Every new surface here — the dialog, the public page,
the recents cards — goes through `var(--token)`. Orange stays rare.

---

## Verification

Stages are independently shippable; verify each before moving on.

1. **`npm run build`** in `frontend/` — catches the type and import errors that
   `next dev` will happily paper over.
2. **Backend suite** for Stage 2/3: from `backend/`, run the `scripts/test_*.py`
   files as standalone scripts with the venv python. `timeout` does not exist on
   macOS — do not wrap them in it.
3. **Stage 2 end to end**: generate a link, open it in a **private window** (no
   session), upload a real video, and confirm the file lands in the project *and*
   that L1 is enqueued. Then verify each rejection path individually: expired,
   revoked, and max-files-exhausted.
4. **Stage 3**: hard-reload the app, go straight to the Export lens without
   opening the AI panel, and confirm it finds the thread. `srt` is the fast kind
   for a loop — it completed in 1.4s.
5. **Stages 4/5**: open a project, open the AI panel so the timeline docks,
   navigate back to `/drive`, and confirm the sidebar is gone, the timeline is
   gone, and Recents lists the project you just left.

---

## Suggested order

Stage 1, then 4 and 5 together (they share the `usePathname` change), then 3,
then 2 last on its own branchable commit. That front-loads the quick visible
wins and leaves the one large feature isolated.
