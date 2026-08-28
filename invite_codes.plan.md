# Invite-code gate — a private beta you can switch off in one click

Status: NOT STARTED. Written 2026-08-27 for implementation on `brain-rework`.

The goal is a code you hand to a person ("EDSO-8CA9"), which lets a fixed number
of people register, enforced somewhere the browser cannot reach, and which you
can remove in a few days without unwinding anything.

**Two independent pieces of work live in here.** Stage 1 (JWT verification) is a
permanent security fix. Stages 2–5 are the temporary gate. They do not depend on
each other mechanically — the gate will function without Stage 1 — but the gate
only *means* anything once Stage 1 lands, because until then the API accepts
forged tokens from people who never registered at all.

Everything in Ground Truth was verified against the running system on
2026-08-27. Do not re-derive it.

---

## STEP 0 — Answer this before writing any code

`backend/app/config.py:126` reads:

```python
dev_user_id: str = "00000000-0000-0000-0000-000000000001"
```

and `app/auth.py` short-circuits on it:

```python
if settings.dev_user_id:
    return settings.dev_user_id
```

The default is **non-empty**, so auth is bypassed unless something explicitly
sets `DEV_USER_ID=""`. It is **not** set in `render.yaml` and **not** in the
local `.env`. The only place it could be set is the `edso-shared` env group in
the Render dashboard (`render.yaml:14` documents that secrets live there).

**Go look at the `edso-shared` group in the Render dashboard now.**

* If `DEV_USER_ID` is `""` — good, production does real (if unverified) auth.
  Proceed.
* If it is **absent or non-empty** — production currently has *no* auth at all,
  and every user shares one placeholder account. In that case the invite gate is
  pointless on its own: fixing this is the entire job, and Stage 1 becomes
  urgent rather than merely a prerequisite. Set it to `""` as part of Stage 1,
  and expect existing sessions to start behaving differently.

Do not guess. This single value decides whether Stage 1 is a hardening step or
an emergency.

---

## Branch and starting state

Work on `brain-rework`, currently at `ec53947` and pushed to
`origin/brain-rework`. It is 2 commits ahead of `origin/main`.

Local dev shares production's Postgres (`public` schema) and R2 bucket, and
isolates only the job queue via `QUEUE_PREFIX=dev`. **This means the migration
below writes to the same database production uses.** That is normal here (see
`local_dev_queue_isolation.plan.md`) but it means the new tables appear in
production the moment you run the migration. That is harmless — nothing reads
them until the hook is switched on in Stage 4.

Frontend conventions are non-negotiable and documented in
`.cursor/skills/frontend-design/SKILL.md`. Read it before touching the signup
page: colors only via `var(--token)`, never hex; Tailwind for layout, CSS vars
for color; `cn()` for conditional classes.

---

## Ground truth (verified 2026-08-27 — do not re-derive)

### Signup never touches your backend

`frontend/src/app/(auth)/signup/page.tsx:23-26` calls
`supabase.auth.signUp({ email, password })` directly from the browser, using the
anon key that ships in the JS bundle. Google is `signInWithOAuth` at lines
46–54, equally direct.

Anyone can therefore register by POSTing to `<project>.supabase.co/auth/v1/signup`
with that public key, never loading your site. **A check in the React form is
decorative.** This is the whole reason the gate has to be a server-side hook.

### The backend does not verify tokens

`backend/app/auth.py:30-34` decodes with `options={"verify_signature": False}`
and there is no JWT secret anywhere in `config.py`. Any token with a `sub` claim
is accepted as that user.

### Your project signs with ES256 and publishes a JWKS

Verified live: `GET <SUPABASE_URL>/auth/v1/.well-known/jwks.json` returns HTTP
200 with a single EC P-256 key (`"alg":"ES256"`, `"kty":"EC"`, `"use":"sig"`).

This is good news — asymmetric keys mean **no new secret to store**. Verification
uses the public JWKS endpoint.

`PyJWT==2.10.1` (pinned, `requirements.txt:8`) exposes `PyJWKClient`, and
`cryptography 48.0.1` is installed. **But `cryptography` is not pinned** — it is
only present transitively. See Traps.

### The hook exists, is free, and runs as SQL

`before-user-created` is available on Free and Pro. It can be a Postgres
function, so nothing is deployed and the request never leaves your database.

Contract: `create function public.f(event jsonb) returns jsonb`. Return
`'{}'::jsonb` to allow; return an `error` object to reject:

```sql
return jsonb_build_object('error', jsonb_build_object(
  'message', 'shown to the user', 'http_code', 403));
```

The event payload includes everything needed:

```json
{ "user": { "id": "ff7fc9ae-...", "email": "a@b.com",
            "app_metadata": { "provider": "email" },
            "user_metadata": {} } }
```

* `user.user_metadata` ← where the invite code arrives from `options.data`
* `user.app_metadata.provider` ← `"email"` or `"google"`, so OAuth is
  distinguishable
* `user.id` ← the UUID the user *will* get; the row does not exist yet

Supabase's own docs document the OAuth case as an intended use: allow sign-in
for users who already exist while blocking new account creation via a provider.

**The hook only fires when an account is created.** Existing users — Google or
email — are never affected. This is what makes it safe to block new Google
signups without locking out anyone who already has an account.

### Current data state

`public` has 31 tables; there is no user, profile, invite, or entitlement table
of any kind (`sync_group_members` is unrelated multicam). `auth.users` has 5
rows. Latest migration is `057_upload_links_drop_auth_fk.sql`, so yours is
**058**.

---

## Stage 1 — Verify JWTs for real

Edit `backend/app/auth.py`. Replace the unverified decode with JWKS-based ES256
verification.

```python
from functools import lru_cache
import jwt
from jwt import PyJWKClient

@lru_cache(maxsize=1)
def _jwks() -> PyJWKClient:
    # Cached: PyJWKClient keeps fetched keys in memory and re-fetches only when
    # it sees an unknown kid, so key rotation heals itself.
    return PyJWKClient(f"{get_settings().supabase_url}/auth/v1/.well-known/jwks.json")
```

and in `get_current_user_id`:

```python
    try:
        signing_key = _jwks().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no sub claim")
        return user_id
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
```

Keep the `dev_user_id` short-circuit above it — local dev depends on it.

Pin the crypto dependency in `backend/requirements.txt`:

```
PyJWT[crypto]==2.10.1
```

**Verify before moving on.** A wrong turn here locks every user out.

1. A forged token must now fail. Mint one locally with
   `jwt.encode({"sub": "anyone", "aud": "authenticated"}, "x", algorithm="HS256")`
   and confirm the API returns 401.
2. A real token must still pass. Sign in through the frontend, copy the access
   token from the session, and confirm an authenticated call returns 200.
3. Local dev must be unaffected while `DEV_USER_ID` is non-empty.

Do this stage on its own and confirm it before starting Stage 2, so that if
sign-in breaks you know exactly which change did it.

---

## Stage 2 — The codes table

Create `backend/migrations/058_invite_codes.sql`.

```sql
-- =============================================
-- 058: invite-code gate for the private beta.
--
-- Deliberately mirrors upload_links (056): a token, a cap, a used counter,
-- revoked, expires_at. Same semantics, already understood.
--
-- NO foreign key from invite_redemptions.user_id to auth.users. The
-- before-user-created hook runs BEFORE the auth.users row exists, so such an
-- FK would reject every signup. 057 learned this same lesson the hard way.
-- =============================================

create table if not exists public.invite_codes (
    code        text primary key,
    max_uses    int not null default 1 check (max_uses > 0),
    used_count  int not null default 0 check (used_count >= 0),
    expires_at  timestamptz,
    revoked     boolean not null default false,
    note        text,
    created_at  timestamptz not null default now()
);

create table if not exists public.invite_redemptions (
    user_id     uuid primary key,
    code        text not null references public.invite_codes(code),
    email       text,
    redeemed_at timestamptz not null default now()
);

create index if not exists invite_redemptions_code_idx
    on public.invite_redemptions (code);

-- The hook function runs as supabase_auth_admin and must reach both tables.
grant all on table public.invite_codes, public.invite_redemptions
    to supabase_auth_admin;
revoke all on table public.invite_codes, public.invite_redemptions
    from authenticated, anon, public;
```

Codes are stored **uppercase**; the hook uppercases and trims what it receives,
so `edso-8ca9` typed with a trailing space still works.

### Generating codes (this is the bit you run by hand)

```sql
insert into public.invite_codes (code, max_uses, note)
select 'EDSO-' || upper(substr(md5(random()::text), 1, 4)), 5, 'launch batch'
from generate_series(1, 10)
returning code, max_uses;
```

Change `5` (signups per code) and `10` (how many codes). Watching and revoking:

```sql
select code, used_count, max_uses, revoked from public.invite_codes
 order by created_at desc;

update public.invite_codes set revoked = true where code = 'EDSO-8CA9';
```

---

## Stage 3 — The hook function

Same migration file. Two rules: a valid unexhausted code, or no account.

```sql
create or replace function public.hook_require_invite_code(event jsonb)
returns jsonb
language plpgsql
as $$
declare
    v_provider text;
    v_code     text;
    v_claimed  text;
begin
    v_provider := event->'user'->'app_metadata'->>'provider';

    -- OAuth flows cannot carry an invite code, so new social accounts are
    -- refused outright. Existing users are unaffected: this hook only runs
    -- when an account is about to be created.
    if v_provider is distinct from 'email' then
        return jsonb_build_object('error', jsonb_build_object(
            'message', 'Edso is invite-only right now. Please sign up with email and your invite code.',
            'http_code', 403));
    end if;

    v_code := upper(trim(event->'user'->'user_metadata'->>'invite_code'));

    if v_code is null or v_code = '' then
        return jsonb_build_object('error', jsonb_build_object(
            'message', 'An invite code is required to sign up.',
            'http_code', 403));
    end if;

    -- Claim a seat atomically. A conditional UPDATE ... RETURNING is a single
    -- statement, so two people racing for the last seat cannot both win.
    update public.invite_codes
       set used_count = used_count + 1
     where code = v_code
       and not revoked
       and (expires_at is null or expires_at > now())
       and used_count < max_uses
    returning code into v_claimed;

    if v_claimed is null then
        return jsonb_build_object('error', jsonb_build_object(
            'message', 'That invite code is not valid or has already been used up.',
            'http_code', 403));
    end if;

    insert into public.invite_redemptions (user_id, code, email)
    values ((event->'user'->>'id')::uuid, v_claimed, event->'user'->>'email')
    on conflict (user_id) do nothing;

    return '{}'::jsonb;
end;
$$;

grant execute on function public.hook_require_invite_code to supabase_auth_admin;
revoke execute on function public.hook_require_invite_code
    from authenticated, anon, public;
```

Deliberately **not** done: distinguishing "no such code" from "code exhausted"
in the message. Telling a stranger that `EDSO-8CA9` exists but is full is an
invitation to go hunting for one that isn't.

---

## Stage 4 — Switch it on (manual, in the dashboard)

Supabase Dashboard → Authentication → Hooks → **Before User Created** → select
the Postgres function `public.hook_require_invite_code` → enable.

There is no SQL for this step; it is the toggle, and it is also the off switch.

Immediately verify with a throwaway email: signing up with no code, and with a
bogus code, must both fail with your message. Signing up with a real code must
succeed and bump `used_count`.

---

## Stage 5 — The invite code field on signup

`frontend/src/app/(auth)/signup/page.tsx`.

1. Add an `inviteCode` state and a required text input above Email, labelled
   "Invite code". Style it exactly like the existing Email input (lines 122–130)
   — same classes, same `var(--border)` / `var(--background)` tokens.
2. Pass it through:

```ts
const { data, error: authError } = await supabase.auth.signUp({
  email,
  password,
  options: { data: { invite_code: inviteCode.trim().toUpperCase() } },
});
```

3. Hide the Google button and its "or" divider **on this page only** (lines
   88–106). New Google accounts are rejected by the hook, so leaving the button
   would offer a path that always fails. Leave `login/page.tsx` completely
   alone: existing Google users must still be able to sign in.
4. Add a line under the form: "Don't have an invite code? Join the waitlist" or
   similar, so a stranger who lands here understands why they are stuck. Copy is
   your call.

The hook's rejection message already arrives in `authError.message` and renders
in the existing error box (lines 109–116), so no new error handling is needed.

---

## Stage 6 — Verify

Run through this end to end, not just the happy path:

* No code → rejected with your message.
* Bogus code → rejected.
* Valid code → account created, `used_count` incremented by exactly 1.
* Same code past `max_uses` → rejected; counter does not exceed `max_uses`.
* `revoked = true` → rejected immediately.
* Google on the signup page → button is gone; a direct OAuth attempt for a *new*
  account is rejected.
* **An existing Google user can still sign in.** This is the one that would hurt
  most if wrong.
* Direct API bypass: POST to `<project>.supabase.co/auth/v1/signup` with the
  anon key and no invite code. It must be rejected — this is the entire point of
  using a hook rather than a form check.
* Frontend: `npx tsc --noEmit` and `npx eslint` clean. Do **not** run
  `npm run build` while the dev server is running; it clobbers `.next` and 500s
  the dev server (this happened on 2026-08-26).
* Backend: the 85-file suite in `backend/scripts/test_*.py` still passes, and
  `pyflakes` is clean on `app/auth.py`.

---

## Removing the gate later (the point of all this)

1. Dashboard → Authentication → Hooks → disable **Before User Created**. Signup
   is open again, immediately. **This is the whole off switch.**
2. When convenient, delete the invite field and restore the Google button on the
   signup page.
3. The tables and function can stay indefinitely — inert and harmless — or be
   dropped whenever.

Nothing needs migrating back. Accounts created during the beta are ordinary
Supabase users, indistinguishable afterward, because the hook only decides
*whether* a user is created, never *how*.

Stage 1 is not part of this and should never be reverted.

---

## Traps

**Do not add an FK from `invite_redemptions.user_id` to `auth.users`.** The hook
runs before that row exists, so every signup would fail. Migration 057 exists
because 056 made exactly this mistake.

**Pin `PyJWT[crypto]`.** ES256 needs `cryptography`, which is currently only a
transitive dependency. It works locally today; it may be absent in a fresh
production build, and the failure mode is every request 401ing.

**A seat is spent at the hook, not at confirmation.** If email confirmation is
enabled and someone never confirms, or signup fails afterward for an unrelated
reason such as a duplicate email, the seat is gone and an orphan row may sit in
`invite_redemptions`. This is a slow leak, not a hole. Raise `max_uses` when it
bites. Do not try to fix it with a cleanup job.

**Do not put the check in the React form.** It cannot work; see Ground Truth.

**Do not disable Supabase's public signup toggle as well.** It is redundant once
the hook is live, and it would also block the admin-created accounts you might
want for support.

**Grants are not optional.** Without `grant execute ... to supabase_auth_admin`
and the table grants, the hook fails at runtime and — depending on how Supabase
treats a failed invocation — may either block all signups or let them through.
Verify Stage 4 by actually signing up, not by reading the SQL.
