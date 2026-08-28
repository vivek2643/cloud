-- =============================================
-- 058: invite-code gate for the private beta (invite_codes.plan.md).
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

-- =============================================
-- Stage 3: the hook function. Two rules: a valid unexhausted code, or no
-- account. Deliberately NOT distinguishing "no such code" from "code
-- exhausted" in the message -- telling a stranger that EDSO-8CA9 exists but
-- is full is an invitation to go hunting for one that isn't.
-- =============================================

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
