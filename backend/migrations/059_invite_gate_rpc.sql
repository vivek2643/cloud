-- =============================================
-- 059: move the invite gate from Supabase's before-user-created hook to a
-- gate on our own front door.
--
-- WHY THE CHANGE. The hook could not let new users register with Google: an
-- OAuth round trip cannot carry an invite code, so the hook only ever saw
-- provider='google' with no way to check a code, and refused. Allowing Google
-- signup requires Supabase's public registration to stay open, which means the
-- anon key in the JS bundle can always be used to register directly. That
-- trade-off is accepted deliberately: this gate stops ordinary visitors, not
-- developers.
--
-- SAFE IN ANY ORDER. The hook function is neutered rather than dropped, so
-- this migration does not care whether the dashboard toggle has been turned
-- off yet. Dropping it outright would break every signup for as long as the
-- toggle stayed on, and putting the drop in a separate later migration would
-- leave a pending file that assert_up_to_date refuses to start the API with.
-- Neutering has neither failure mode.
-- =============================================

-- Turn the 058 hook into an unconditional allow. If the dashboard toggle is
-- still on, Supabase keeps calling this and every signup now passes; if it is
-- off, this is dead code. Either way the gate has moved to /invite. Once the
-- toggle is off for good this function can be dropped in a later migration.
create or replace function public.hook_require_invite_code(event jsonb)
returns jsonb
language plpgsql
as $$
begin
    return '{}'::jsonb;
end;
$$;

-- Claims a seat and reports whether the code was good. Called by the /invite
-- page before the visitor has any account, so it must be callable by anon --
-- hence security definer, since anon deliberately has no grants on
-- invite_codes (058) and must not be able to read the code list.
--
-- A seat is claimed on successful entry rather than at account creation.
-- Without the hook there is no server-side moment where a signup can be tied
-- back to a code, so "a seat" means "a browser that entered this code" rather
-- than "an account". Close enough for allocating a private beta, and the cap
-- still holds.
create or replace function public.redeem_invite_code(p_code text)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_claimed text;
begin
    -- Same atomic claim the hook used: a conditional UPDATE ... RETURNING is
    -- one statement, so two people racing for the last seat cannot both win.
    update public.invite_codes
       set used_count = used_count + 1
     where code = upper(trim(p_code))
       and not revoked
       and (expires_at is null or expires_at > now())
       and used_count < max_uses
    returning code into v_claimed;

    if v_claimed is null then
        -- Deliberately one outcome for every failure. Telling a stranger that
        -- a code exists but is full invites hunting for one that isn't.
        return jsonb_build_object('ok', false);
    end if;

    return jsonb_build_object('ok', true);
end;
$$;

revoke all on function public.redeem_invite_code(text) from public;
grant execute on function public.redeem_invite_code(text) to anon, authenticated;
