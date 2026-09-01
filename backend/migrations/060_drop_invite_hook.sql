-- =============================================
-- 060: drop the before-user-created hook function, superseded by the front-door
-- gate in 059.
--
-- Safe now, and only now: the hook was deleted in the Supabase dashboard first.
-- 059 deliberately neutered this function rather than dropping it, because
-- dropping it while the dashboard toggle was still live would have failed every
-- signup. With the toggle gone nothing calls it, so the stub can go.
--
-- public.invite_redemptions is intentionally left in place. It is empty and now
-- unwritten -- the gate runs before an account exists, so there is no user_id to
-- record -- but it is exactly the right shape if per-account redemption
-- tracking is ever wanted, and an empty table costs nothing to keep.
-- =============================================

drop function if exists public.hook_require_invite_code(jsonb);
