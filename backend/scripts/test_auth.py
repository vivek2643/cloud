"""
Tests for invite_codes.plan.md Stage 1: JWKS-based ES256 verification in
app/auth.py, replacing the old verify_signature=False decode. No real
network call -- _jwks() is patched to a fake PyJWKClient wired to a
locally-generated ES256 keypair standing in for Supabase's, so these tests
never hit the real JWKS endpoint.

Live-verified once, separately, against a real Supabase-issued token from a
throwaway signup (created and deleted via the admin API) -- not repeated
here to keep this suite offline. These tests instead pin the exact
verification properties that mattered: signature checked against the right
key, wrong signer rejected, missing/garbage tokens rejected, dev mode
untouched.

Run:  .venv/bin/python scripts/test_auth.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import jwt  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from app import auth  # noqa: E402

TEST_KID = "test-kid"
_PRIV = ec.generate_private_key(ec.SECP256R1())
_OTHER_PRIV = ec.generate_private_key(ec.SECP256R1())  # a different signer


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJWKClient:
    """Stands in for PyJWKClient: resolves TEST_KID to our test public key,
    raises on anything else -- exactly like a real client would for an
    unknown kid."""

    def get_signing_key_from_jwt(self, token):
        header = jwt.get_unverified_header(token)
        if header.get("kid") == TEST_KID:
            return _FakeSigningKey(_PRIV.public_key())
        raise jwt.exceptions.PyJWKClientError(f"Unable to find a signing key that matches: {header.get('kid')!r}")


class _FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


class _Patcher:
    def __init__(self):
        self._orig = {}

    def set(self, obj, name, value):
        self._orig[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def restore(self):
        for (obj, name), value in self._orig.items():
            setattr(obj, name, value)


class _SettingsOverride:
    """Wraps the real settings but forces dev_user_id off, so the JWT path
    actually runs (matches production; local dev's real default is
    non-empty, which the dev-mode test below exercises separately)."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def dev_user_id(self):
        return ""


def _sign(claims, priv=None, kid=TEST_KID, alg="ES256"):
    return jwt.encode(claims, priv or _PRIV, algorithm=alg, headers={"kid": kid} if kid else {})


def _real_settings():
    from app.config import get_settings
    return get_settings()


def test_valid_es256_token_accepted():
    token = _sign({"sub": "user-123", "aud": "authenticated"})
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        user_id = auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {token}"}))
    finally:
        p.restore()
    assert user_id == "user-123"
    print("ok  test_valid_es256_token_accepted")


def test_forged_hs256_token_rejected():
    forged = jwt.encode({"sub": "anyone", "aud": "authenticated"}, "x", algorithm="HS256")
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {forged}"}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        p.restore()
    print("ok  test_forged_hs256_token_rejected")


def test_wrong_signer_rejected():
    # Correct kid (so the client resolves a key at all) but signed with a
    # DIFFERENT private key -- the signature must fail verification even
    # though the token is otherwise well-formed ES256.
    token = _sign({"sub": "anyone", "aud": "authenticated"}, priv=_OTHER_PRIV, kid=TEST_KID)
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {token}"}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        p.restore()
    print("ok  test_wrong_signer_rejected")


def test_unknown_kid_rejected():
    token = _sign({"sub": "anyone", "aud": "authenticated"}, kid="not-a-real-kid")
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {token}"}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        p.restore()
    print("ok  test_unknown_kid_rejected")


def test_wrong_audience_rejected():
    token = _sign({"sub": "anyone", "aud": "some-other-audience"})
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {token}"}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        p.restore()
    print("ok  test_wrong_audience_rejected")


def test_expired_token_rejected():
    token = _sign({"sub": "anyone", "aud": "authenticated", "exp": int(time.time()) - 60})
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {token}"}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        p.restore()
    print("ok  test_expired_token_rejected")


def test_missing_sub_claim_rejected():
    token = _sign({"aud": "authenticated"})
    p = _Patcher()
    p.set(auth, "_jwks", lambda: _FakeJWKClient())
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({"Authorization": f"Bearer {token}"}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
            assert "sub" in e.detail
    finally:
        p.restore()
    print("ok  test_missing_sub_claim_rejected")


def test_missing_header_rejected():
    p = _Patcher()
    p.set(auth, "get_settings", lambda: _SettingsOverride(_real_settings()))
    try:
        try:
            auth.get_current_user_id(_FakeRequest({}))
            assert False, "expected HTTPException"
        except HTTPException as e:
            assert e.status_code == 401
    finally:
        p.restore()
    print("ok  test_missing_header_rejected")


def test_dev_mode_short_circuits_before_any_token_parsing():
    # settings.dev_user_id non-empty (local dev's real default): no header
    # at all, no JWKS call -- must resolve immediately, exactly as before.
    request = _FakeRequest({})
    user_id = auth.get_current_user_id(request)
    assert user_id == _real_settings().dev_user_id
    print("ok  test_dev_mode_short_circuits_before_any_token_parsing")


def main():
    test_dev_mode_short_circuits_before_any_token_parsing()
    test_valid_es256_token_accepted()
    test_forged_hs256_token_rejected()
    test_wrong_signer_rejected()
    test_unknown_kid_rejected()
    test_wrong_audience_rejected()
    test_expired_token_rejected()
    test_missing_sub_claim_rejected()
    test_missing_header_rejected()
    print("\nall auth tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
