from functools import lru_cache

import jwt
from fastapi import Request, HTTPException
from jwt import PyJWKClient

from app.config import get_settings


@lru_cache(maxsize=1)
def _jwks() -> PyJWKClient:
    # Cached: PyJWKClient keeps fetched keys in memory and re-fetches only when
    # it sees an unknown kid, so key rotation heals itself.
    return PyJWKClient(f"{get_settings().supabase_url}/auth/v1/.well-known/jwks.json")


def get_current_user_id(request: Request) -> str:
    """
    Resolve the user ID for the current request.

    Dev mode: when settings.dev_user_id is non-empty, return it unconditionally
    and skip all token parsing. This lets the platform run without sign-up/login
    while we work on the rest of the system.

    Production: verify the Supabase JWT against the project's published JWKS
    (ES256) and return its `sub` claim. Re-enabled automatically by clearing
    dev_user_id (set it to "" in .env or env var DEV_USER_ID).
    """
    settings = get_settings()

    if settings.dev_user_id:
        return settings.dev_user_id

    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = auth.removeprefix("Bearer ")

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
