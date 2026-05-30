import jwt
from fastapi import Request, HTTPException
from app.config import get_settings


def get_current_user_id(request: Request) -> str:
    """
    Resolve the user ID for the current request.

    Dev mode: when settings.dev_user_id is non-empty, return it unconditionally
    and skip all token parsing. This lets the platform run without sign-up/login
    while we work on the rest of the system.

    Production: decode the Supabase JWT from the Authorization header and return
    its `sub` claim. Re-enabled automatically by clearing dev_user_id (set it to
    "" in .env or env var DEV_USER_ID).
    """
    settings = get_settings()

    if settings.dev_user_id:
        return settings.dev_user_id

    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = auth.removeprefix("Bearer ")

    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["HS256"],
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no sub claim")
        return user_id
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
