from typing import Optional
from supabase import create_client, Client
from app.config import get_settings

_client: Optional[Client] = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        settings = get_settings()
        schema = settings.db_schema.strip()
        if schema:
            # DB_SCHEMA (local-dev isolation): PostgREST does NOT honor the
            # psycopg search_path -- it targets a schema chosen per-request.
            # Pin the whole client to the dev schema. REQUIRES that schema to
            # be added to Supabase's "Exposed schemas" (Project Settings ->
            # API) or every call 404s. When DB_SCHEMA is unset we take the
            # branch below, leaving the client on PostgREST's default
            # `public` schema -- byte-for-byte production behavior.
            from supabase.lib.client_options import ClientOptions

            _client = create_client(
                settings.supabase_url,
                settings.supabase_service_key,
                options=ClientOptions(schema=schema),
            )
        else:
            _client = create_client(
                settings.supabase_url, settings.supabase_service_key
            )
    return _client
