from typing import Optional
from supabase import create_client, Client
from app.config import get_settings

_client: Optional[Client] = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client
