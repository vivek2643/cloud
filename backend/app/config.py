from __future__ import annotations
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_key: str

    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket_name: str = "aerodrive"

    # Direct Postgres connection string for procrastinate + pgvector queries
    # (the supabase-py REST client can't run raw SQL or HNSW search).
    # Example: postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres
    database_url: str = ""

    cors_origins: List[str] = ["http://localhost:3000"]

    # Dev mode: when set (non-empty), the backend bypasses JWT validation and
    # treats every request as this user. Set to "" to re-enable real auth.
    dev_user_id: str = "00000000-0000-0000-0000-000000000001"

    # L1 guardrail: anything longer than this just gets S1 (proxy + thumb).
    max_l1_duration_seconds: int = 3600

    # Phase 3a: L3 query parsing via Anthropic Claude
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    # Phase 2 Stage D: hosted Qwen2.5-VL endpoint (Replicate / Anyscale style)
    # Leave empty until L2 Stage D is in use.
    qwen_vl_endpoint_url: str = ""
    qwen_vl_api_key: str = ""

    @property
    def r2_endpoint(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
