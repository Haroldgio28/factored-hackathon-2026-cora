"""Runtime configuration (task 0.3, REQ-36, REQ-48).

All values come from environment variables or a gitignored `.env` file; `.env.example`
documents the names. No credentials live here: AWS access is resolved by boto3 from the
named SSO profile, so the settings only carry profile names, regions and model ids.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default session TTL (absolute, in minutes) for the mock identity service (REQ-11). Named once
# so the field default and the blank-template validator agree. REQ-11 phrases this as "idle";
# the mock implements an absolute expiry (stricter), with sliding refresh deferred to task 2.3.
DEFAULT_SESSION_TTL_MINUTES = 15

# Default runtime location for the JSONL trace file (task 5.1, REQ-39). Mirrors the `data/_state/`
# gitignored runtime convention `JsonHandoffStore`/`JsonCardOverlay` already use, so traces never
# get committed. Named once so the field default and the blank-template validator agree.
DEFAULT_TRACE_DIR = Path("data/_state/traces")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Personal AWS account (Bedrock) - documentation/AWS_SETUP.md
    aws_profile: str = Field("cora-dev", alias="AWS_PROFILE")
    aws_region: str = Field("us-east-1", alias="AWS_REGION")

    # Bedrock inference profile ids; empty means "not configured" and LLM calls fail closed.
    bedrock_model_id: str = Field("", alias="CORA_BEDROCK_MODEL_ID")
    bedrock_judge_model_id: str = Field("", alias="CORA_BEDROCK_JUDGE_MODEL_ID")
    bedrock_fallback_model_id: str = Field("", alias="CORA_BEDROCK_FALLBACK_MODEL_ID")

    # Mock identity service (task 2.1, REQ-10/REQ-11). The HMAC key signs session JWTs; it is
    # a secret, so only its NAME ships in `.env.example` (empty) and the mock IdP fails closed
    # when it is unset. The session TTL is the absolute expiry in minutes (default 15).
    identity_signing_key: str = Field("", alias="CORA_IDENTITY_SIGNING_KEY")
    identity_session_ttl_minutes: int = Field(
        DEFAULT_SESSION_TTL_MINUTES, alias="CORA_IDENTITY_SESSION_TTL_MINUTES"
    )

    # Agent console API key (task 5.3, REQ-16). The static shared key the agent console sends on
    # the `/handoffs` read endpoints; it is a secret, so only its NAME ships in `.env.example`
    # (empty) and the endpoints fail CLOSED (401) when it is unset or does not match. This is the
    # LOCAL stand-in for a real agent principal (AWS Cognito group/role, design section 12 /
    # Phase 8); the key is replaced by a Cognito claim on AWS - a documented seam, not JWT/role
    # machinery here.
    agent_console_key: str = Field("", alias="CORA_AGENT_CONSOLE_KEY")

    # Observability (task 5.1, REQ-39). The per-turn JSONL trace exporter appends to
    # `<trace_dir>/traces.jsonl`; the dir lives under the gitignored `data/_state/` runtime area.
    trace_dir: Path = Field(DEFAULT_TRACE_DIR, alias="CORA_TRACE_DIR")

    # Customer Streamlit UI (task 5.4, REQ-05). Base URL of the FastAPI service the UI calls over
    # stdlib urllib; not a secret, so a real local default ships here. Point it at the deployed
    # service URL on AWS.
    api_base_url: str = Field("http://127.0.0.1:8000", alias="CORA_API_BASE_URL")

    # Organizer datathon bucket (read-only profile configured outside the repo)
    datathon_profile: str = Field("cora-datathon", alias="CORA_DATATHON_PROFILE")
    datathon_bucket: str = Field("", alias="CORA_DATATHON_BUCKET")
    datathon_region: str = Field("us-east-2", alias="CORA_DATATHON_REGION")

    # Cost assumptions (USD per 1K tokens); None means unknown, reported as n/a.
    price_input_per_1k: float | None = Field(None, alias="CORA_PRICE_INPUT_PER_1K")
    price_output_per_1k: float | None = Field(None, alias="CORA_PRICE_OUTPUT_PER_1K")

    @field_validator("price_input_per_1k", "price_output_per_1k", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # `.env.example` ships `CORA_PRICE_*=` blank; treat that as unknown, not as an error.
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("identity_session_ttl_minutes", mode="before")
    @classmethod
    def _blank_ttl_is_default(cls, value: object) -> object:
        # `.env.example` ships `CORA_IDENTITY_SESSION_TTL_MINUTES=` blank; fall back to the default.
        return DEFAULT_SESSION_TTL_MINUTES if isinstance(value, str) and not value.strip() else value

    @field_validator("trace_dir", mode="before")
    @classmethod
    def _blank_trace_dir_is_default(cls, value: object) -> object:
        # `.env.example` ships `CORA_TRACE_DIR=` blank; fall back to the default runtime dir.
        return DEFAULT_TRACE_DIR if isinstance(value, str) and not value.strip() else value

    @field_validator("api_base_url", mode="before")
    @classmethod
    def _blank_api_base_url_is_default(cls, value: object) -> object:
        # `.env.example` ships `CORA_API_BASE_URL=` blank; fall back to the local default.
        return "http://127.0.0.1:8000" if isinstance(value, str) and not value.strip() else value

    @property
    def bedrock_configured(self) -> bool:
        return bool(self.bedrock_model_id.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
