"""Runtime configuration (task 0.3, REQ-36, REQ-48).

All values come from environment variables or a gitignored `.env` file; `.env.example`
documents the names. No credentials live here: AWS access is resolved by boto3 from the
named SSO profile, so the settings only carry profile names, regions and model ids.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default session TTL (absolute, in minutes) for the mock identity service (REQ-11). Named once
# so the field default and the blank-template validator agree. REQ-11 phrases this as "idle";
# the mock implements an absolute expiry (stricter), with sliding refresh deferred to task 2.3.
DEFAULT_SESSION_TTL_MINUTES = 15


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

    @property
    def bedrock_configured(self) -> bool:
        return bool(self.bedrock_model_id.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
