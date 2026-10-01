"""Tests for task 0.3: configuration from env vars, `.env.example` with names only (REQ-36, REQ-48)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cora.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# Shapes of real credentials that must never appear in the template.
SECRET_PATTERNS = [
    re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"(?i)aws_secret_access_key"),
    re.compile(r"(?i)aws_session_token"),
    re.compile(r"(?i)(password|secret|token)\s*=\s*\S+"),
]


def _example_keys() -> dict[str, str]:
    pairs = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
    return pairs


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """Isolate from the developer's real env vars and `.env`."""
    for alias in (f.alias for f in Settings.model_fields.values()):
        monkeypatch.delenv(alias, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_defaults_without_env(clean_env: pytest.MonkeyPatch) -> None:
    s = Settings()
    assert s.aws_profile == "cora-dev"
    assert s.aws_region == "us-east-1"
    assert s.datathon_profile == "cora-datathon"
    assert not s.bedrock_configured
    assert s.price_input_per_1k is None


def test_env_overrides(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("AWS_PROFILE", "other-profile")
    clean_env.setenv("CORA_BEDROCK_MODEL_ID", "us.example.model-v1:0")
    clean_env.setenv("CORA_PRICE_INPUT_PER_1K", "0.003")
    s = Settings()
    assert s.aws_profile == "other-profile"
    assert s.bedrock_configured
    assert s.price_input_per_1k == pytest.approx(0.003)


def test_env_example_loads_as_dotenv(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The shipped template (blank model ids and prices) must parse without errors.
    (tmp_path / ".env").write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    s = Settings()
    assert not s.bedrock_configured
    assert s.price_output_per_1k is None


def test_env_example_matches_settings() -> None:
    aliases = {f.alias for f in Settings.model_fields.values()}
    assert set(_example_keys()) == aliases


def test_env_example_has_no_secrets() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    hits = [p.pattern for p in SECRET_PATTERNS if p.search(text)]
    assert not hits, f".env.example looks like it contains credentials: {hits}"
    # Model ids must be configured per environment, never shipped.
    keys = _example_keys()
    assert all(not v for k, v in keys.items() if k.startswith("CORA_BEDROCK_"))
