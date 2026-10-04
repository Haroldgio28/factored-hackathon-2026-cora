"""Tests for the Bedrock Converse client's request shaping (task 4.2, ADR-005).

The one behaviour that is invisible to the generator tests (which stop at `StubLLMClient`) and
cost a real-Bedrock run to find: Converse REJECTS a `system` block whose text is empty with a
`ParamValidationError`, so `BedrockLLMClient` must OMIT the field entirely when the caller passes
no system prompt (the generator always calls `complete(system="")`) and include it, with the
right shape, when there is one. These tests pin that with a fake `converse` that captures its
kwargs - no boto3 session, no network, no AWS.
"""

from __future__ import annotations

from cora.agent.llm import BedrockLLMClient
from cora.settings import Settings


class _FakeConverseClient:
    """Captures the kwargs of the last `converse` call and returns a minimal valid response."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def converse(self, **kwargs: object) -> dict:
        self.last_kwargs = kwargs
        return {"output": {"message": {"content": [{"text": "ok"}]}}}


def _client_with_fake() -> tuple[BedrockLLMClient, _FakeConverseClient]:
    """A `BedrockLLMClient` with its boto3 client swapped for the capturing fake (no AWS).

    `__new__` skips `__init__` (which would build a real boto3 session); we set only the two
    attributes `_converse` reads: the settings (for the model id) and the fake client.
    """
    client = BedrockLLMClient.__new__(BedrockLLMClient)
    # `_env_file=None` isolates the test from the developer's .env so the model id is deterministic.
    # Settings fields use env-var aliases and `populate_by_name` is off, so construct by alias.
    client._settings = Settings(CORA_BEDROCK_MODEL_ID="test-model-id", AWS_PROFILE="", _env_file=None)
    fake = _FakeConverseClient()
    client._client = fake
    return client, fake


def test_empty_system_is_omitted_from_converse() -> None:
    """REQ-40/ADR-005: an empty system prompt must NOT be sent (Bedrock rejects an empty block)."""
    client, fake = _client_with_fake()
    client.complete(system="", user="hola")
    assert fake.last_kwargs is not None
    assert "system" not in fake.last_kwargs


def test_whitespace_only_system_is_omitted_from_converse() -> None:
    """A whitespace-only system prompt is treated as empty and omitted."""
    client, fake = _client_with_fake()
    client.complete(system="   \n\t ", user="hola")
    assert fake.last_kwargs is not None
    assert "system" not in fake.last_kwargs


def test_non_empty_system_is_present_with_the_expected_shape() -> None:
    """A real system prompt is sent as the Converse `system` list-of-text-blocks."""
    client, fake = _client_with_fake()
    client.complete(system="You are helpful.", user="hola")
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["system"] == [{"text": "You are helpful."}]


def test_converse_carries_the_configured_model_id_and_user_text() -> None:
    """The model id comes from settings only (never hard-coded) and the user text is forwarded."""
    client, fake = _client_with_fake()
    reply = client.complete(system="", user="¿cuál es mi saldo?")
    assert reply == "ok"
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["modelId"] == "test-model-id"
    assert fake.last_kwargs["messages"] == [{"role": "user", "content": [{"text": "¿cuál es mi saldo?"}]}]
