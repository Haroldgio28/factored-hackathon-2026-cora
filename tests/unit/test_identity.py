"""Tests for task 2.1: mock identity service (REQ-10, REQ-11).

Covers the four behaviours the task calls out: a valid issue+verify round-trip, a tampered
token rejected, an expired token rejected (via an offset clock, never `sleep`), and the
"ID only" refusal (a bare customer number / national ID with no valid session yields a
refusal, not data access). Follows the `test_settings.py` conventions: pytest + monkeypatch,
isolated env, repo-root resolution.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import pytest

from cora.identity import (
    AuthenticationError,
    ExpiredTokenError,
    IdentityConfigError,
    InvalidTokenError,
    MockIdentityService,
    Session,
)
from cora.settings import Settings

_KEY = "test-signing-key-not-a-real-secret"
_CUSTOMER = "CUST-000123"


class _Clock:
    """A controllable clock: tests advance `now` instead of sleeping."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _make_service(clock: _Clock | None = None, ttl_minutes: int = 15) -> MockIdentityService:
    return MockIdentityService(
        signing_key=_KEY,
        session_ttl=timedelta(minutes=ttl_minutes),
        clock=clock or _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC)),
    )


def _authenticate(service: MockIdentityService, customer_id: str = _CUSTOMER) -> str:
    """Run the full OTP flow and return the issued session token."""
    challenge = service.begin_authentication(customer_id)
    return service.complete_authentication(challenge.challenge_id, challenge.code)


def test_issue_and_verify_round_trip() -> None:
    service = _make_service()
    token = _authenticate(service)
    session = service.verify_token(token)
    assert isinstance(session, Session)
    assert session.customer_id == _CUSTOMER
    assert session.jti  # a unique id is present
    assert session.expires_at > session.issued_at


def test_each_token_has_a_unique_jti() -> None:
    service = _make_service()
    first = service.verify_token(_authenticate(service))
    second = service.verify_token(_authenticate(service))
    assert first.jti != second.jti


def test_session_never_carries_the_raw_token() -> None:
    # Tokens must never be echoed to the LLM; the Session surface only exposes customer_id.
    service = _make_service()
    token = _authenticate(service)
    session = service.verify_token(token)
    assert token not in repr(session)
    assert not hasattr(session, "token")


def test_tampered_token_is_rejected() -> None:
    service = _make_service()
    token = _authenticate(service)
    # Flip the last character of the signature segment -> signature no longer matches.
    head, payload, signature = token.split(".")
    swapped = "A" if signature[-1] != "A" else "B"
    tampered = f"{head}.{payload}.{signature[:-1]}{swapped}"
    with pytest.raises(InvalidTokenError):
        service.verify_token(tampered)


def test_token_signed_with_another_key_is_rejected() -> None:
    # A token forged with a different key must fail closed (tamper/forgery).
    forged = jwt.encode(
        {"sub": _CUSTOMER, "exp": 9999999999, "jti": "x", "iat": 1},
        "a-different-key",
        algorithm="HS256",
    )
    with pytest.raises(InvalidTokenError):
        _make_service().verify_token(forged)


@pytest.mark.parametrize("claim", ["exp", "iat"])
def test_non_numeric_time_claim_is_rejected(claim: str) -> None:
    # A correctly-signed token carrying a non-numeric exp/iat must fail closed as an
    # InvalidTokenError, not escape the guard as a raw ValueError.
    claims = {"sub": _CUSTOMER, "exp": 9999999999, "iat": 1, "jti": "x"}
    claims[claim] = "not-a-number"
    forged = jwt.encode(claims, _KEY, algorithm="HS256")
    with pytest.raises(InvalidTokenError):
        _make_service().verify_token(forged)


def test_expired_token_is_rejected() -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    service = _make_service(clock=clock, ttl_minutes=15)
    token = _authenticate(service)
    # Advance past the session TTL without sleeping.
    clock.now = clock.now + timedelta(minutes=16)
    with pytest.raises(ExpiredTokenError):
        service.verify_token(token)


def test_token_valid_just_before_expiry() -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    service = _make_service(clock=clock, ttl_minutes=15)
    token = _authenticate(service)
    clock.now = clock.now + timedelta(minutes=14)
    assert service.verify_token(token).customer_id == _CUSTOMER


def test_id_only_is_refused_not_access() -> None:
    # REQ-10: a bare customer number / national ID is not a session. Verifying it as a token
    # must refuse (raise), never return a Session granting data access.
    service = _make_service()
    for bare_id in (_CUSTOMER, "12345678", ""):
        with pytest.raises(InvalidTokenError):
            service.verify_token(bare_id)


def test_begin_authentication_alone_grants_no_token() -> None:
    # Starting the OTP step (what "having an ID" lets you do) produces a challenge, not a
    # session; there is no way to a Session without completing the code.
    service = _make_service()
    challenge = service.begin_authentication(_CUSTOMER)
    assert challenge.customer_id == _CUSTOMER
    assert challenge.code  # mock delivers the code
    assert not isinstance(challenge, Session)


def test_wrong_otp_code_is_refused() -> None:
    service = _make_service()
    challenge = service.begin_authentication(_CUSTOMER)
    with pytest.raises(AuthenticationError):
        service.complete_authentication(
            challenge.challenge_id, "000000" if challenge.code != "000000" else "111111"
        )


@pytest.mark.parametrize("bad_code", [None, "12345", "abcdef", "１２３４５６", "   "])
def test_malformed_otp_code_fails_closed(bad_code: object) -> None:
    # A non-str / non-ASCII / non-digit code must raise AuthenticationError (an IdentityError),
    # never a raw TypeError from compare_digest, so callers catching IdentityError fail closed.
    service = _make_service()
    challenge = service.begin_authentication(_CUSTOMER)
    with pytest.raises(AuthenticationError):
        service.complete_authentication(challenge.challenge_id, bad_code)  # type: ignore[arg-type]


def test_otp_challenge_is_single_use() -> None:
    service = _make_service()
    challenge = service.begin_authentication(_CUSTOMER)
    service.complete_authentication(challenge.challenge_id, challenge.code)
    with pytest.raises(AuthenticationError):
        service.complete_authentication(challenge.challenge_id, challenge.code)


def test_expired_otp_challenge_is_refused() -> None:
    clock = _Clock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    service = _make_service(clock=clock)
    challenge = service.begin_authentication(_CUSTOMER)
    clock.now = clock.now + timedelta(minutes=6)  # past the OTP TTL
    with pytest.raises(AuthenticationError):
        service.complete_authentication(challenge.challenge_id, challenge.code)


def test_service_fails_closed_without_a_signing_key() -> None:
    with pytest.raises(IdentityConfigError):
        MockIdentityService(signing_key="", session_ttl=timedelta(minutes=15))


def test_from_settings_fails_closed_without_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Isolate from the developer's real env vars AND a checkout-local `.env`: pydantic-settings
    # resolves `env_file=".env"` against the cwd, so chdir to an empty tmp dir (mirrors the
    # `clean_env` fixture in test_settings.py). Without this the test passes only in a worktree
    # that happens to have no `.env`.
    for alias in (f.alias for f in Settings.model_fields.values()):
        monkeypatch.delenv(alias, raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings()  # no signing key configured
    with pytest.raises(IdentityConfigError):
        MockIdentityService.from_settings(settings)


def test_from_settings_uses_configured_key_and_default_ttl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for alias in (f.alias for f in Settings.model_fields.values()):
        monkeypatch.delenv(alias, raising=False)
    monkeypatch.chdir(tmp_path)  # ignore any checkout-local `.env` TTL override
    monkeypatch.setenv("CORA_IDENTITY_SIGNING_KEY", _KEY)
    settings = Settings()
    assert settings.identity_session_ttl_minutes == 15  # default session TTL (REQ-11)
    service = MockIdentityService.from_settings(settings)
    session = service.verify_token(_authenticate(service))
    assert session.customer_id == _CUSTOMER
