"""Mock identity provider: OTP step + HMAC-signed session JWTs (task 2.1, REQ-10/REQ-11).

Design sections 4 (state machine: `Unauthenticated -> Authenticated` only on a valid token
plus OTP) and 11 (security: the IdP issues an HMAC-signed JWT with `sub=customer_id`, `exp`
and a unique `jti`; tokens are never echoed to the LLM).

The service is deliberately small and deterministic so the agent and the test-suite can
drive it without a real IdP:

- `begin_authentication(customer_id)` runs the OTP step: it mints a one-time code and
  returns the `OtpChallenge` (the mock "delivers" the code). Supplying only a customer
  number / national ID this way is NOT access — no token exists until the code is verified.
- `complete_authentication(challenge_id, otp_code)` verifies the code and issues the signed
  session token. The challenge is single-use.
- `verify_token(token)` returns a `Session` carrying the verified `customer_id`, or FAILS
  CLOSED: on a tampered signature, a malformed token or expiry it raises and discloses
  nothing. The caller must then discard any pending confirmations and re-authenticate.

Security notes:
- The signing key is a secret read from `settings.get_settings()`; the service fails closed
  (`IdentityConfigError`) when it is unset. The key is never logged.
- `Session` intentionally does NOT hold the raw token, so a verified session can be passed
  around (and summarised to the LLM as `customer_id`) without ever exposing the credential.
- Time is injectable (`clock`) so expiry is tested by offsetting the clock, never by
  sleeping; `exp` is checked against that clock rather than wall-time directly.

Note on the TTL: `exp` is fixed at issuance, so this is an ABSOLUTE session TTL - the session
expires `session_ttl` after login regardless of activity. REQ-11 phrases the default as "15 min
idle"; sliding refresh-on-activity (which would make it a true idle timeout) belongs to the
session layer introduced in task 2.3 / the agent graph. Absolute expiry is strictly stricter
than idle expiry, so it cannot fail open; the vocabulary here is "session"/"absolute" to match
the behaviour.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from cora.settings import Settings, get_settings

# HMAC with SHA-256: a symmetric signature is all a single-process mock needs, and it matches
# design section 11 ("HMAC-signed JWT").
_ALGORITHM = "HS256"

# One-time code: 6 digits is the usual banking OTP shape. Short-lived by default.
_OTP_DIGITS = 6
DEFAULT_OTP_TTL = timedelta(minutes=5)


class IdentityError(Exception):
    """Base class for every identity failure. All are fail-closed by design."""


class IdentityConfigError(IdentityError):
    """The service is misconfigured (e.g. no signing key); it refuses to operate."""


class AuthenticationError(IdentityError):
    """The OTP step failed: unknown/expired challenge or a wrong code. No token is issued."""


class InvalidTokenError(IdentityError):
    """The token is malformed, has a bad signature, or is missing required claims."""


class ExpiredTokenError(IdentityError):
    """The token's TTL has elapsed; the session is over and state must be discarded."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class OtpChallenge:
    """A pending one-time-code challenge produced by the OTP step.

    In a real IdP the `code` would be delivered out-of-band (SMS/email) and never returned
    here; the mock returns it so tests and the demo can complete the flow. It is single-use
    and expires at `expires_at`.
    """

    challenge_id: str
    customer_id: str
    code: str
    expires_at: datetime


@dataclass(frozen=True)
class Session:
    """A verified session. Carries the `customer_id` the rest of the system trusts.

    Deliberately holds no raw token: identity, authorization and confirmations downstream
    use `customer_id` only, and the credential is never echoed to the LLM (design section 11).
    """

    customer_id: str
    jti: str
    issued_at: datetime
    expires_at: datetime


class MockIdentityService:
    """In-process mock IdP: OTP challenge, signed-token issuance, fail-closed verification."""

    def __init__(
        self,
        *,
        signing_key: str,
        session_ttl: timedelta,
        otp_ttl: timedelta = DEFAULT_OTP_TTL,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if not signing_key:
            # Fail closed: without a key we cannot sign or trust anything.
            raise IdentityConfigError("identity signing key is not configured")
        if session_ttl <= timedelta(0):
            raise IdentityConfigError("session TTL must be positive")
        self._signing_key = signing_key
        self._session_ttl = session_ttl
        self._otp_ttl = otp_ttl
        self._clock = clock
        self._pending: dict[str, OtpChallenge] = {}

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> MockIdentityService:
        """Build the service from runtime settings, failing closed if the key is unset."""
        settings = settings or get_settings()
        if not settings.identity_signing_key:
            raise IdentityConfigError(
                "CORA_IDENTITY_SIGNING_KEY is not set; the mock identity service refuses to start"
            )
        return cls(
            signing_key=settings.identity_signing_key,
            session_ttl=timedelta(minutes=settings.identity_session_ttl_minutes),
            clock=clock,
        )

    # -- OTP step -----------------------------------------------------------------------

    def begin_authentication(self, customer_id: str) -> OtpChallenge:
        """Start the OTP step for a customer and return the (mock-delivered) challenge.

        This is NOT access: it only proves a code was requested. Access requires the code to
        be returned to `complete_authentication` (REQ-10: an ID/number alone is refused).
        """
        if not customer_id or not customer_id.strip():
            raise AuthenticationError("a customer id is required to begin authentication")
        now = self._clock()
        challenge = OtpChallenge(
            challenge_id=uuid.uuid4().hex,
            customer_id=customer_id,
            code="".join(secrets.choice("0123456789") for _ in range(_OTP_DIGITS)),
            expires_at=now + self._otp_ttl,
        )
        self._pending[challenge.challenge_id] = challenge
        return challenge

    def complete_authentication(self, challenge_id: str, otp_code: str) -> str:
        """Verify the one-time code and, on success, issue a signed session token.

        The challenge is single-use (consumed whether or not the code matches) and fails
        closed on an unknown/expired challenge or a wrong/malformed code.
        """
        challenge = self._pending.pop(challenge_id, None)
        if challenge is None:
            raise AuthenticationError("unknown or already-used OTP challenge")
        if self._clock() >= challenge.expires_at:
            raise AuthenticationError("OTP challenge has expired")
        # Validate the input shape before comparing: `secrets.compare_digest` requires ASCII-only
        # str, so `None` or a non-ASCII code would raise TypeError (not an IdentityError) and
        # bypass the fail-closed contract. Reject it as a clean authentication failure instead.
        if not isinstance(otp_code, str) or not otp_code.isascii() or not otp_code.isdigit():
            raise AuthenticationError("malformed one-time code")
        # Constant-time compare so a wrong code leaks no timing signal.
        if not secrets.compare_digest(otp_code, challenge.code):
            raise AuthenticationError("incorrect one-time code")
        return self._issue_token(challenge.customer_id)

    # -- Token issuance / verification --------------------------------------------------

    def _issue_token(self, customer_id: str) -> str:
        now = self._clock()
        expires_at = now + self._session_ttl
        claims = {
            "sub": customer_id,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(claims, self._signing_key, algorithm=_ALGORITHM)

    def verify_token(self, token: str) -> Session:
        """Verify a session token, returning a `Session` or failing closed.

        Signature verification (tamper detection) is done by PyJWT; expiry is checked against
        the injected clock so it is deterministic in tests. Any problem discloses nothing.
        """
        if not isinstance(token, str) or not token:
            raise InvalidTokenError("no token supplied")
        try:
            # Verify the signature here; check `exp` ourselves against the service clock.
            claims = jwt.decode(
                token,
                self._signing_key,
                algorithms=[_ALGORITHM],
                options={"verify_exp": False, "require": ["sub", "exp", "jti"]},
            )
        except jwt.PyJWTError as exc:
            # Bad signature, malformed token or missing required claim: refuse, disclose nothing.
            raise InvalidTokenError("token is invalid") from exc

        # Type-check every claim we rely on BEFORE coercing it, so a correctly-signed token
        # carrying a non-numeric `exp`/`iat` fails closed as an InvalidTokenError instead of a
        # raw ValueError escaping the guard. (`bool` is a subclass of `int`, so exclude it.)
        customer_id = claims.get("sub")
        jti = claims.get("jti")
        exp = claims.get("exp")
        if not isinstance(customer_id, str) or not customer_id or not isinstance(jti, str) or not jti:
            raise InvalidTokenError("token is missing a required claim")
        if not isinstance(exp, (int, float)) or isinstance(exp, bool):
            raise InvalidTokenError("token has a malformed exp claim")

        now = self._clock()
        expires_at = datetime.fromtimestamp(int(exp), tz=UTC)
        if now >= expires_at:
            raise ExpiredTokenError("token has expired")

        iat = claims.get("iat")
        if iat is None:
            issued_at = now
        elif isinstance(iat, (int, float)) and not isinstance(iat, bool):
            issued_at = datetime.fromtimestamp(int(iat), tz=UTC)
        else:
            raise InvalidTokenError("token has a malformed iat claim")
        return Session(customer_id=customer_id, jti=jti, issued_at=issued_at, expires_at=expires_at)
