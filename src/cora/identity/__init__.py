"""Mock identity service for CORA (task 2.1, REQ-10/REQ-11, design sections 4/11).

A self-contained mock IdP that gates every customer-data read behind a one-time code (OTP)
step and an HMAC-signed session JWT. It exposes a small surface:

- `MockIdentityService` — begins OTP challenges, issues session tokens after the code is
  verified, and verifies tokens, failing closed on tampering or expiry.
- `Session` — the verified result of `verify_token`; carries the `customer_id` the rest of
  the system trusts. It never carries the raw token (tokens are never echoed to the LLM).
- `OtpChallenge` — the mock "sent" one-time code and its expiry.
- The identity exceptions (`IdentityError` and friends), all fail-closed by design.
"""

from __future__ import annotations

from cora.identity.service import (
    AuthenticationError,
    ExpiredTokenError,
    IdentityConfigError,
    IdentityError,
    InvalidTokenError,
    MockIdentityService,
    OtpChallenge,
    Session,
)

__all__ = [
    "AuthenticationError",
    "ExpiredTokenError",
    "IdentityConfigError",
    "IdentityError",
    "InvalidTokenError",
    "MockIdentityService",
    "OtpChallenge",
    "Session",
]
