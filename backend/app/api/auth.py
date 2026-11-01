from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass


class AuthConfigError(RuntimeError):
    pass


def hash_password(password: str, salt: str | None = None) -> str:
    salt_text = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt_text.encode("utf-8"),
        n=2**14,
        r=8,
        p=1,
    ).hex()
    return f"scrypt${salt_text}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$", 2)
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hash_password(password, salt).split("$", 2)[2]
    return hmac.compare_digest(candidate, digest)


@dataclass(frozen=True)
class AuthConfig:
    password_hash: str
    secret_key: str
    https_only: bool = False

    @classmethod
    def from_env(cls) -> AuthConfig:
        stored = os.environ.get("ASHARE_PASSWORD_HASH")
        password = os.environ.get("ASHARE_PASSWORD")
        if not stored and password:
            stored = hash_password(password)
        if not stored:
            raise AuthConfigError(
                "Set ASHARE_PASSWORD or ASHARE_PASSWORD_HASH; no default password is provided"
            )
        secret = os.environ.get("ASHARE_SECRET_KEY")
        if not secret:
            secret = secrets.token_urlsafe(32)
        return cls(
            password_hash=stored,
            secret_key=secret,
            https_only=os.environ.get("ASHARE_HTTPS_ONLY", "0") == "1",
        )

    def check(self, password: str) -> bool:
        return verify_password(password, self.password_hash)
