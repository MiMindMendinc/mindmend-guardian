"""JWT authentication helpers for the Luna safety backend."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from flask import Request

from guardian.luna.config import LunaConfig


def extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ").strip()
    return None


def verify_request_token(request: Request, config: LunaConfig) -> tuple[bool, str | None]:
    """Validate bearer JWT when ``config.require_auth`` is enabled."""

    if not config.require_auth:
        return True, None

    token = extract_bearer_token(request)
    if not token:
        return False, "Missing bearer token"

    try:
        jwt.decode(token, config.secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        return False, f"Invalid token: {exc}"

    return True, None


def verify_bootstrap_token(request: Request, config: LunaConfig) -> tuple[bool, str | None]:
    if not config.auth_bootstrap_token:
        return True, None

    provided = request.headers.get("X-Luna-Bootstrap-Token", "")
    if provided != config.auth_bootstrap_token:
        return False, "Unauthorized"
    return True, None


def issue_user_token(user_id: str, config: LunaConfig) -> str:
    payload = {
        "user": user_id,
        "exp": datetime.now(UTC) + timedelta(days=1),
    }
    return jwt.encode(payload, config.secret_key, algorithm="HS256")
