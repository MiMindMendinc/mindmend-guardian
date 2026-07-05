"""Request validation helpers for Luna HTTP endpoints."""

from __future__ import annotations

from typing import Any

MAX_MESSAGE_LENGTH = 4000
MAX_USER_ID_LENGTH = 128


def validate_chat_payload(data: dict[str, Any] | None) -> tuple[bool, str | None, dict[str, Any]]:
    payload = data or {}
    message = payload.get("message", "")
    if not isinstance(message, str):
        return False, "message must be a string", payload

    text = message.strip()
    if not text:
        return False, "Missing message", payload
    if len(text) > MAX_MESSAGE_LENGTH:
        return False, f"message exceeds {MAX_MESSAGE_LENGTH} characters", payload

    parent_token = payload.get("parent_token", "")
    if parent_token is not None and not isinstance(parent_token, str):
        return False, "parent_token must be a string", payload

    return True, None, {"message": text, "parent_token": str(parent_token or "")}


def validate_location_payload(
    data: dict[str, Any] | None,
) -> tuple[bool, str | None, dict[str, Any]]:
    payload = data or {}
    lat = payload.get("lat")
    lon = payload.get("lon")
    if lat is None or lon is None:
        return False, "Missing coords", payload

    parent_token = payload.get("parent_token", "")
    if parent_token is not None and not isinstance(parent_token, str):
        return False, "parent_token must be a string", payload

    return True, None, {"lat": lat, "lon": lon, "parent_token": str(parent_token or "")}


def validate_auth_payload(data: dict[str, Any] | None) -> tuple[bool, str | None, str]:
    payload = data or {}
    user_id = payload.get("user_id")
    if not user_id or not isinstance(user_id, str):
        return False, "Missing user_id", ""
    user_id = user_id.strip()
    if not user_id:
        return False, "Missing user_id", ""
    if len(user_id) > MAX_USER_ID_LENGTH:
        return False, f"user_id exceeds {MAX_USER_ID_LENGTH} characters", ""
    return True, None, user_id
