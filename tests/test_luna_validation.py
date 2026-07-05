"""Validation and auth rejection tests for Luna endpoints."""

import jwt
import pytest
from guardian.luna.app import create_app
from guardian.luna.config import LunaConfig


@pytest.fixture
def protected_config() -> LunaConfig:
    return LunaConfig(
        secret_key="x" * 32,
        firebase_credentials_path=None,
        safe_lat=42.3314,
        safe_lon=-83.0458,
        safe_radius_km=5,
        flask_debug=False,
        flask_host="127.0.0.1",
        flask_port=5000,
        require_auth=True,
        auth_bootstrap_token="bootstrap-token",
    )


@pytest.fixture
def protected_client(protected_config):
    app = create_app(protected_config)
    app.config["TESTING"] = True
    return app.test_client()


def test_check_chat_rejects_oversized_message(protected_client, protected_config):
    token = jwt.encode({"user": "child-1"}, protected_config.secret_key, algorithm="HS256")
    response = protected_client.post(
        "/check_chat",
        json={"message": "a" * 5000},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400


def test_check_chat_rejects_non_string_message(protected_client, protected_config):
    token = jwt.encode({"user": "child-1"}, protected_config.secret_key, algorithm="HS256")
    response = protected_client.post(
        "/check_chat",
        json={"message": 123},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400


def test_check_location_rejects_missing_coordinates(protected_client, protected_config):
    token = jwt.encode({"user": "child-1"}, protected_config.secret_key, algorithm="HS256")
    response = protected_client.post(
        "/check_location",
        json={"lat": 42.0},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400


def test_auth_kid_rejects_missing_user_id(protected_client):
    response = protected_client.post(
        "/auth_kid",
        json={},
        headers={"X-Luna-Bootstrap-Token": "bootstrap-token"},
    )

    assert response.status_code == 400
