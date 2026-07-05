"""Tests for Luna configuration, safety helpers, and HTTP endpoints."""

import jwt
import pytest
from guardian.luna.app import create_app
from guardian.luna.config import LunaConfig, load_config
from guardian.luna.safety import build_chat_alert_message, scan_message, toxicity_score


@pytest.fixture
def luna_config() -> LunaConfig:
    return LunaConfig(
        secret_key="x" * 32,
        firebase_credentials_path=None,
        safe_lat=42.3314,
        safe_lon=-83.0458,
        safe_radius_km=5,
        flask_debug=False,
        flask_host="127.0.0.1",
        flask_port=5000,
        require_auth=False,
        auth_bootstrap_token="bootstrap-token",
    )


@pytest.fixture
def client(luna_config):
    app = create_app(luna_config)
    app.config["TESTING"] = True
    return app.test_client()


def test_load_config_requires_secret_key(monkeypatch):
    monkeypatch.delenv("LUNA_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="LUNA_SECRET_KEY"):
        load_config()


def test_scan_message_flags_grooming_language():
    result = scan_message("Can we meet alone at the hotel?")

    assert result["is_flagged"] is True
    assert result["score"] >= 1


def test_toxicity_score_uses_keyword_fallback_without_spacy():
    result = toxicity_score("you are stupid and I hate you")

    assert result["toxic"] is True


def test_build_chat_alert_message_truncates_preview():
    message = build_chat_alert_message("x" * 200, max_preview=50)

    assert len(message) < 200
    assert "..." in message


def test_check_chat_blocks_flagged_message(client):
    response = client.post(
        "/check_chat",
        json={"message": "send pic and meet me alone", "parent_token": "token"},
    )

    assert response.status_code == 200
    assert response.get_json()["blocked"] is True


def test_check_chat_rejects_empty_message(client):
    response = client.post("/check_chat", json={"message": "   "})

    assert response.status_code == 400


def test_check_location_alerts_outside_safe_zone(client):
    response = client.post(
        "/check_location",
        json={"lat": 40.0, "lon": -75.0, "parent_token": "token"},
    )

    assert response.status_code == 200
    assert response.get_json()["alert"] == "Outside safe zone"


def test_auth_kid_requires_bootstrap_token(client):
    response = client.post("/auth_kid", json={"user_id": "child-1"})

    assert response.status_code == 401


def test_auth_kid_issues_token_with_bootstrap_header(client, luna_config):
    response = client.post(
        "/auth_kid",
        json={"user_id": "child-1"},
        headers={"X-Luna-Bootstrap-Token": luna_config.auth_bootstrap_token},
    )

    assert response.status_code == 200
    token = response.get_json()["token"]
    payload = jwt.decode(token, luna_config.secret_key, algorithms=["HS256"])
    assert payload["user"] == "child-1"


def test_protected_routes_require_bearer_token(luna_config):
    protected_config = LunaConfig(
        secret_key=luna_config.secret_key,
        firebase_credentials_path=None,
        safe_lat=luna_config.safe_lat,
        safe_lon=luna_config.safe_lon,
        safe_radius_km=luna_config.safe_radius_km,
        flask_debug=False,
        flask_host="127.0.0.1",
        flask_port=5000,
        require_auth=True,
        auth_bootstrap_token=None,
    )
    app = create_app(protected_config)
    app.config["TESTING"] = True
    protected_client = app.test_client()

    response = protected_client.post("/check_chat", json={"message": "hello"})

    assert response.status_code == 401

    token = jwt.encode({"user": "child-1"}, protected_config.secret_key, algorithm="HS256")
    authorized = protected_client.post(
        "/check_chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert authorized.status_code == 200
