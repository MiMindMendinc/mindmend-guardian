"""Flask application factory for the Luna safety backend."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging

import jwt
from flask import Flask, jsonify, request

from guardian.luna.alerts import initialize_firebase, send_alert_async
from guardian.luna.config import LunaConfig, load_config
from guardian.luna.safety import (
    build_chat_alert_message,
    is_out_of_bounds,
    scan_message,
    toxicity_score,
)

logger = logging.getLogger(__name__)


def _extract_bearer_token() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ").strip()
    return None


def _verify_request_token(config: LunaConfig) -> tuple[bool, str | None]:
    if not config.require_auth:
        return True, None

    token = _extract_bearer_token()
    if not token:
        return False, "Missing bearer token"

    try:
        jwt.decode(token, config.secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        return False, f"Invalid token: {exc}"

    return True, None


def create_app(config: LunaConfig | None = None) -> Flask:
    """Build a Flask app from explicit or environment-backed configuration."""

    app_config = config or load_config()
    initialize_firebase(app_config.firebase_credentials_path)

    app = Flask(__name__)
    app.config["LUNA_CONFIG"] = app_config

    @app.route("/check_chat", methods=["POST"])
    def check_incoming():
        allowed, error = _verify_request_token(app_config)
        if not allowed:
            return jsonify({"error": error}), 401

        data = request.get_json(silent=True) or {}
        text = data.get("message", "").strip()
        parent_token = data.get("parent_token", "")
        if not text:
            return jsonify({"error": "Missing message"}), 400

        danger = scan_message(text)
        toxic = toxicity_score(text)
        if danger["is_flagged"] or toxic["toxic"]:
            alert_msg = build_chat_alert_message(text)
            send_alert_async(parent_token, alert_msg)
            return jsonify({"blocked": True, "details": {"danger": danger, "toxicity": toxic}}), 200

        return jsonify({"safe": True}), 200

    @app.route("/check_location", methods=["POST"])
    def track_location():
        allowed, error = _verify_request_token(app_config)
        if not allowed:
            return jsonify({"error": error}), 401

        data = request.get_json(silent=True) or {}
        lat = data.get("lat")
        lon = data.get("lon")
        parent_token = data.get("parent_token", "")
        if lat is None or lon is None:
            return jsonify({"error": "Missing coords"}), 400

        if is_out_of_bounds(
            lat,
            lon,
            safe_lat=app_config.safe_lat,
            safe_lon=app_config.safe_lon,
            radius_km=app_config.safe_radius_km,
        ):
            send_alert_async(
                parent_token,
                "Child outside configured safe zone. Review location in the Luna app.",
            )
            return jsonify({"alert": "Outside safe zone"}), 200

        return jsonify({"safe": True}), 200

    @app.route("/auth_kid", methods=["POST"])
    def generate_token():
        """Issue a short-lived JWT for demo clients.

        When ``LUNA_AUTH_BOOTSTRAP_TOKEN`` is configured, callers must present it
        in the ``X-Luna-Bootstrap-Token`` header. This endpoint is intended for
        local development only, not unsupervised production use.
        """

        if app_config.auth_bootstrap_token:
            provided = request.headers.get("X-Luna-Bootstrap-Token", "")
            if provided != app_config.auth_bootstrap_token:
                return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        payload = {
            "user": user_id,
            "exp": datetime.now(UTC) + timedelta(days=1),
        }
        token = jwt.encode(payload, app_config.secret_key, algorithm="HS256")
        return jsonify({"token": token}), 200

    return app
