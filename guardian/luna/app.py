"""Flask application factory for the Luna safety backend."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from guardian.luna.alerts import initialize_firebase, send_alert_async
from guardian.luna.auth import issue_user_token, verify_bootstrap_token, verify_request_token
from guardian.luna.config import LunaConfig, load_config
from guardian.luna.geofence import is_out_of_bounds
from guardian.luna.safety import build_chat_alert_message, scan_message, toxicity_score
from guardian.luna.validation import (
    validate_auth_payload,
    validate_chat_payload,
    validate_location_payload,
)

logger = logging.getLogger(__name__)


def create_app(config: LunaConfig | None = None) -> Flask:
    """Build a Flask app from explicit or environment-backed configuration.

    Safe defaults:
    - binds to 127.0.0.1 unless configured otherwise
    - debug disabled unless ``LUNA_FLASK_DEBUG=true``
    - JWT auth optional via ``LUNA_REQUIRE_AUTH``
    """

    app_config = config or load_config()
    initialize_firebase(app_config.firebase_credentials_path)

    app = Flask(__name__)
    app.config["LUNA_CONFIG"] = app_config

    @app.route("/check_chat", methods=["POST"])
    def check_incoming():
        allowed, error = verify_request_token(request, app_config)
        if not allowed:
            return jsonify({"error": error}), 401

        valid, validation_error, payload = validate_chat_payload(request.get_json(silent=True))
        if not valid:
            return jsonify({"error": validation_error}), 400

        danger = scan_message(payload["message"])
        toxic = toxicity_score(payload["message"])
        if danger["is_flagged"] or toxic["toxic"]:
            alert_msg = build_chat_alert_message(payload["message"])
            send_alert_async(payload["parent_token"], alert_msg)
            return jsonify({"blocked": True, "details": {"danger": danger, "toxicity": toxic}}), 200

        return jsonify({"safe": True}), 200

    @app.route("/check_location", methods=["POST"])
    def track_location():
        allowed, error = verify_request_token(request, app_config)
        if not allowed:
            return jsonify({"error": error}), 401

        valid, validation_error, payload = validate_location_payload(request.get_json(silent=True))
        if not valid:
            return jsonify({"error": validation_error}), 400

        if is_out_of_bounds(
            payload["lat"],
            payload["lon"],
            safe_lat=app_config.safe_lat,
            safe_lon=app_config.safe_lon,
            radius_km=app_config.safe_radius_km,
        ):
            send_alert_async(
                payload["parent_token"],
                "Child outside configured safe zone. Review location in the Luna app.",
            )
            return jsonify({"alert": "Outside safe zone"}), 200

        return jsonify({"safe": True}), 200

    @app.route("/auth_kid", methods=["POST"])
    def generate_token():
        """Issue a short-lived JWT for demo clients (local development only)."""

        allowed, error = verify_bootstrap_token(request, app_config)
        if not allowed:
            return jsonify({"error": error}), 401

        valid, validation_error, user_id = validate_auth_payload(request.get_json(silent=True))
        if not valid:
            return jsonify({"error": validation_error}), 400

        token = issue_user_token(user_id, app_config)
        return jsonify({"token": token}), 200

    return app
