"""Privacy-boundary tests for Luna safety helpers."""

from guardian.core import stable_hash
from guardian.luna.config import LunaConfig
from guardian.luna.safety import build_chat_alert_message, scan_message


def test_luna_alert_message_omits_raw_chat_text():
    raw_text = "please meet me alone at the hotel after school"

    alert = build_chat_alert_message(raw_text)

    assert raw_text not in alert
    assert "meet me" not in alert
    assert "hotel" not in alert
    assert stable_hash(raw_text)[:16] in alert
    assert "Raw message content is intentionally omitted" in alert


def test_luna_scan_still_detects_risky_signal():
    result = scan_message("please meet me alone at the hotel")

    assert result["is_flagged"] is True
    assert result["score"] >= 1


def test_luna_config_can_require_auth_by_default_shape():
    config = LunaConfig(
        secret_key="test-secret",
        firebase_credentials_path=None,
        safe_lat=42.3314,
        safe_lon=-83.0458,
        safe_radius_km=5.0,
        flask_debug=False,
        flask_host="127.0.0.1",
        flask_port=5000,
        require_auth=True,
        auth_bootstrap_token=None,
    )

    assert config.require_auth is True
    assert config.flask_debug is False
    assert config.flask_host == "127.0.0.1"
