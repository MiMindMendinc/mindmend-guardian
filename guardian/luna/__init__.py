"""Luna Safety Core - Child safety module for threat detection, geofencing, and alerts."""

from guardian.luna.app import create_app
from guardian.luna.auth import issue_user_token, verify_bootstrap_token, verify_request_token
from guardian.luna.config import LunaConfig, load_config
from guardian.luna.geofence import is_out_of_bounds
from guardian.luna.safety import build_chat_alert_message, scan_message, toxicity_score

__version__ = "0.2.0"

__all__ = [
    "LunaConfig",
    "build_chat_alert_message",
    "create_app",
    "is_out_of_bounds",
    "issue_user_token",
    "load_config",
    "scan_message",
    "toxicity_score",
    "verify_bootstrap_token",
    "verify_request_token",
]
