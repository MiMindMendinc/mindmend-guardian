"""Alert delivery helpers for Luna."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Thread

logger = logging.getLogger(__name__)

_firebase_initialized = False


def initialize_firebase(credentials_path: str | None) -> bool:
    """Initialize Firebase when credentials are configured."""

    global _firebase_initialized

    if _firebase_initialized:
        return True

    if not credentials_path:
        logger.info("Firebase credentials not configured; alerts remain local-only.")
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials

        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        logger.info("Firebase initialized.")
        return True
    except Exception as exc:
        logger.error("Firebase init failed: %s", exc)
        return False


def send_alert_async(
    parent_token: str,
    alert_msg: str,
    *,
    send_message: Callable[[str, str], None] | None = None,
) -> None:
    """Send an alert without blocking the request thread."""

    def _send() -> None:
        if send_message is not None:
            send_message(parent_token, alert_msg)
            return

        try:
            from firebase_admin import messaging
        except ImportError:
            logger.warning("Firebase unavailable. Alert suppressed without logging message body.")
            return

        if not _firebase_initialized:
            logger.warning("Firebase unavailable. Alert suppressed without logging message body.")
            return

        message = messaging.Message(
            notification=messaging.Notification(title="Luna Alert!", body=alert_msg),
            token=parent_token,
        )
        try:
            messaging.send(message)
        except Exception as exc:
            logger.error("Alert delivery failed: %s", exc)

    Thread(target=_send, daemon=True).start()
