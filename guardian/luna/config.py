"""Environment-driven configuration for the Luna safety backend."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LunaConfig:
    """Runtime settings loaded from environment variables."""

    secret_key: str
    firebase_credentials_path: str | None
    safe_lat: float
    safe_lon: float
    safe_radius_km: float
    flask_debug: bool
    flask_host: str
    flask_port: int
    require_auth: bool
    auth_bootstrap_token: str | None


def load_config() -> LunaConfig:
    """Load Luna settings from the environment.

    ``LUNA_SECRET_KEY`` is required for any runtime that signs or verifies JWTs.
    """

    secret_key = os.environ.get("LUNA_SECRET_KEY", "").strip()
    if not secret_key:
        raise ValueError(
            "LUNA_SECRET_KEY is required. Set it in your environment or .env file."
        )

    firebase_path = os.environ.get("LUNA_FIREBASE_CREDENTIALS", "").strip() or None

    return LunaConfig(
        secret_key=secret_key,
        firebase_credentials_path=firebase_path,
        safe_lat=float(os.environ.get("LUNA_SAFE_LAT", "42.3314")),
        safe_lon=float(os.environ.get("LUNA_SAFE_LON", "-83.0458")),
        safe_radius_km=float(os.environ.get("LUNA_SAFE_RADIUS_KM", "5")),
        flask_debug=_env_bool("LUNA_FLASK_DEBUG", default=False),
        flask_host=os.environ.get("LUNA_FLASK_HOST", "127.0.0.1"),
        flask_port=int(os.environ.get("LUNA_FLASK_PORT", "5000")),
        require_auth=_env_bool("LUNA_REQUIRE_AUTH", default=False),
        auth_bootstrap_token=os.environ.get("LUNA_AUTH_BOOTSTRAP_TOKEN", "").strip() or None,
    )
