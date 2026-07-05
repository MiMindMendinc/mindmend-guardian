"""Tests for Luna configuration validation."""

import pytest
from guardian.luna.config import load_config


def test_load_config_rejects_short_secret_key(monkeypatch):
    monkeypatch.setenv("LUNA_SECRET_KEY", "too-short")

    with pytest.raises(ValueError, match="at least 32 characters"):
        load_config()


def test_load_config_defaults_require_auth_true(monkeypatch):
    monkeypatch.setenv("LUNA_SECRET_KEY", "x" * 32)
    monkeypatch.delenv("LUNA_REQUIRE_AUTH", raising=False)

    config = load_config()

    assert config.require_auth is True
    assert config.flask_debug is False
    assert config.flask_host == "127.0.0.1"
