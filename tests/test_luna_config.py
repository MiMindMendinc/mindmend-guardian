"""Tests for Luna configuration validation."""

import pytest
from guardian.luna.config import load_config


def test_load_config_rejects_short_secret_key(monkeypatch):
    monkeypatch.setenv("LUNA_SECRET_KEY", "too-short")

    with pytest.raises(ValueError, match="at least 32 characters"):
        load_config()
