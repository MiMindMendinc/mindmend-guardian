"""Shared pytest configuration."""

import os

os.environ.setdefault("LUNA_SECRET_KEY", "pytest-secret-key-thirty-two-chars-min")
