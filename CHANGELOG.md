# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MIT open source governance upgrades: Contributor Covenant, issue templates,
  Dependabot, CODEOWNERS, CHANGELOG, release docs, and pre-commit hooks
- CI coverage reporting for the `guardian` package

## [0.3.0] - 2026-07-05

### Added

- Local CLI demo: `python -m guardian.demo` with synthetic sample messages
- Optional Streamlit dashboard: `python -m guardian.dashboard`
- Modular Luna services: auth, validation, geofence, alerts, and safety helpers
- Import-safe voice runtime with lazy-loaded heavy dependencies
- Expanded pytest coverage for core, Luna, demo, scenarios, validation, and privacy
- CI quality gates: pytest, ruff lint/format, pip-audit, secret-pattern scan, demo smoke test
- Documentation set: ARCHITECTURE, API, SECURITY_CONFIG, DEMO, STATUS, ROADMAP

### Changed

- Python 3.11 baseline with optional dependency extras (`core`, `api`, `voice`, `simulator`, `dev`)
- Luna JWT auth enabled by default (`LUNA_REQUIRE_AUTH=true`)
- Privacy-preserving alert messages using content hashes instead of raw chat previews
- Safe server defaults: bind to `127.0.0.1`, debug disabled

### Security

- Require 32+ character `LUNA_SECRET_KEY` for JWT signing
- Request validation on Luna API payloads
- Secret-pattern scan in CI over Python sources

[Unreleased]: https://github.com/MiMindMendinc/mindmend-guardian/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MiMindMendinc/mindmend-guardian/releases/tag/v0.3.0
