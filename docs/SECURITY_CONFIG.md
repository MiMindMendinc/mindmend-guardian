# Security Configuration

This document describes environment variables, safe defaults, and authentication behavior for MindMend Guardian runtimes.

## Core demo (no secrets required)

```bash
python -m guardian.demo
```

The CLI demo uses synthetic sample text only and does not start network services.

## Luna API configuration

Copy [`.env.example`](../.env.example) and set values before starting the API:

```bash
export LUNA_SECRET_KEY="your-long-random-secret-at-least-32-characters"
python guardian/luna/luna_safety_core.py
```

### Required variables

| Variable | Description |
|----------|-------------|
| `LUNA_SECRET_KEY` | HS256 signing key (**32+ characters**) |

### Safe defaults

| Variable | Default | Notes |
|----------|---------|-------|
| `LUNA_FLASK_HOST` | `127.0.0.1` | Avoid `0.0.0.0` unless you understand exposure |
| `LUNA_FLASK_DEBUG` | `false` | Never enable debug on shared networks |
| `LUNA_REQUIRE_AUTH` | `true` | Set `false` only for deliberate local-only demos |
| `LUNA_FIREBASE_CREDENTIALS` | unset | Optional path to Firebase service account JSON |

### Optional hardening

| Variable | Purpose |
|----------|---------|
| `LUNA_AUTH_BOOTSTRAP_TOKEN` | Protects `/auth_kid` token issuance in local dev |
| `LUNA_SAFE_LAT` / `LUNA_SAFE_LON` / `LUNA_SAFE_RADIUS_KM` | Geofence defaults |

## Authentication model (prototype)

- `/check_chat` and `/check_location` accept POST JSON payloads.
- Safety endpoints require JWT bearer auth by default (`LUNA_REQUIRE_AUTH=true`)
- Opt out with `LUNA_REQUIRE_AUTH=false` only for deliberate local-only demos
- When auth is enabled, callers must send `Authorization: Bearer <jwt>`.
- `/auth_kid` is POST-only and intended for local development demos.
- Bootstrap token header: `X-Luna-Bootstrap-Token`

This is **not** a complete identity platform. Real deployments need professional auth design, rate limits, and monitoring.

## Logging and sensitive data

Safe defaults in this repository:

- audit events hash input text instead of storing raw content
- Luna alert messages truncate chat previews and avoid exact coordinates in alert bodies
- Firebase alerts are optional and disabled when credentials are absent

Verify your deployment logging separately before handling real family data.

## Dependency groups

Install only what you need:

```bash
pip install -e ".[dev]"        # tests + lint + audit tools
pip install -e ".[api,dev]"    # Luna API + tests
pip install -e ".[voice]"        # voice runtime
pip install -e ".[simulator]"   # dashboard + simulator
```

## CI security checks

GitHub Actions runs:

- `ruff` lint/format checks
- `pytest`
- `pip-audit` on installed dependencies
- secret-pattern grep over `guardian/`, `tests/`, and `docs/`

## Responsible disclosure

See [`SECURITY.md`](../SECURITY.md) for private reporting instructions.
