# MindMend Guardian - Developer Guidelines

## Project Overview

**MindMend Guardian** is a privacy-first, local-first youth safety prototype from
Michigan MindMend Inc. It includes a rules engine, Luna safety API, synthetic
CLI/dashboard demos, and an optional voice runtime. The project is MIT-licensed
open source and is **not** a clinical or crisis service.

## Tech Stack

- **Python 3.11+** — core language and packaging (`pyproject.toml`)
- **Flask + PyJWT** — Luna API backend (optional `[api]` extra)
- **Streamlit** — optional dashboard demo (`[simulator]` extra)
- **pytest + ruff + pip-audit** — tests, lint/format, dependency audit
- **GitHub Actions** — CI on every push/PR

Voice runtime dependencies (`[voice]` extra) are heavy and lazy-loaded in
`guardian/mindmend_guardian.py` so imports remain safe for CI.

## Preferred Coding Style and Conventions

- Use idiomatic Python with type hints
- Follow PEP 8 via `ruff` (configured in `pyproject.toml`)
- Prefer modular components for maintainability
- Unit-test-first for safety-sensitive logic
- Document public module APIs with clear docstrings
- Keep functions focused and single-purpose

## Important Rules

**Never introduce:**

- Secrets, credentials, or committed `.env` files
- Real youth/family/private data in code, tests, or docs
- Telemetry or undisclosed external data collection
- Over-claims about clinical validation or compliance

**Always:**

- Add tests for safety-sensitive behavior changes
- Run `ruff` and `pytest` before submitting PRs
- Update docs and `CHANGELOG.md` for user-visible changes
- Report security issues via [`SECURITY.md`](../SECURITY.md)

## Setup & Developer Workflow

```bash
git clone https://github.com/MiMindMendinc/mindmend-guardian.git
cd mindmend-guardian
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

Common commands:

```bash
pytest --cov=guardian --cov-report=term-missing
ruff check guardian tests
ruff format --check guardian tests
python -m guardian.demo --json
python -m guardian.dashboard   # requires [simulator] extra
```

See [`SETUP.md`](../SETUP.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md) for full guidance.

## Architecture & Module Notes

### Core rules engine
`guardian/core.py` — import-safe risk evaluation and audit hashing.

### Luna Safety Core
Modular Flask backend under `guardian/luna/`:

- **auth** — JWT bearer verification and bootstrap token issuance
- **validation** — request payload validation
- **safety** — chat scanning and toxicity helpers
- **geofence** — location bounds checks
- **alerts** — optional Firebase delivery (disabled without credentials)

### Demo surfaces
- `python -m guardian.demo` — synthetic CLI demo
- `python -m guardian.dashboard` — optional Streamlit dashboard

### Experimental subprojects
- `ani-2027/` — experimental Tauri/React UI prototype (not required for core CI)
- `perrien-simulator/` — optional simulator dashboard

These subprojects inherit the repository MIT license but are not production-ready.

## Ethics & False-Positive Guidance

- Minimize false positives to maintain trust
- Design compassionate, non-alarmist responses
- Human-in-the-loop for serious alerts
- Respect privacy — default audit output uses hashes, not raw chat text

## Security & Privacy

- Safe Luna defaults: bind to `127.0.0.1`, debug off, JWT auth on by default
- Report security issues privately via [`SECURITY.md`](../SECURITY.md)
- Follow the [Code of Conduct](../CODE_OF_CONDUCT.md)

**Questions?** See [`SUPPORT.md`](../SUPPORT.md) or open a structured GitHub issue.
