# MindMend Guardian Roadmap

MindMend Guardian is being hardened from a mission-driven prototype into a recruiter-ready, sponsor-ready, contributor-ready privacy-first youth safety portfolio project.

## Completed (v0.3)

- [x] Import-safe core safety helpers with tests
- [x] Local CLI demo (`python -m guardian.demo`)
- [x] Optional Streamlit dashboard demo
- [x] Luna API modularization and env-based security config
- [x] Expanded pytest coverage (core, Luna, demo, scenarios)
- [x] CI with linting, dependency audit, and secret-pattern scan
- [x] Architecture, API, security config, and demo documentation

## Near-term

- [ ] Unify Luna grooming checks with `guardian/core.py` rules
- [ ] Wire voice runtime transcripts into `assess_text()`
- [ ] Parent/guardian review workflow mock
- [ ] Encrypted local audit log example (SQLite)
- [ ] Sensitivity configuration file (YAML/TOML)
- [ ] Demo screenshots and short walkthrough GIF
- [ ] Docker and Raspberry Pi deployment notes

## Safety hardening

- [ ] Rate limiting and request size enforcement at reverse proxy layer
- [ ] Expanded false-positive / false-negative scenario documentation
- [ ] Consent/onboarding copy for family-facing flows
- [ ] Incident escalation runbook for maintainers
- [ ] Offline dependency audit automation in CI

## Portfolio polish

- [ ] One-page architecture card PDF for sponsors
- [ ] Partner-facing summary deck
- [ ] Verified edge deployment benchmark notes
- [ ] Demo video script and recorded walkthrough

## Review before real deployment

Before any real-world youth/family deployment, MindMend Guardian needs professional privacy, legal, security, and child-safety review. It should remain a prototype until those reviews and operational controls exist.
