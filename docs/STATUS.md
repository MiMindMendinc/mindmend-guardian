# MindMend Guardian Status

## Current Status

**Active prototype / portfolio-grade demo (v0.3.0).**

MindMend Guardian demonstrates privacy-first youth safety and family wellness product thinking with runnable local demos, import-safe core logic, modular API code, and CI quality gates. It is not a finished product, clinical tool, crisis service, or compliance-certified system.

## What Works Today

- import-safe rules engine in `guardian/core.py`
- local CLI demo: `python -m guardian.demo` (synthetic samples only)
- optional Streamlit dashboard demo (synthetic samples only)
- Luna API backend with env-based configuration and request validation
- privacy-preserving audit hashes (raw text not stored by default)
- pytest suite covering core, Luna, demo, and scenario cases
- GitHub Actions CI: pytest, ruff, pip-audit, secret-pattern scan
- architecture, API, security, demo, privacy, and threat-model docs

## What Must Be Verified Before Real Deployment

- risk-detection behavior under realistic but synthetic test cases
- false positive and false negative handling in real family contexts
- local logging behavior, retention, and encryption
- parent/guardian review workflow
- emergency and crisis routing language in product UX
- deployment environment and network behavior
- legal/privacy/compliance review for youth data

## Not Claimed

MindMend Guardian does not currently claim:

- clinical validation
- HIPAA compliance
- COPPA compliance
- medical-device status
- crisis-service status
- guaranteed detection of self-harm, grooming, bullying, abuse, or distress
- replacement for parents, guardians, schools, clinicians, or emergency services

## Release Readiness Checklist

- [x] all tests pass with `pytest`
- [x] safety workflow / scenario tests added
- [x] synthetic examples only in demos and tests
- [x] docs/THREAT_MODEL.md present and aligned
- [x] docs/PRIVACY_AND_SAFETY.md present and aligned
- [ ] local encrypted logging example documented and implemented
- [ ] screenshots or demo GIF added
- [x] architecture diagram added
- [ ] Raspberry Pi / edge deployment notes expanded
- [x] README claims checked against code
- [ ] professional review completed before sensitive deployment
