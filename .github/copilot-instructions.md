# MindMend Guardian - Developer Guidelines

## Project Overview

**MindMend Guardian** — 100% offline, privacy-first guardian AI for kids/teens/families. Local wake-word listener, Luna threat/grooming detection, geofencing, parent alerts, low-power on Raspberry Pi. No cloud, no telemetry. Provides compassionate resources (e.g., [988 Suicide & Crisis Lifeline](https://988lifeline.org/)) for distress cases.

## Tech Stack

- **Python 3** — Core language
- **whisper.cpp** — Voice processing (requires submodule setup)
- **llama.cpp** (optional) — Local LLMs
- **Threading + VAD** — Low-power always-listening
- **pytest** — Testing framework
- **GitHub Actions** — CI/CD

Note: whisper.cpp is included as a submodule and requires initialization during setup.

## Preferred Coding Style and Conventions

- Use **idiomatic Python** with type hints
- Follow **PEP 8** style guide
- Prefer **modular components** for maintainability
- **Unit-test-first** for safety-sensitive logic
- Avoid heavyweight libraries that increase power/CPU usage
- Document all public module APIs with clear docstrings
- Use meaningful variable names that convey intent
- Keep functions focused and single-purpose

## Important Rules

**Never introduce:**
- Cloud dependencies
- Telemetry or analytics
- External API calls
- Model weights in commits
- `.env` files or personal data
- Secrets or credentials

**Always:**
- Add tests for voice activation, threat logic, and geofencing
- Follow linting and formatting rules
- Obtain approval before changing legacy/safety-critical modules
- Run tests before submitting PRs
- Document breaking changes

## Setup & Developer Workflow

```bash
# Clone repository
git clone https://github.com/MiMindMendinc/mindmend-guardian.git
cd mindmend-guardian

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Initialize whisper.cpp submodule
git submodule update --init --recursive

# Run test script
python tools/test_tts.py

# Or run main listener script
python guardian/mindmend_guardian.py
```

**Optional:** Run GitHub Actions locally using [act](https://github.com/nektos/act) for pre-submission validation.

## How to Run Tests/Build/Dev Server

```bash
# Run all tests quietly
pytest -q

# Run specific test markers (e.g., skip hardware tests)
pytest -m "not hardware"

# Run hardware tests (recommended on Raspberry Pi)
pytest -m hardware
```

**Testing notes:**
- Lightweight unit tests run on CI
- Manual/hardware tests recommended for Raspberry Pi
- Use `@pytest.mark.hardware` for tests requiring physical hardware
- Test voice activation, threat detection, and geofencing thoroughly

## Architecture & Module Notes

### Luna Safety Core
Real-time threat/grooming analysis engine. Modular separation includes:

- **Listener** — Wake-word detection
- **VAD** — Voice activity detection (Silero)
- **Recognition** — Speech-to-text processing
- **Threat Classifier** — Safety analysis
- **Notifier** — Parent alert system
- **Geofence** — Location-based safety

### Low-Power Considerations
- Use blocking I/O where possible
- Efficient VAD to minimize CPU usage
- Reduce model sizes for edge deployment
- Target <1W power consumption in idle mode
- Optimize for Raspberry Pi constraints

### Do-Not-Touch Areas
Request permission before modifying:
- `/models/` — Model storage
- `/weights/` — Model weights
- `/data/` — Training/test data
- `/test-audio/` — Audio samples
- `.env` — Environment configuration
- Files matching `*.ckpt` or `*.bin`

## Ethics & False-Positive Guidance

**Core Principles:**
- Minimize false positives to maintain trust
- Design compassionate, non-alarming responses
- Human-in-the-loop for serious alerts
- Provide helpful resources, not just warnings
- Consider emotional impact on children and families
- Respect privacy — no data leaves the device

**Testing Ethical Scenarios:**
- Test edge cases that might trigger false alarms
- Verify appropriate responses to genuine threats
- Ensure alert messages are age-appropriate
- Document reasoning for threat classification thresholds

## Security & Privacy

- All processing happens locally on-device
- No internet connectivity required for core features
- No telemetry, tracking, or data collection
- Parent alerts are local (no cloud services)
- Report security issues via GitHub Security Advisories

---

**Questions?** Open an issue or see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

<!-- Last updated: 2026-01-27 -->
