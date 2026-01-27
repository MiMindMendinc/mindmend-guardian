# Contributing to MindMend Guardian

Thanks for looking — I'm a solo Michigan dad building MindMend Guardian, a privacy-first, offline child-safety AI. Family, farm, and nonprofit life mean I'm short on time, so any help is deeply appreciated.

## How You Can Help

- Small docs fixes (README, install steps)
- Tests or CI improvements
- Raspberry Pi / low-power install recipes
- Voice phrases, sample alerts, and UX wording
- Code review, debugging, or feature PRs
- **RPi optimizations** — Battery life and performance improvements
- **Wake-word improvements** — Better detection accuracy
- **Dataset-free testing harnesses** — Privacy-preserving test frameworks
- **Threat-model improvements** — Enhanced safety detection
- **Ethical messaging** — Compassionate user communications

## Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/mindmend-guardian.git
   cd mindmend-guardian
   ```
3. **Create a feature branch** following our naming convention:
   - `feature/short-description` — for new features
   - `fix/short-description` — for bug fixes
   ```bash
   git checkout -b feature/your-improvement
   ```
4. **Make changes** and add tests where applicable
5. **Commit** with clear messages:
   ```bash
   git commit -m "feat: add voice activation threshold tuning"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/your-improvement
   ```
7. **Open a Pull Request** against `main`

## Setup & Development Environment

### Initial Setup

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when requirements.txt is available)
pip install -r requirements.txt

# Initialize whisper.cpp submodule
git submodule update --init --recursive

# Run test script to verify setup
python tools/test_tts.py

# Or run main listener script
python guardian/mindmend_guardian.py
```

### Development Workflow

- Keep your fork updated with upstream changes
- Work in feature branches, not directly on `main`
- Test locally before pushing
- Request reviewers when opening PRs

## Pull Request Checklist

Before submitting your PR, please ensure:

- [ ] **Reference relevant issue** — Link to the issue your PR addresses
- [ ] **Include tests** — Add or update tests for your changes
- [ ] **Document changes** — Update relevant documentation
- [ ] **Screenshots/samples** — For voice/workflow changes, include audio samples or screenshots
- [ ] **Request reviewers** — Tag appropriate maintainers
- [ ] **Link verification steps** — Describe how you tested (e.g., "Tested on Raspberry Pi 4")
- [ ] **Run linters** — Ensure code passes style checks
- [ ] **Run tests** — All tests pass locally

## Testing Guidance

### Running Tests

```bash
# Run all tests quietly
pytest -q

# Run tests excluding hardware tests (for CI/development)
pytest -m "not hardware"

# Run hardware tests (requires Raspberry Pi or similar)
pytest -m hardware

# Run specific test file
pytest tests/test_specific.py -v
```

### Hardware Testing

For performance-sensitive changes:
- Test on Raspberry Pi 4 or similar edge device
- Measure power consumption in idle mode (target: <1W)
- Verify wake-word detection accuracy
- Check CPU/memory usage under load

### Test Markers

Use pytest markers to categorize tests:
- `@pytest.mark.hardware` — Requires physical hardware
- `@pytest.mark.slow` — Long-running tests
- `@pytest.mark.integration` — Integration tests

## Code Style & Linters

We follow Python best practices:

- **Style Guide:** PEP 8
- **Type Hints:** Use type annotations
- **Formatters:** Run `black` and `ruff` (or `flake8`)
- **Import Sorting:** Keep imports organized

### Running Linters

```bash
# Format code with black (if installed)
black .

# Lint with ruff (if installed)
ruff check .

# Alternative: flake8
flake8 .
```

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

This project is about protecting kids — please be respectful, collaborative, and maintain a welcoming environment for all contributors.

## Security & Secrets

**Never commit:**
- Credentials or API keys
- Model files (`.pth`, `.bin`, `.ckpt`)
- Personal data or test data containing PII
- `.env` files or environment secrets

**To report security issues:**
- Use GitHub Security Advisories (preferred)
- Or email the maintainer directly (see README)

## Communication

- Open issues for questions or small tasks
- Tag me on X: @p_perrien if you want a quick ping
- Be patient — this is a solo-maintained project with limited availability

## Perks & Thanks

- First contributors (meaningful PRs merged) get a named farm animal (cow, goat, or chicken) and a shoutout in the README. 🐄🐐🐔
- Contributors who help significantly will be added to the project acknowledgements.

Thank you — your time matters here.

---

**See also:** [.github/copilot-instructions.md](.github/copilot-instructions.md) for detailed developer guidelines.