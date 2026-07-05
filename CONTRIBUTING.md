# Contributing to MindMend Guardian

Thank you for your interest in MindMend Guardian.

MindMend Guardian is a Michigan MindMend Inc. prototype for privacy-first youth
safety and family wellness workflows. Contributions should protect families, avoid
hype, and keep humans in control.

By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Project Goals

MindMend Guardian should remain:

- privacy-first
- local-first where possible
- clear about prototype status
- careful with youth and family safety language
- supportive, not clinical or emergency-service replacement
- human-in-the-loop by design

## Good Contributions

Helpful contributions include:

- safety workflow tests
- clearer risk-category documentation
- local-only demo paths
- parent/guardian review workflow examples
- encrypted local logging examples
- Raspberry Pi / edge deployment notes
- screenshots, diagrams, and demo assets
- documentation that clarifies limits instead of exaggerating claims

## Development Setup

Requires **Python 3.11+**.

```bash
git clone https://github.com/MiMindMendinc/mindmend-guardian.git
cd mindmend-guardian
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

Optional extras:

```bash
pip install -e ".[simulator]"   # Streamlit dashboard demo
pip install -e ".[voice]"       # voice runtime (heavy deps)
```

Run quality checks locally:

```bash
ruff check guardian tests
ruff format --check guardian tests
pytest --cov=guardian --cov-report=term-missing
python -m guardian.demo --json
```

Optional pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Pull Request Checklist

A good PR should include:

- [ ] clear problem statement
- [ ] small focused change
- [ ] tests for new behavior
- [ ] documentation updates when behavior changes
- [ ] [`CHANGELOG.md`](CHANGELOG.md) entry for user-visible changes
- [ ] no real youth, family, medical, school, or private records
- [ ] no secrets, API keys, `.env` files, or private logs
- [ ] honest claims tied to visible code and tests

Use the PR template checklist when opening a pull request.

## Safety Rules

Do not add code, prompts, datasets, screenshots, or logs that include real private
child/family data.

Do not claim that this project is:

- clinically validated
- a therapist
- a crisis service
- HIPAA / COPPA compliant out of the box
- a replacement for parents, guardians, clinicians, schools, or emergency services

## Crisis Boundary

If someone may be in immediate danger, contact local emergency services. In the
United States, call or text **988** for the Suicide & Crisis Lifeline.

## Reporting Security Issues

Do not open public issues for vulnerabilities or private data exposure. Follow
[`SECURITY.md`](SECURITY.md).

## Communication

- Bug reports and feature requests: [GitHub Issues](https://github.com/MiMindMendinc/mindmend-guardian/issues/new/choose)
- General support routing: [`SUPPORT.md`](SUPPORT.md)
- Public updates: Lyle Perrien II on X at [@p_perrien](https://x.com/p_perrien)

## Releases

Maintainers follow [`docs/RELEASE.md`](docs/RELEASE.md) for version bumps, tags,
and GitHub Releases.

## License

By contributing, you agree that your contribution will be licensed under the MIT
License in [`LICENSE`](LICENSE).
