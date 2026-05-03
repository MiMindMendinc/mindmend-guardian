# Contributing to MindMend Guardian

Thank you for your interest in MindMend Guardian.

MindMend Guardian is a Michigan MindMend Inc. prototype for privacy-first youth safety and family wellness workflows. Contributions should protect families, avoid hype, and keep humans in control.

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

```bash
git clone https://github.com/YOUR-USERNAME/mindmend-guardian.git
cd mindmend-guardian
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install pytest
pytest
```

## Pull Request Checklist

A good PR should include:

- [ ] clear problem statement
- [ ] small focused change
- [ ] tests for new behavior
- [ ] documentation updates when behavior changes
- [ ] no real youth, family, medical, school, or private records
- [ ] no secrets, API keys, `.env` files, or private logs
- [ ] honest claims tied to visible code and tests

## Safety Rules

Do not add code, prompts, datasets, screenshots, or logs that include real private child/family data.

Do not claim that this project is:

- clinically validated
- a therapist
- a crisis service
- HIPAA / COPPA compliant out of the box
- a replacement for parents, guardians, clinicians, schools, or emergency services

## Crisis Boundary

If someone may be in immediate danger, contact local emergency services. In the United States, call or text **988** for the Suicide & Crisis Lifeline.

## Reporting Security Issues

Do not open public issues for vulnerabilities or private data exposure. Use `SECURITY.md`.

## Communication

Open GitHub issues for bugs, documentation fixes, and focused feature requests. For quick public updates, Lyle Perrien II can also be reached on X at [@p_perrien](https://x.com/p_perrien).

## License

By contributing, you agree that your contribution will be licensed under the MIT License used by this repository.
