# MindMend Guardian

**Privacy-first edge guardian prototype for youth safety and family wellness.**

MindMend Guardian is a local-first AI safety prototype designed to help families explore child/family safety workflows without default cloud dependency. It focuses on privacy-first risk detection, calm support language, family-controlled review, and human escalation paths.

Built by **Michigan MindMend Inc.** as a portfolio demonstration of responsible, edge-deployable AI safety technology.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![CI](https://github.com/MiMindMendinc/mindmend-guardian/actions/workflows/ci.yml/badge.svg)](https://github.com/MiMindMendinc/mindmend-guardian/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)](tests/test_basic.py)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Prototype-blue)](#current-status)

---

## Problem

Many youth safety tools create a hard tradeoff:

- send sensitive family data to the cloud, or
- use blunt tools that miss context and feel scary or clinical.

Families need safety support that is private, understandable, and human-guided.

## Solution

MindMend Guardian explores a different pattern:

```text
Local input / family app event
        ↓
Guardian safety layer
        ↓
Risk and wellness check
        ↓
Supportive response or resource suggestion
        ↓
Parent / guardian review or escalation path
```

The goal is not to replace parents, clinicians, or crisis teams. The goal is to build a privacy-first support layer that helps surface risk while keeping humans in control.

---

## Key Features

- **Local-first and offline-capable direction** — designed for laptops, mini-PCs, Raspberry Pi-style devices, or local networks.
- **Youth-appropriate support language** — calm, supportive response framing instead of fear-based alerts.
- **Risk-signal detection direction** — grooming language, self-harm signals, bullying patterns, crisis keywords, and distress indicators.
- **Private family logs** — intended for parent/guardian-controlled review where logging is enabled.
- **Human escalation paths** — clear boundaries and handoff when adult or professional help is needed.
- **Configurable sensitivity direction** — families and deployments should be able to tune what counts as risk.

---

## Tech Stack

- Python 3.10+
- Local-first Python package structure
- Rule-based and prototype safety checks
- Optional local model direction: Ollama / llama.cpp / Transformers
- Local storage direction: SQLite or encrypted local logs
- Test coverage with `pytest`
- GitHub Actions CI

---

## Quick Demo / How To Try It

Clone and run the current test suite:

```bash
git clone https://github.com/MiMindMendinc/mindmend-guardian.git
cd mindmend-guardian
python -m pip install --upgrade pip
pip install pytest
pytest
```

Prototype/demo paths may change as the repo is packaged. The current tests verify package structure, required files, and Python syntax for key modules.

A dedicated runnable demo command remains on the roadmap. The commands above describe the currently verified path.


---

## Privacy & Safety Commitments

MindMend Guardian is designed around these commitments:

- **No default cloud dependency** for core safety concepts.
- **Data minimization**: collect only what the workflow needs.
- **Family-controlled logs** where logging exists.
- **Human-in-the-loop escalation**: AI does not make final decisions about a child’s safety.
- **Clear safety boundaries**: supportive software, not a therapist or crisis service.
- **Truth over hype**: prototype claims stay tied to visible code, docs, and tests.

---

## What MindMend Guardian Does Not Claim

MindMend Guardian does **not** currently claim:

- medical or clinical validation
- 100% detection accuracy
- HIPAA / COPPA compliance out of the box
- replacement for parents, guardians, clinicians, or crisis professionals
- guaranteed abuse, grooming, self-harm, or bullying detection
- production readiness for unsupervised child safety use

If someone may be in immediate danger, contact emergency services. In the United States, call or text **988** for the Suicide & Crisis Lifeline.

---

## Current Status

**Active prototype / portfolio project.**

This repo demonstrates privacy-first child/family safety product thinking, local-first architecture, responsible AI documentation, and early engineering structure. It is not a finished product.

---

## Roadmap

- [ ] Full safety workflow tests
- [ ] GitHub Actions CI badge verified green
- [ ] Docker and Raspberry Pi deployment notes
- [ ] Parent dashboard v1 mockup
- [ ] Config file for sensitivity thresholds
- [ ] Demo video and screenshots
- [ ] Clear installable package entrypoint
- [ ] Local logging / storage example

---

## Recruiter / Sponsor Notes

This project demonstrates:

- responsible AI safety engineering
- privacy-first / offline-capable product design
- real-world youth and family safety use-case thinking
- local-first architecture for sensitive environments
- end-to-end ownership: detection, escalation, logging, and documentation

It is a strong signal for roles or partnerships involving AI safety, trust and safety, child protection technology, privacy engineering, and community-focused AI.

---

## Built By

**Lyle Perrien II**  
Founder, **Michigan MindMend Inc.**  
Owosso, Michigan  
X: [@p_perrien](https://x.com/p_perrien)

Building privacy-first, offline-capable AI safety tools for kids, families, and communities.

## License

MIT
