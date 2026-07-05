# Security Policy

Michigan MindMend Inc. takes privacy, youth safety, and responsible disclosure
seriously.

## Supported Versions

Security fixes are prioritized for the active development line on `main`.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

Please do **not** open a public issue for security problems, secrets, private
data exposure, child-safety concerns, or exploit details.

Report concerns privately using one of these channels:

1. **Email (preferred):** [michiganmindmendinc@proton.me](mailto:michiganmindmendinc@proton.me)
2. **GitHub Private Security Advisories:** use the
   [Security tab](https://github.com/MiMindMendinc/mindmend-guardian/security/advisories/new)
   on this repository when available

Please include:

- affected repository and file/path
- clear reproduction steps
- expected impact
- screenshots or logs if safe to share
- whether any private data, keys, or sensitive records were exposed

## Response Timeline

We aim to:

- acknowledge receipt within **72 hours**
- provide an initial assessment within **7 days**
- coordinate a fix and disclosure timeline based on severity

We may request additional information to reproduce or validate the report.

## Disclosure Policy

- We prefer coordinated disclosure and will work with reporters on timing.
- Credit will be given in release notes when reporters agree.
- Do not publicly disclose details until a fix is available or we agree on a
  disclosure date.

## Scope

This policy applies to MindMend Guardian and related Michigan MindMend Inc.
safety/prototype repositories, including:

- Python packages under `guardian/`
- CI workflows under `.github/workflows/`
- documentation and configuration that affect deployment safety

Out of scope unless they introduce a security regression in this repository:

- third-party model weights or external services you configure separately
- experimental subprojects documented as non-production (for example `ani-2027/`)

## Important Boundaries

MindMend Guardian is a prototype. It is not a crisis service, therapist, medical
device, law-enforcement tool, or replacement for parents, guardians, clinicians,
or emergency services.

If someone may be in immediate danger, contact local emergency services. In the
United States, call or text **988** for the Suicide & Crisis Lifeline.
