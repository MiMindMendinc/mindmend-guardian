# Privacy and Safety Notes

MindMend Guardian is designed around a simple rule: sensitive family and youth data should be protected first, and AI should support humans rather than replace them.

## Privacy Principles

- Collect as little data as possible.
- Prefer local processing where practical.
- Treat logs as sensitive.
- Use synthetic examples in tests and documentation.
- Do not commit real child, family, school, medical, therapy, or crisis records.
- Clearly disclose when any remote service, hosted asset, analytics tool, or external model is used.

## Safety Principles

- AI assists; humans decide.
- Supportive language is preferred over fear-based alerts.
- Parent, guardian, clinician, school, or emergency escalation should remain human-controlled.
- The system should avoid punishment-first framing.
- Safety claims must be tied to visible tests, code, or documentation.

## Crisis Boundary

MindMend Guardian is not a crisis service. It cannot guarantee detection or response.

If someone may be in immediate danger, contact local emergency services. In the United States, call or text **988** for the Suicide & Crisis Lifeline.

## Deployment Verification Checklist

Before using any deployment with sensitive data, verify:

- [ ] model routing is local or clearly disclosed
- [ ] no unexpected third-party analytics are present
- [ ] no remote CDNs are required for sensitive workflows
- [ ] local logs are documented
- [ ] retention behavior is documented
- [ ] stored data is encrypted or intentionally disabled
- [ ] users understand prototype status
- [ ] crisis and emergency boundaries are visible
- [ ] legal/privacy/safety review has been completed for real-world use

## What Not To Do

Do not use this prototype to make final decisions about a child’s safety, discipline, diagnosis, treatment, school action, legal reporting, or emergency response without qualified human review.
