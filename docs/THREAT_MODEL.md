# MindMend Guardian Threat Model

## Purpose

MindMend Guardian is a privacy-first youth safety and family wellness prototype. This document identifies the main risks the project is designed to reduce and the risks it does not solve by itself.

## Assets to Protect

- child and family text inputs
- wellness notes or safety signals
- parent / guardian review logs
- local configuration
- device identity and local storage
- API keys or model configuration, if any
- user trust and safety boundaries

## Trust Boundaries

```text
Family / App Event
  -> Guardian safety workflow
  -> local policy / model path
  -> supportive response or escalation suggestion
  -> parent / guardian / trusted adult review
```

Important boundaries:

- child/user input is untrusted and sensitive
- AI output is untrusted and must be bounded
- logs are sensitive even when stored locally
- local model routing must be verified before claiming offline operation
- humans remain responsible for decisions and escalation

## Primary Threats

### Private Youth or Family Data Exposure

Risk: sensitive child/family content is stored, logged, shared, or sent to a remote service without clear consent.

Mitigations:

- local-first architecture direction
- data minimization
- synthetic test data only
- future encrypted local logs
- clear documentation of model routing and log storage

### Unsafe Reliance on AI

Risk: users rely on the system as a therapist, crisis service, or guaranteed detector.

Mitigations:

- clear non-claims
- human-in-the-loop language
- crisis boundary language
- escalation suggestions instead of final decisions

### False Negatives

Risk: the system misses self-harm, grooming, bullying, abuse, or distress signals.

Mitigations:

- conservative safety wording
- testing across synthetic scenarios
- adult review workflows
- no claims of complete detection

### False Positives

Risk: normal or creative speech is flagged as risk, causing unnecessary fear or punishment.

Mitigations:

- supportive language
- review-before-action design
- configurable sensitivity direction
- focus on care and context, not punishment

### Misconfigured Offline Claims

Risk: a deployment is described as offline while using remote models, CDNs, analytics, or hosted assets.

Mitigations:

- deployment verification checklist
- explicit offline dependency review
- README language tied to evidence

### Unsafe Logs

Risk: family logs preserve sensitive information without encryption, retention limits, or access control.

Mitigations:

- local storage documentation
- future encrypted logs
- family-controlled review direction
- retention policy before real deployment

## Out of Scope

MindMend Guardian alone does not solve:

- emergency response
- clinical diagnosis
- therapy
- legal reporting duties
- complete abuse detection
- complete self-harm detection
- malicious device administrators
- physical device security
- certified compliance requirements

## Production Hardening Backlog

- [ ] synthetic safety scenario test suite
- [ ] encrypted local logging
- [ ] retention controls
- [ ] parent / guardian review UI
- [ ] offline dependency audit
- [ ] Raspberry Pi deployment notes
- [ ] professional privacy/legal/safety review
- [ ] clear consent and onboarding copy
- [ ] incident escalation documentation

## Crisis Boundary

If someone may be in immediate danger, contact local emergency services. In the United States, call or text **988** for the Suicide & Crisis Lifeline.
