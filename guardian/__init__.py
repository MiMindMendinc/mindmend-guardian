"""MindMend Guardian package.

Privacy-first youth-support AI prototype from Michigan MindMend Inc.

The package exposes lightweight, testable safety helpers separately from the
hardware-dependent voice demo so the core behavior can be imported in CI,
examples, and downstream apps.
"""

from .core import (
    CRISIS_FOOTER,
    GROUNDING_STEPS,
    GuardianAssessment,
    assess_text,
    audit_event,
    stable_hash,
    supportive_reply,
)

__version__ = "0.3.0"
__author__ = "Michigan MindMend Inc."

__all__ = [
    "CRISIS_FOOTER",
    "GROUNDING_STEPS",
    "GuardianAssessment",
    "assess_text",
    "audit_event",
    "stable_hash",
    "supportive_reply",
]
