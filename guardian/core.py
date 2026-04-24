"""Core safety helpers for MindMend Guardian.

This module is intentionally lightweight and import-safe. It contains pure Python
logic that can be tested without microphones, GPUs, local LLMs, or audio models.

The production voice loop can call these helpers, and recruiters/collaborators can
see a clear testable safety layer instead of only a hardware-dependent demo script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
from typing import Iterable, Literal

RiskLevel = Literal["low", "medium", "high", "crisis"]

CRISIS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(i\s+want\s+to\s+die|kill\s+myself|end\s+my\s+life)\b", re.I),
    re.compile(r"\b(suicidal|self\s*harm|hurt\s+myself)\b", re.I),
)

HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(abuse|abused|grooming|predator|threatened|blackmail)\b", re.I),
    re.compile(r"\b(run\s+away|unsafe\s+at\s+home|scared\s+to\s+go\s+home)\b", re.I),
)

MEDIUM_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(anxious|panic|depressed|overwhelmed|lonely|hopeless)\b", re.I),
    re.compile(r"\b(bullied|bullying|harassed|excluded)\b", re.I),
)

GROUNDING_STEPS: tuple[str, ...] = (
    "Take one slow breath in and one slow breath out.",
    "Name five things you can see around you.",
    "Put both feet on the floor and notice that you are here right now.",
    "Reach out to a trusted adult if this feels too heavy to hold alone.",
)

CRISIS_FOOTER = (
    "If you or someone else may be in immediate danger, contact emergency services. "
    "In the United States, call or text 988 for the Suicide & Crisis Lifeline."
)


@dataclass(frozen=True)
class GuardianAssessment:
    """Structured result from a lightweight safety scan."""

    risk_level: RiskLevel
    should_escalate: bool
    matched_categories: tuple[str, ...] = field(default_factory=tuple)
    input_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _matches(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def stable_hash(text: str) -> str:
    """Return a stable SHA-256 hash for audit logs without storing raw text."""

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def assess_text(text: str) -> GuardianAssessment:
    """Classify a user message into a simple safety tier.

    This is not a medical or diagnostic classifier. It is a transparent,
    conservative rules-based layer for prototypes and demos.
    """

    normalized = text or ""
    categories: list[str] = []

    if _matches(normalized, CRISIS_PATTERNS):
        categories.append("crisis_language")
        return GuardianAssessment(
            risk_level="crisis",
            should_escalate=True,
            matched_categories=tuple(categories),
            input_hash=stable_hash(normalized),
        )

    if _matches(normalized, HIGH_RISK_PATTERNS):
        categories.append("high_risk_safety_language")
        return GuardianAssessment(
            risk_level="high",
            should_escalate=True,
            matched_categories=tuple(categories),
            input_hash=stable_hash(normalized),
        )

    if _matches(normalized, MEDIUM_RISK_PATTERNS):
        categories.append("emotional_distress")
        return GuardianAssessment(
            risk_level="medium",
            should_escalate=False,
            matched_categories=tuple(categories),
            input_hash=stable_hash(normalized),
        )

    return GuardianAssessment(
        risk_level="low",
        should_escalate=False,
        matched_categories=tuple(categories),
        input_hash=stable_hash(normalized),
    )


def supportive_reply(user_text: str) -> str:
    """Return a calm, bounded support response for demos and fallback mode."""

    assessment = assess_text(user_text)

    if assessment.risk_level == "crisis":
        return (
            "I'm really glad you said something. You deserve immediate support from a real person. "
            "Please move closer to a trusted adult or call/text 988 now. "
            f"{CRISIS_FOOTER}"
        )

    if assessment.risk_level == "high":
        return (
            "That sounds serious, and you should not have to handle it alone. "
            "Please tell a trusted adult, caregiver, counselor, or local emergency contact as soon as you can. "
            "I can stay with you for a grounding step, but a real person needs to know."
        )

    if assessment.risk_level == "medium":
        return (
            "I'm sorry this feels heavy right now. Your feelings are real, and you are not alone. "
            f"{GROUNDING_STEPS[0]} {GROUNDING_STEPS[-1]}"
        )

    return "I'm here with you. Take it one step at a time, and tell me what's on your mind."


def audit_event(event_type: str, assessment: GuardianAssessment) -> dict[str, object]:
    """Create a privacy-preserving audit event."""

    return {
        "event_type": event_type,
        "risk_level": assessment.risk_level,
        "should_escalate": assessment.should_escalate,
        "matched_categories": list(assessment.matched_categories),
        "input_hash": assessment.input_hash,
        "timestamp": assessment.timestamp,
    }
