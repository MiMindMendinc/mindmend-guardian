"""Synthetic safety scenario tests for guardian.core."""

import pytest
from guardian import assess_text, audit_event, supportive_reply


@pytest.mark.parametrize(
    ("message", "expected_level"),
    [
        ("Someone at school keeps bullying and harassing me.", "medium"),
        ("A stranger asked me to send a pic and meet alone.", "low"),
        ("I feel unsafe at home because of abuse.", "high"),
        ("I want to kill myself tonight.", "crisis"),
        ("I finished my homework and played basketball.", "low"),
    ],
)
def test_synthetic_scenarios(message, expected_level):
    result = assess_text(message)
    assert result.risk_level == expected_level


def test_bullying_language_is_medium_not_crisis():
    result = assess_text("I am being bullied and harassed at lunch.")
    assert result.risk_level == "medium"
    assert result.should_escalate is False


def test_grooming_phrase_without_core_pattern_stays_low():
    # Core rules focus on distress/safety language; Luna handles grooming keywords separately.
    result = assess_text("Can you send pic and meet me alone later?")
    assert result.risk_level == "low"


def test_supportive_reply_never_empty():
    reply = supportive_reply("Everything feels hopeless and lonely.")
    assert len(reply.strip()) > 20


def test_audit_events_never_store_raw_text():
    sample = "I feel unsafe at home and I am scared to go home."
    event = audit_event("scenario_test", assess_text(sample))
    serialized = str(event)

    assert sample not in serialized
    assert event["input_hash"]
    assert event["should_escalate"] is True
