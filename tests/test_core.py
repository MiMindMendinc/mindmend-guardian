"""Tests for the import-safe MindMend Guardian core helpers."""

from guardian import assess_text, audit_event, stable_hash, supportive_reply


def test_stable_hash_is_deterministic():
    assert stable_hash("hello") == stable_hash("hello")
    assert stable_hash("hello") != stable_hash("goodbye")


def test_low_risk_assessment():
    result = assess_text("I had a normal day and want to journal.")

    assert result.risk_level == "low"
    assert result.should_escalate is False
    assert result.matched_categories == ()


def test_medium_risk_assessment():
    result = assess_text("I feel anxious and overwhelmed today.")

    assert result.risk_level == "medium"
    assert result.should_escalate is False
    assert "emotional_distress" in result.matched_categories


def test_high_risk_assessment():
    result = assess_text("I feel unsafe at home and I am scared to go home.")

    assert result.risk_level == "high"
    assert result.should_escalate is True
    assert "high_risk_safety_language" in result.matched_categories


def test_crisis_assessment():
    result = assess_text("I want to die and I need help.")

    assert result.risk_level == "crisis"
    assert result.should_escalate is True
    assert "crisis_language" in result.matched_categories


def test_supportive_reply_has_crisis_boundary():
    reply = supportive_reply("I want to die")

    assert "988" in reply
    assert "real person" in reply


def test_audit_event_does_not_include_raw_text():
    assessment = assess_text("I feel anxious")
    event = audit_event("demo", assessment)

    assert event["event_type"] == "demo"
    assert event["risk_level"] == "medium"
    assert "input_hash" in event
    assert "I feel anxious" not in str(event)
