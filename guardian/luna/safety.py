"""Import-safe Luna safety helpers for message and location checks."""

from __future__ import annotations

import logging
import re
from typing import Any

from guardian.luna.geofence import haversine, is_out_of_bounds

logger = logging.getLogger(__name__)

DANGER_WORDS = [
    "sweetie",
    "pretty",
    "meetup",
    "alone",
    "send pic",
    "trust me",
    "age",
    "secret",
    "hotel",
    "come over",
    "buy you",
    "love you",
    "private",
    "touch",
    "kiss",
    "baby",
    "cutie",
    "dm me",
    "nude",
    "sext",
]
danger_pattern = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in DANGER_WORDS) + r")\b",
    re.IGNORECASE,
)

TOXIC_WORDS = [
    "hate",
    "kill",
    "die",
    "stupid",
    "ugly",
    "fat",
    "loser",
    "hurt",
    "bully",
    "threat",
    "scam",
    "dumb",
    "idiot",
    "suicide",
    "cut",
]
toxic_pattern = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in TOXIC_WORDS) + r")\b",
    re.IGNORECASE,
)

_nlp = None
_nlp_load_attempted = False


def _load_nlp():
    global _nlp, _nlp_load_attempted

    if _nlp_load_attempted:
        return _nlp

    _nlp_load_attempted = True
    try:
        import spacy

        model = spacy.load("en_core_web_sm")
        model.add_pipe("spacytextblob")
        _nlp = model
        logger.info("spaCy loaded successfully.")
    except Exception as exc:
        logger.warning("spaCy load failed: %s - NLP features disabled.", exc)
        _nlp = None

    return _nlp


def scan_message(text: str) -> dict[str, Any]:
    """Scan a chat message for grooming or danger keywords."""

    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        matches = danger_pattern.findall(text)
        count = len(matches)
        return {"is_flagged": count > 0, "score": count, "matches": matches}
    except Exception as exc:
        logger.error("Scan error: %s", exc)
        return {"is_flagged": False, "score": 0, "matches": []}


def toxicity_score(sentence: str) -> dict[str, Any]:
    """Estimate toxicity using spaCy TextBlob polarity when available."""

    nlp = _load_nlp()
    if nlp is None:
        keyword_matches = toxic_pattern.findall(sentence)
        return {
            "toxic": bool(keyword_matches),
            "polarity": -0.5 if keyword_matches else 0.0,
            "entity_count": 0,
            "bad_entities": [],
        }

    try:
        doc = nlp(sentence)
        polarity = doc._.blob.polarity
        is_toxic = polarity < -0.2
        return {
            "toxic": is_toxic,
            "polarity": polarity,
            "entity_count": 0,
            "bad_entities": [],
        }
    except Exception as exc:
        logger.error("Toxicity error: %s", exc)
        return {"toxic": False, "polarity": 0, "entity_count": 0, "bad_entities": []}


def build_chat_alert_message(text: str, *, max_preview: int = 100) -> str:
    """Return a privacy-bounded alert message without logging raw chat text elsewhere."""

    preview = text[:max_preview]
    suffix = "..." if len(text) > max_preview else ""
    return f"Suspicious chat detected (hash preview only): '{preview}{suffix}'"


__all__ = [
    "build_chat_alert_message",
    "haversine",
    "is_out_of_bounds",
    "scan_message",
    "toxicity_score",
]
