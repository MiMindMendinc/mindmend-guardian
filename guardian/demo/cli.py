"""Import-safe local demo using synthetic sample text only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from guardian.core import assess_text, audit_event, supportive_reply

# Synthetic demo inputs only — no real child or family data.
SAMPLE_MESSAGES: tuple[str, ...] = (
    "I had a normal day and want to journal about soccer practice.",
    "I feel anxious and overwhelmed about school today.",
    "I feel unsafe at home and I am scared to go home.",
    "I want to die and I need help right now.",
)


@dataclass(frozen=True)
class DemoResult:
    sample_text: str
    risk_level: str
    matched_categories: tuple[str, ...]
    should_escalate: bool
    supportive_response: str
    human_review_recommended: bool
    audit_hash: str
    privacy_note: str


PRIVACY_NOTE = (
    "Raw message text is not stored in audit output by default. "
    "Only a SHA-256 hash and risk metadata are recorded for review workflows."
)


def evaluate_sample(text: str) -> DemoResult:
    assessment = assess_text(text)
    event = audit_event("local_demo", assessment)

    return DemoResult(
        sample_text=text,
        risk_level=assessment.risk_level,
        matched_categories=assessment.matched_categories,
        should_escalate=assessment.should_escalate,
        supportive_response=supportive_reply(text),
        human_review_recommended=assessment.should_escalate,
        audit_hash=str(event["input_hash"]),
        privacy_note=PRIVACY_NOTE,
    )


def format_result(result: DemoResult) -> str:
    lines = [
        "MindMend Guardian — Local Demo (synthetic sample only)",
        "=" * 58,
        f"Sample text: {result.sample_text}",
        f"Risk level: {result.risk_level}",
        f"Matched signals: {', '.join(result.matched_categories) or 'none'}",
        f"Human review recommended: {'yes' if result.human_review_recommended else 'no'}",
        f"Audit hash: {result.audit_hash[:16]}…",
        "",
        "Supportive response:",
        result.supportive_response,
        "",
        "Privacy note:",
        result.privacy_note,
    ]
    return "\n".join(lines)


def run_demo(*, json_output: bool = False) -> list[DemoResult]:
    results = [evaluate_sample(text) for text in SAMPLE_MESSAGES]

    if json_output:
        payload = [asdict(item) for item in results]
        print(json.dumps(payload, indent=2))
        return results

    for index, result in enumerate(results, start=1):
        if index > 1:
            print("\n")
        print(format_result(result))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local MindMend Guardian safety demo using synthetic samples."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of human-readable output.",
    )
    args = parser.parse_args(argv)
    run_demo(json_output=args.json)
    return 0
