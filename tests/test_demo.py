"""Tests for the local MindMend Guardian demo."""

import json
import subprocess
import sys

from guardian.demo.cli import PRIVACY_NOTE, evaluate_sample, run_demo


def test_evaluate_sample_uses_synthetic_text_only():
    result = evaluate_sample("I feel anxious and overwhelmed today.")

    assert result.risk_level == "medium"
    assert result.human_review_recommended is False
    assert PRIVACY_NOTE in result.privacy_note
    assert len(result.audit_hash) == 64


def test_audit_output_excludes_raw_text():
    sensitive = "I want to die and I need help."
    result = evaluate_sample(sensitive)

    assert sensitive not in result.audit_hash
    assert result.risk_level == "crisis"


def test_run_demo_json_output(capsys):
    run_demo(json_output=True)
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert isinstance(payload, list)
    assert len(payload) >= 4
    assert all("audit_hash" in item for item in payload)
    assert all("sample_text" not in str(item.get("audit_hash", "")) for item in payload)


def test_python_m_guardian_demo_module():
    completed = subprocess.run(
        [sys.executable, "-m", "guardian.demo", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert len(payload) >= 1
