# Copilot & Contributor Instructions — MindMend Guardian

## Project overview
100% offline guardian AI for kids/teens: local wake-word listener ("hey mindmend"), gentle TTS conversation, Luna Safety Core for threat/grooming detection, geofencing, parent alerts. No cloud, no telemetry, compassionate responses (e.g., 988 resources).

## Tech stack
Python 3, whisper.cpp (submodule), optional llama.cpp, Silero VAD/threading for low-power listening, pytest, GitHub Actions CI.

## Rules
No cloud/telemetry/external APIs. Never commit .env, model weights (*.ckpt/*.bin/*.pth), personal audio. Add tests for voice/threat/geofence. Prioritize low-power (<1W idle on RPi Zero). Ethical: minimize false positives, gentle/non-alarmist alerts.

## Setup
Clone → venv → install dependencies (if available) → git submodule update --init --recursive → run tts_test.py or main listener.

## Do-not-touch
/models/* /weights/* /data/* /test-audio/* .env large binaries.

## Ethics
Human-in-loop for high alerts; supportive flows.
