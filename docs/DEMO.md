# Demo Guide

MindMend Guardian includes two local demos that require **no cloud services** and use **synthetic sample text only**.

## CLI demo (recommended first step)

```bash
pip install -e ".[dev]"
python -m guardian.demo
```

### What it shows

For each synthetic sample message, the demo prints:

- risk level
- matched signal categories
- supportive response
- human review recommendation
- privacy-preserving audit hash
- privacy note (raw text not stored by default)

### JSON output

```bash
python -m guardian.demo --json
```

Useful for scripts, screenshots, and CI verification.

### Sample categories covered

| Sample theme | Expected risk |
|--------------|---------------|
| Normal day / journaling | `low` |
| Anxiety / overwhelm | `medium` |
| Unsafe at home language | `high` |
| Crisis/self-harm language | `crisis` |

## Dashboard demo (optional)

```bash
pip install -e ".[simulator]"
python -m guardian.dashboard
```

Alternative launch:

```bash
streamlit run guardian/dashboard/app.py
```

### Dashboard panels

- Risk level metric
- Human review needed metric
- Matched risk signals
- Supportive response card
- Privacy notes
- Audit hash display
- Expandable list of all synthetic samples

## Luna API demo (optional, requires config)

```bash
pip install -e ".[api,dev]"
export LUNA_SECRET_KEY="your-long-random-secret-at-least-32-characters"
python guardian/luna/luna_safety_core.py
```

Example request:

```bash
curl -s -X POST http://127.0.0.1:5000/check_chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"I feel anxious and overwhelmed today."}'
```

See [`API.md`](API.md) and [`SECURITY_CONFIG.md`](SECURITY_CONFIG.md).

## Capture assets for README / sponsors

Suggested local capture workflow:

```bash
python -m guardian.demo > docs/assets/cli-demo-sample.txt
python -m guardian.dashboard
# Take screenshots and save to docs/assets/
```

Assets referenced by the README (captured locally with synthetic samples only):

- `docs/assets/dashboard-medium.png`: "anxious and overwhelmed" sample, MEDIUM risk
- `docs/assets/dashboard-high.png`: "unsafe at home" sample, HIGH risk, human review suggested
- `docs/assets/cli-demo-sample.txt`: full `python -m guardian.demo` output

## Demo boundaries

These demos:

- do not store raw sample text in audit output
- do not require Firebase, OpenAI, or other third-party APIs
- do not claim clinical validation or guaranteed detection

They are portfolio and contributor onboarding tools, not unsupervised child-safety products.
