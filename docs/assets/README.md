# Demo assets

Captured locally from the Streamlit dashboard and CLI demo using the built-in
synthetic sample messages only (no real user, child, or family data):

- `dashboard-medium.png`: "anxious and overwhelmed" sample, MEDIUM risk
- `dashboard-high.png`: "unsafe at home" sample, HIGH risk, human review suggested
- `cli-demo-sample.txt`: full output of `python -m guardian.demo`

To regenerate:

    pip install -e ".[simulator]"
    python -m guardian.demo > docs/assets/cli-demo-sample.txt
    python -m guardian.dashboard
