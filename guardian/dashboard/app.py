"""Streamlit dashboard demo using synthetic sample data only."""

from __future__ import annotations

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        'Streamlit is required for the dashboard demo. Install with: pip install -e ".[simulator]"'
    ) from exc

from guardian.demo.cli import PRIVACY_NOTE, SAMPLE_MESSAGES, evaluate_sample

st.set_page_config(
    page_title="MindMend Guardian Demo",
    page_icon="🛡️",
    layout="wide",
)

st.title("MindMend Guardian")
st.caption(
    "Privacy-first, local-first AI safety prototype · synthetic demo data only · "
    "Built by Michigan MindMend Inc."
)

st.info(
    "This dashboard uses sample text only. It does not store raw messages, "
    "require cloud services, or claim clinical validation."
)

selected = st.selectbox("Choose a synthetic sample message", SAMPLE_MESSAGES)
result = evaluate_sample(selected)

col1, col2, col3 = st.columns(3)
col1.metric("Risk level", result.risk_level.upper())
col2.metric("Human review needed", "Yes" if result.human_review_recommended else "No")
col3.metric("Escalation suggested", "Yes" if result.should_escalate else "No")

st.subheader("Risk signals")
if result.matched_categories:
    st.write(", ".join(result.matched_categories))
else:
    st.write("No elevated risk categories matched.")

st.subheader("Supportive response")
st.success(result.supportive_response)

st.subheader("Privacy notes")
st.write(result.privacy_note)

st.subheader("Audit hash (privacy-preserving)")
st.code(result.audit_hash, language="text")

with st.expander("All synthetic samples"):
    for message in SAMPLE_MESSAGES:
        sample = evaluate_sample(message)
        st.markdown(
            f"**{message}** → `{sample.risk_level}` · review="
            f"{'yes' if sample.human_review_recommended else 'no'}"
        )

st.caption(PRIVACY_NOTE)
