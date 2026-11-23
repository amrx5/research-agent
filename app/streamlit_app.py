"""Minimal demo UI:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from agent import Config, ResearchAgent  # noqa: E402

st.set_page_config(page_title="Research Agent", layout="wide")
st.title("Multi-tool research assistant")


@st.cache_resource
def get_agent() -> ResearchAgent:
    return ResearchAgent(Config())


question = st.text_area("Question", placeholder="e.g. What BLEU did the original Transformer report on WMT14 EN-DE, "
                        "what is the best reported score now, and what is the % improvement?")
if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Planning, calling tools, validating..."):
        ans = get_agent().run(question.strip(), question_id="ui")
    st.markdown(ans.to_markdown())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tools", " -> ".join(ans.tools_called) or "none")
    c2.metric("Latency", f"{ans.latency_s:.1f} s")
    c3.metric("Tokens", f"{ans.tokens_in + ans.tokens_out:,}")
    c4.metric("Cost", f"${ans.cost_usd:.4f}")
    if ans.trace_path:
        with st.expander("Execution trace"):
            st.code(Path(ans.trace_path).read_text(encoding="utf-8"), language="json")
