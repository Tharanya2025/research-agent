import os
import tempfile

import pandas as pd
import streamlit as st

import llm
from agent import run_agent
from litreview import review_papers
from rag import get_paper_text, ingest_pdf, list_sources
from reproducibility import check_reproducibility
from verifier import verify_answer

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 Research Assistant Agent")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat = []  # (role, text, verification or None, model name or None)

COLORS = {"supported": "green", "partially_supported": "orange", "unsupported": "red"}


def show_verification(v):
    st.progress(min(max(v["score"], 0.0), 1.0), text=f"Faithfulness score: {v['score']:.0%}")
    for c in v["claims"]:
        color = COLORS.get(c.get("verdict"), "gray")
        st.markdown(f":{color}[{c.get('claim', '')}]  \n*{c.get('evidence', '')}*")


# ---------- Sidebar: upload papers and settings ----------
with st.sidebar:
    st.header("Your papers")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index documents"):
        for f in files:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                with st.spinner(f"Indexing {f.name}..."):
                    n = ingest_pdf(path)
            st.success(f"{f.name}: {n} chunks")

    verify_on = st.toggle("Verify answers", value=True, help="Turn off for faster answers while testing.")

    if st.button("🗑️ New chat"):
        st.session_state.history = []
        st.session_state.chat = []
        st.rerun()

    st.caption("Indexed: " + (", ".join(list_sources()) or "nothing yet"))

sources = list_sources()
tab_chat, tab_repro, tab_lit = st.tabs(["💬 Ask", "🧪 Reproducibility", "📚 Literature review"])

# ---------- Tab 1: chat + verifier ----------
with tab_chat:
    for role, text, ver, model_name in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)
            if model_name:
                st.caption(f"Answered by {model_name}")
            if ver:
                with st.expander("Verification"):
                    show_verification(ver)

    if q := st.chat_input("Ask about your papers..."):
        st.chat_message("user").write(q)
        try:
            with st.spinner("Researching (this can take a minute if Gemini is busy)..."):
                answer, st.session_state.history, log, evidence = run_agent(q, st.session_state.history)
            used_model = getattr(llm, "LAST_MODEL", None)

            ver = None
            if verify_on and evidence and not answer.startswith("I couldn't"):
                with st.spinner("Verifying the answer..."):
                    try:
                        ver = verify_answer(answer, "\n\n".join(evidence))
                    except Exception as e:
                        st.warning(f"Verification failed: {e}")

            with st.chat_message("assistant"):
                with st.expander("Tools used"):
                    for name, args in log:
                        st.code(f"{name}({args})")
                st.write(answer)
                if used_model:
                    st.caption(f"Answered by {used_model}")
                if ver:
                    with st.expander("Verification", expanded=True):
                        show_verification(ver)

            st.session_state.chat += [("user", q, None, None), ("assistant", answer, ver, used_model)]
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------- Tab 2: reproducibility ----------
with tab_repro:
    if not sources:
        st.info("Upload and index a paper first.")
    else:
        paper = st.selectbox("Choose a paper", sources, key="repro_paper")
        if st.button("Check reproducibility"):
            try:
                with st.spinner("Auditing paper..."):
                    res = check_reproducibility(get_paper_text(paper))
                st.metric("Reproducibility score", f"{res['score']:.0%}")
                st.dataframe(pd.DataFrame(res["rows"]))
                if res["missing"]:
                    st.warning("Not reported: " + ", ".join(res["missing"]))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------- Tab 3: literature review ----------
with tab_lit:
    if len(sources) < 2:
        st.info("Index at least two papers to generate a literature review.")
    else:
        chosen = st.multiselect("Select papers (2 to 10)", sources)
        if st.button("Generate review"):
            if len(chosen) < 2:
                st.warning("Pick at least 2 papers.")
            else:
                try:
                    with st.spinner("Reading papers and writing the review..."):
                        summaries, draft = review_papers(chosen, get_paper_text)
                    st.subheader("Comparison table")
                    st.dataframe(pd.DataFrame(summaries))
                    st.subheader("Draft related-work section")
                    st.write(draft)
                    st.download_button("Download draft", draft, file_name="related_work.txt")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")