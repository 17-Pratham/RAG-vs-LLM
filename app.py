import streamlit as st
import pandas as pd
from task import *

st.set_page_config(page_title="LLM vs RAG", layout="centered")
st.title("Resolute.ai | LLM vs RAG Comparison Tool")
question = st.text_input("Ask a question about Resolute.ai:")

if question:
    with st.spinner("Generating answers..."):
        llm_response = for_llm(question)
        rag_response = for_rag(question)

    st.subheader("LLM-only Answer")
    st.write(llm_response)

    st.subheader("RAG-based Answer")
    st.write(rag_response)

    st.subheader(" Rate the Answers")

    col1, col2 = st.columns(2)
    with col1:
        llm_acc = st.radio("LLM Accuracy", [1, 0])
        llm_rel = st.slider("LLM Relevance (1-5)", 1, 5, 3)
    with col2:
        rag_acc = st.radio("RAG Accuracy", [1, 0])
        rag_rel = st.slider("RAG Relevance (1-5)", 1, 5, 3)

    if st.button("Save Ratings"):
        df = pd.DataFrame([{
            "question": question,
            "llm_accuracy": llm_acc,
            "rag_accuracy": rag_acc,
            "llm_relevance": llm_rel,
            "rag_relevance": rag_rel
        }])
        df.to_csv("results.csv", mode="a", index=False, header=not st.session_state.get("header_written", False))
        st.session_state.header_written = True
        st.success("✅ Ratings saved successfully!")
