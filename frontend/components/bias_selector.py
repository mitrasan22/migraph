import streamlit as st


def render_bias_selector(default: str = "neutral") -> str:
    options = ["neutral", "risk_averse", "optimistic", "skeptical"]
    index = options.index(default) if default in options else 0
    return st.selectbox(
        "Bias profile",
        options=options,
        index=index,
    )
