import time
import streamlit as st


def render_shock_gauge(shock: dict | None = None):
    st.subheader("Shock")
    box = st.container()

    current = 0.0
    if shock is not None:
        current = float(shock.get("overall", 0.0))

    prev = st.session_state.get("shock_prev", 0.0)
    st.session_state["shock_prev"] = current

    with box:
        st.metric("Overall", f"{current:.2f}")
        bar = st.progress(0)

        steps = 12
        for i in range(steps):
            val = prev + (current - prev) * ((i + 1) / steps)
            bar.progress(min(max(val, 0.0), 1.0))
            time.sleep(0.02)

    return box
