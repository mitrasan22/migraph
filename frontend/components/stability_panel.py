import streamlit as st


def render_stability_panel(uncertainty: dict | None = None):
    st.subheader("Uncertainty")
    box = st.container()

    stability = 0.0
    entropy = 0.0
    agreement = 0.0
    if uncertainty:
        stability = float(uncertainty.get("stability", {}).get("stability", 0.0))
        entropy = float(uncertainty.get("entropy", 0.0))
        agreement = float(uncertainty.get("agreement", 0.0))

    with box:
        st.metric("Stability", f"{stability:.2f}")
        st.metric("Entropy", f"{entropy:.2f}")
        st.metric("Agreement", f"{agreement:.2f}")

        st.bar_chart(
            {
                "stability": [stability],
                "1-entropy": [max(1.0 - entropy, 0.0)],
                "agreement": [agreement],
            }
        )

    return box
