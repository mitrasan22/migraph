import streamlit as st

from services.api_client import fetch_history


def _apply_filters(episodes, *, bias, min_conf, shock_range, query_text):
    out = []
    for ep in episodes:
        if bias != "all" and ep.get("bias") != bias:
            continue
        if ep.get("confidence", 0.0) < min_conf:
            continue
        shock = ep.get("shock_overall", 0.0)
        if not (shock_range[0] <= shock <= shock_range[1]):
            continue
        if query_text and query_text.lower() not in ep.get("query", "").lower():
            continue
        out.append(ep)
    return out


def render_memory_timeline(api_url: str):
    st.subheader("Memory Timeline")
    qp = st.query_params
    if "memory_limit" in qp and "memory_limit" not in st.session_state:
        st.session_state["memory_limit"] = int(qp["memory_limit"])
    if "memory_bias" in qp and "memory_bias" not in st.session_state:
        st.session_state["memory_bias"] = qp["memory_bias"]
    if "memory_min_conf" in qp and "memory_min_conf" not in st.session_state:
        st.session_state["memory_min_conf"] = float(qp["memory_min_conf"])
    if "memory_shock" in qp and "memory_shock_range" not in st.session_state:
        parts = [p.strip() for p in qp["memory_shock"].split(",")]
        if len(parts) == 2:
            st.session_state["memory_shock_range"] = (float(parts[0]), float(parts[1]))
    if "memory_query" in qp and "memory_query_text" not in st.session_state:
        st.session_state["memory_query_text"] = qp["memory_query"]

    if "memory_limit" not in st.session_state:
        st.session_state["memory_limit"] = 25
    limit = st.slider(
        "Episodes",
        min_value=5,
        max_value=200,
        value=st.session_state["memory_limit"],
        step=5,
    )
    st.session_state["memory_limit"] = limit

    filter_col_a, filter_col_b = st.columns([1, 1])
    with filter_col_a:
        if "memory_bias" not in st.session_state:
            st.session_state["memory_bias"] = "all"
        if "memory_min_conf" not in st.session_state:
            st.session_state["memory_min_conf"] = 0.0

        bias = st.selectbox(
            "Bias filter",
            options=["all", "neutral", "risk_averse", "optimistic", "skeptical"],
            index=["all", "neutral", "risk_averse", "optimistic", "skeptical"].index(
                st.session_state["memory_bias"]
            ),
        )
        min_conf = st.slider(
            "Min confidence",
            0.0,
            1.0,
            st.session_state["memory_min_conf"],
            0.05,
        )
    with filter_col_b:
        if "memory_shock_range" not in st.session_state:
            st.session_state["memory_shock_range"] = (0.0, 1.0)
        if "memory_query_text" not in st.session_state:
            st.session_state["memory_query_text"] = ""

        shock_range = st.slider(
            "Shock range",
            0.0,
            1.0,
            st.session_state["memory_shock_range"],
            0.05,
        )
        query_text = st.text_input(
            "Search query",
            value=st.session_state["memory_query_text"],
        )

    st.session_state["memory_bias"] = bias
    st.session_state["memory_min_conf"] = min_conf
    st.session_state["memory_shock_range"] = shock_range
    st.session_state["memory_query_text"] = query_text

    st.query_params["memory_limit"] = str(limit)
    st.query_params["memory_bias"] = bias
    st.query_params["memory_min_conf"] = f"{min_conf:.2f}"
    st.query_params["memory_shock"] = f"{shock_range[0]:.2f},{shock_range[1]:.2f}"
    st.query_params["memory_query"] = query_text

    if st.button("Refresh memory", use_container_width=True):
        st.session_state["memory_refresh"] = True

    if "memory_refresh" not in st.session_state:
        st.session_state["memory_refresh"] = True

    if st.session_state["memory_refresh"]:
        try:
            st.session_state["memory_data"] = fetch_history(api_url, limit=limit)
        except Exception as exc:
            st.error(f"Failed to load memory: {exc}")
            st.session_state["memory_data"] = []
        st.session_state["memory_refresh"] = False

    episodes = st.session_state.get("memory_data", [])
    episodes = _apply_filters(
        episodes,
        bias=bias,
        min_conf=min_conf,
        shock_range=shock_range,
        query_text=query_text,
    )

    if not episodes:
        st.caption("No episodes match the current filters.")
        return

    # Sparkline timeline
    series = {
        "confidence": [e.get("confidence", 0.0) for e in episodes],
        "shock": [e.get("shock_overall", 0.0) for e in episodes],
    }
    st.line_chart(series)

    compare_options = [
        f"{ep['timestamp']} • {ep['query']} • {ep['id']}" for ep in episodes
    ]
    compare_selected = st.multiselect(
        "Compare episodes",
        options=compare_options,
        default=[],
    )

    if compare_selected:
        st.subheader("Comparison")
        selected_ids = {s.split(" • ")[-1] for s in compare_selected}
        selected = [ep for ep in episodes if ep["id"] in selected_ids]
        cols = st.columns(len(selected))
        for col, ep in zip(cols, selected):
            with col:
                st.markdown(f"**{ep['timestamp']}**")
                st.markdown(f"**Confidence:** {ep['confidence']:.2f}")
                st.markdown(f"**Shock:** {ep['shock_overall']:.2f}")
                st.markdown(f"**Bias:** {ep['bias']}")
                st.markdown(f"**Entities:** {', '.join(ep['entities'])}")
                st.markdown("**Answer:**")
                st.write(ep["answer"])

    for ep in reversed(episodes):
        title = f"{ep['timestamp']} • {ep['query']}"
        with st.expander(title, expanded=False):
            st.markdown(f"**Confidence:** {ep['confidence']:.2f}")
            st.markdown(f"**Shock:** {ep['shock_overall']:.2f}")
            st.markdown(f"**Bias:** {ep['bias']}")
            st.markdown(f"**Entities:** {', '.join(ep['entities'])}")
            st.markdown("**Answer:**")
            st.write(ep["answer"])
