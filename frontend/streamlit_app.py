import streamlit as st
from dynaconf import Dynaconf

from components.query_box import render_query_box
from components.answer_panel import render_answer_panel
from components.shock_gauge import render_shock_gauge
from components.stability_panel import render_stability_panel
from components.bias_selector import render_bias_selector
from components.graph_view import render_graph_view
from components.memory_timeline import render_memory_timeline
from services.api_client import stream_query, query_once, fetch_graph_stats


settings = Dynaconf(
    envvar_prefix="MIGRAPH",
    load_dotenv=True,
    settings_files=[],
)

st.set_page_config(page_title="migraph UI", layout="wide")

with open("frontend/styles/theme.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "qp_init" not in st.session_state:
    qp = st.query_params
    for key in ["api_url", "use_streaming", "bias"]:
        if key in qp:
            st.session_state[key] = qp[key]
    st.session_state["qp_init"] = True

st.markdown('<div class="app-title">migraph</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Mutable-inference GraphRAG with streaming answers</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Backend")
    default_url = settings.get("API_URL", "http://localhost:8000")
    api_url = st.text_input(
        "API base URL",
        value=st.session_state.get("api_url", default_url),
    )
    use_streaming = st.checkbox(
        "Streaming response",
        value=st.session_state.get("use_streaming", "true") == "true",
    )

    st.divider()
    st.subheader("Bias")
    bias = render_bias_selector(default=st.session_state.get("bias", "neutral"))

    st.divider()
    st.subheader("Backend Status")
    if st.button("Check status", use_container_width=True):
        st.session_state["backend_status_refresh"] = True

    if "backend_status_refresh" not in st.session_state:
        st.session_state["backend_status_refresh"] = True

    if st.session_state["backend_status_refresh"]:
        try:
            stats = fetch_graph_stats(api_url)
            st.session_state["backend_stats"] = stats
            st.session_state["backend_status_error"] = None
        except Exception as exc:
            st.session_state["backend_stats"] = None
            st.session_state["backend_status_error"] = str(exc)
        st.session_state["backend_status_refresh"] = False

    if st.session_state.get("backend_status_error"):
        st.error(f"Backend error: {st.session_state['backend_status_error']}")
    else:
        stats = st.session_state.get("backend_stats")
        if stats:
            st.metric("Nodes", stats.get("nodes", 0))
            st.metric("Edges", stats.get("edges", 0))
        else:
            st.caption("No stats yet.")

st.query_params["api_url"] = api_url
st.query_params["use_streaming"] = "true" if use_streaming else "false"
st.query_params["bias"] = bias

query, entities, run = render_query_box(api_url)

col_left, col_right = st.columns([2, 1])
with col_left:
    answer_box, status_box = render_answer_panel()
with col_right:
    shock_box = render_shock_gauge(st.session_state.get("last_shock"))
    stability_box = render_stability_panel(st.session_state.get("last_uncertainty"))

st.divider()
render_graph_view(api_url)
st.divider()
render_memory_timeline(api_url)

if run:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    payload = {
        "query": query,
        "entities": entities,
        "bias": bias,
    }

    if use_streaming:
        status_box.info("Streaming response...")
        answer = ""
        last_meta = None
        for event in stream_query(api_url, payload):
            event_type = event.get("type")
            if event_type == "chunk":
                answer += event.get("data", "")
                answer_box.markdown(answer)
                continue
            if event_type == "metadata":
                last_meta = event.get("data", {})
                continue
            if event_type == "error":
                status_box.error(event.get("data", "Streaming error"))
                break

        if last_meta:
            st.session_state["last_shock"] = last_meta.get("shock")
            st.session_state["last_uncertainty"] = last_meta.get("uncertainty")
            if not answer:
                answer = last_meta.get("answer", "")
                answer_box.markdown(answer)

        status_box.success("Done")
    else:
        status_box.info("Running query...")
        try:
            result = query_once(api_url, payload)
        except Exception as exc:
            status_box.error(f"Backend error: {exc}")
            st.stop()

        st.session_state["last_shock"] = result.get("shock")
        st.session_state["last_uncertainty"] = result.get("uncertainty")
        answer_box.markdown(result.get("answer", ""))
        status_box.success("Done")
