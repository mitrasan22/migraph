import streamlit as st

from services.api_client import fetch_graph


def render_graph_view(api_url: str):
    st.subheader("Graph View")
    qp = st.query_params
    if "graph_max_nodes" in qp and "graph_max_nodes" not in st.session_state:
        st.session_state["graph_max_nodes"] = int(qp["graph_max_nodes"])
    if "graph_max_edges" in qp and "graph_max_edges" not in st.session_state:
        st.session_state["graph_max_edges"] = int(qp["graph_max_edges"])
    if "graph_types" in qp and "graph_type_filter" not in st.session_state:
        st.session_state["graph_type_filter"] = [
            t for t in qp["graph_types"].split(",") if t
        ]
    if "graph_max_nodes" not in st.session_state:
        st.session_state["graph_max_nodes"] = 24
    if "graph_max_edges" not in st.session_state:
        st.session_state["graph_max_edges"] = 40

    max_nodes = st.slider(
        "Max nodes",
        min_value=5,
        max_value=80,
        value=st.session_state["graph_max_nodes"],
        step=5,
    )
    max_edges = st.slider(
        "Max edges",
        min_value=5,
        max_value=120,
        value=st.session_state["graph_max_edges"],
        step=5,
    )
    st.session_state["graph_max_nodes"] = max_nodes
    st.session_state["graph_max_edges"] = max_edges

    if st.button("Refresh graph", use_container_width=True):
        st.session_state["graph_refresh"] = True

    if "graph_refresh" not in st.session_state:
        st.session_state["graph_refresh"] = True

    if st.session_state["graph_refresh"]:
        try:
            st.session_state["graph_data"] = fetch_graph(api_url)
        except Exception as exc:
            st.error(f"Failed to load graph: {exc}")
            st.session_state["graph_data"] = {"nodes": [], "edges": []}
        st.session_state["graph_refresh"] = False

    data = st.session_state.get("graph_data", {"nodes": [], "edges": []})
    all_nodes = data.get("nodes", [])
    all_edges = data.get("edges", [])

    types = sorted({n.get("type", "entity") for n in all_nodes})
    if "graph_type_filter" not in st.session_state:
        st.session_state["graph_type_filter"] = types

    st.markdown("**Node type filters**")
    selected_types = st.multiselect(
        "Show types",
        options=types,
        default=st.session_state["graph_type_filter"],
    )
    st.session_state["graph_type_filter"] = selected_types

    st.query_params["graph_max_nodes"] = str(max_nodes)
    st.query_params["graph_max_edges"] = str(max_edges)
    st.query_params["graph_types"] = ",".join(selected_types)

    legend_cols = st.columns(min(max(len(types), 1), 4))
    type_colors = {
        "entity": "#c2592d",
        "event": "#0e5e6f",
        "concept": "#8a6f3b",
        "system": "#3c6e47",
    }
    for i, t in enumerate(types):
        color = type_colors.get(t, "#6b5b95")
        legend_cols[i % len(legend_cols)].markdown(
            f"<div style='display:flex;align-items:center;gap:6px;'>"
            f"<span style='width:12px;height:12px;border-radius:3px;background:{color};display:inline-block;'></span>"
            f"<span>{t}</span></div>",
            unsafe_allow_html=True,
        )

    nodes = [n for n in all_nodes if n.get("type", "entity") in selected_types]
    nodes = nodes[:max_nodes]
    node_ids = {n["id"] for n in nodes}
    edges = [
        e
        for e in all_edges
        if e["source"] in node_ids and e["target"] in node_ids
    ][:max_edges]

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        node_objs = []
        for n in nodes:
            n_type = n.get("type", "entity")
            label = n.get("label", n["id"])
            color = type_colors.get(n_type, "#6b5b95")
            title = f"{label}\n{n_type}"
            node_objs.append(
                Node(
                    id=n["id"],
                    label=label,
                    size=18,
                    color=color,
                    title=title,
                )
            )

        edge_objs = [
            Edge(
                source=e["source"],
                target=e["target"],
                label=e.get("relation", ""),
            )
            for e in edges
        ]

        config = Config(
            width="100%",
            height=420,
            directed=True,
            physics=True,
            hierarchical=False,
        )
        agraph(nodes=node_objs, edges=edge_objs, config=config)
    except Exception:
        st.info("Interactive graph requires `streamlit-agraph`.")
