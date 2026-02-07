import streamlit as st

from services.api_client import fetch_graph


def _parse_entities(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def render_query_box(api_url: str):
    st.subheader("Query")
    query = st.text_area("Question", height=120, placeholder="Ask a question...")

    use_dropdown = st.checkbox("Pick entities from graph", value=True)

    entities: list[str] = []
    if use_dropdown:
        try:
            data = fetch_graph(api_url)
            nodes = data.get("nodes", [])
        except Exception:
            nodes = []

        type_options = sorted({n.get("type", "unknown") for n in nodes})
        selected_types = st.multiselect(
            "Filter by type",
            options=type_options,
            default=type_options,
        )

        search = st.text_input("Search nodes", value="")
        search_lower = search.lower().strip()

        filtered = []
        for n in nodes:
            n_type = n.get("type", "unknown")
            if n_type not in selected_types:
                continue
            label = str(n.get("label", n["id"]))
            node_id = str(n["id"])
            if search_lower and search_lower not in label.lower() and search_lower not in node_id.lower():
                continue
            filtered.append(n)

        options = [f"{n['id']} — {n.get('label', n['id'])} ({n.get('type','unknown')})" for n in filtered]
        selected = st.multiselect(
            "Entities (id — label)",
            options=options,
            default=[],
        )

        entities = [s.split(" — ")[0] for s in selected]
    else:
        entities_raw = st.text_input(
            "Entities (comma-separated)",
            value="",
            placeholder="e.g., united_states, india, aapl",
        )
        entities = _parse_entities(entities_raw)

    col_a, _ = st.columns([1, 3])
    with col_a:
        run = st.button("Run", type="primary", use_container_width=True)

    return query, entities, run
