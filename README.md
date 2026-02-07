# migraph

migraph is a mutable-inference GraphRAG library that enables shock-aware, counterfactual, uncertainty-aware, and bias-conditioned reasoning by dynamically modifying the knowledge graph during inference.

Unlike traditional GraphRAG systems that treat graphs as static retrieval structures, migraph treats the graph as an active reasoning substrate.

---

## Core Idea

**Reason by editing the graph, not just retrieving from it.**

For a single user question, migraph:

1. Detects novelty and causal complexity (shock)
2. Creates multiple temporary graph variants
3. Mutates graph structure during inference
4. Runs GraphRAG reasoning on each variant
5. Compares answers across graph worlds
6. Returns the answer with structural confidence and uncertainty

---

## What migraph Enables

- **Shock-aware reasoning**  
  Automatically adapts traversal and reasoning strategy for novel or out-of-distribution queries.

- **Counterfactual reasoning**  
  Answers "what if this relationship did not exist?" by structurally removing edges and re-running inference.

- **Uncertainty awareness**  
  Confidence is derived from explanation stability across graph variants, not LLM token probabilities.

- **Bias-conditioned reasoning**  
  Different cognitive or risk biases are implemented as graph edge re-weighting policies.

---

## Public API

The public API is intentionally small and explicit:

```python
from migraph import GraphStore, GraphBuilder, GraphMutator, ShockDetector
```

All other modules are considered internal and may evolve.

---

## Conceptual Example

```python
graph = GraphStore()
GraphBuilder(graph).build_from_data(data)

shock = ShockDetector(graph).score(query)

graph_variants = GraphMutator(graph).generate_variants(shock)

answers = [run_graphrag(g, query) for g in graph_variants]

final_answer = aggregate_answers(answers)
```

The difference between answers is as important as the answer itself.

---

## Datasets Tested (Notebooks)

These datasets are used in the included notebooks and ETL pipeline:

- COVID-19 (Our World in Data): `data/raw/covid/owid-covid-latest.csv` and `data/raw/covid/owid-covid-data.csv`
- World Bank indicators (multi-country CSVs): GDP, GDP per capita, inflation, unemployment
- Finance sample (tickers): `data/raw/finance/*.csv`
- Wikipedia sample (simplewiki dump)

Notebooks:

- `notebooks/datasets_download.ipynb` (downloads raw datasets)
- `notebooks/etl_processed.ipynb` (builds graph nodes/edges/embeddings)

---

## Datasets This Library Can Use

migraph works best with datasets that can be represented as entities and relationships:

- Economic indicators (World Bank, IMF, OECD)
- Health and epidemiology (WHO, OWID, CDC)
- Finance and markets (companies, tickers, exposures, supply chains)
- Knowledge bases (Wikipedia, Wikidata)
- Policy, regulation, and public sector data
- Scientific domains with temporal signals (climate, energy, emissions)

---

## Real-World Use Cases

- **Jira blocker analysis (engineering ops)**  
  Link tickets, services, owners, incidents, and dependencies to identify true blockers, likely root causes, and what will unlock a release.

- **Incident reasoning**  
  Combine service graphs, SLOs, rollout timelines, and change logs to explain why an outage spread and which rollback path is safest.

- **Cross-team dependency planning**  
  Connect initiatives, platform changes, and team ownership to surface hidden coupling and risk before launch.

- **Risk and shock analysis**  
  Detect structural breaks and adapt reasoning based on novelty or shocks.

- **Counterfactual reasoning**  
  Remove or reweight edges (e.g., supply chain links or service dependencies) to test alternate worlds.

---

## Architecture

End-to-end pipeline (data → graph → reasoning → UI):

```text
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ Raw Datasets                 │    │ Notebooks / ETL              │
│ covid / world_bank / finance │───▶│ datasets_download + etl       │
│ wikipedia                    │    │ nodes.parquet / edges.parquet │
└──────────────────────────────┘    └──────────────────────────────┘
                                                │
                                                v
┌───────────────────────────────────────────────────────────────┐
│ Graph Store (NetworkX)                                        │
│ - GraphBuilder loads nodes/edges/embeddings                    │
│ - GraphMutator edits edges for bias + counterfactual variants  │
└───────────────────────────────────────────────────────────────┘
                                                │
                                                v
┌───────────────────────────────────────────────────────────────┐
│ GraphRAG Inference                                            │
│ ShockDetector → GraphMutator → Retriever → ContextBuilder     │
│ AnswerSynthesizer → LLM Generator → Episodic Memory           │
└───────────────────────────────────────────────────────────────┘
                                                │
                                                v
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ FastAPI Backend              │    │ Streamlit UI                 │
│ /query, /query/stream         │───▶│ Query + streaming answer     │
│ /graph/export, /graph/stats   │    │ Graph view + memory timeline │
└──────────────────────────────┘    └──────────────────────────────┘
```

Internal component view (core modules and responsibilities):

```text
GraphBuilder  -> loads processed nodes/edges/embeddings
GraphStore    -> stores graph, node/edge access, traversal
GraphMutator  -> creates graph variants for bias/counterfactuals
ShockDetector -> detects novelty/complexity, tunes mutation strategy
Retriever     -> retrieves candidate subgraphs per variant
ContextBuilder-> turns subgraph into structured context
Synthesizer   -> scores variants, picks dominant context
Generator     -> produces answer from context + uncertainty
Memory        -> saves episodes for timeline + replay
```

---

## UI Flow (User Journey)

1. User selects backend URL, bias, and answer length.
2. User enters a question and optionally selects entities.
3. UI sends `/query` or `/query/stream` to backend.
4. Streaming responses render token-by-token in the Answer panel.
5. Metadata updates the Shock gauge and Stability panel.
6. Graph view refreshes from `/graph/export`.
7. Memory timeline loads episodes from `/query/history`.

---

## UI Components (What Each Does)

- `Query Box`  
  Collects question text and entity IDs. Supports search and grouping by node type.

- `Answer Panel`  
  Renders the final answer. Supports JSON length selection (short/normal/detailed).

- `Shock Gauge`  
  Shows novelty/complexity signal for the current query.

- `Stability Panel`  
  Shows uncertainty metrics across graph variants.

- `Graph View`  
  Visualizes the current graph subset and node types. Uses `/graph/export`.

- `Memory Timeline`  
  Shows past episodes (question, confidence, shock, bias). Uses `/query/history`.

---

## Walkthrough (Example Session)

1. Open the UI and set `API base URL` to your backend.
2. Choose `Bias` (neutral, conservative, skeptic, explorer).
3. Pick an `Answer Length` (short/normal/detailed).
4. Enter a question and select a few entities.
5. Click `Run` to stream the answer.
6. Inspect Shock and Stability panels for uncertainty.
7. Open `Graph View` to see connected nodes and edges.
8. Use `Memory Timeline` to compare your past queries.

---

## Design Principles

- Graphs are mutable only during inference
- Reasoning is multi-agent and comparative
- Confidence is structural, not probabilistic
- Public API is minimal and explicit

migraph is designed for decision intelligence, causal reasoning research, and agentic AI systems.

---

## Library Structure

The `migraph` library is organized as follows:

```plaintext
src/
└── migraph/                       # CORE LIBRARY (reusable)
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    ├── graph/
    │   ├── __init__.py
    │   ├── graph_schema.py        # Node / Edge definitions
    │   ├── graph_store.py         # Graph backend (NetworkX / Neo4j)
    │   ├── graph_builder.py       # Build graph from raw data
    │   ├── graph_query.py         # Traversal & retrieval
    │   └── graph_mutator.py       # inference-time graph mutation
    ├── embeddings/
    │   ├── __init__.py
    │   ├── encoder.py
    │   └── similarity.py
    ├── shock/
    │   ├── __init__.py
    │   ├── shock_detector.py      # novelty / shock detection
    │   └── shock_score.py
    ├── agents/
    │   ├── __init__.py
    │   ├── base_agent.py
    │   ├── conservative_agent.py
    │   ├── explorer_agent.py
    │   ├── causal_agent.py
    │   ├── skeptic_agent.py
    │   └── judge_agent.py
    ├── rag/
    │   ├── __init__.py
    │   ├── retriever.py           # subgraph ensemble retrieval
    │   ├── context_builder.py
    │   ├── generator.py           # LLM wrapper
    │   └── synthesizer.py
    ├── uncertainty/
    │   ├── __init__.py
    │   ├── stability.py           # explanation stability
    │   └── entropy.py
    ├── bias/
    │   ├── __init__.py
    │   └── bias_profiles.py       # bias -> edge reweight rules
    ├── memory/
    │   ├── __init__.py
    │   ├── episodic.py
    │   └── replay.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── confidence.py
    │   └── metrics.py
    └── utils/
        ├── __init__.py
        ├── text.py
        ├── time.py
        └── helpers.py
```
---

## Status

**Alpha** — core abstractions are stabilizing and may evolve.

---

## License

MIT License

---

## Author

**Santanu Mitra**
