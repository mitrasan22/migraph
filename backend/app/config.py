from dataclasses import dataclass
from dynaconf import Dynaconf
from backend.app.constants import DEFAULTS

from migraph.config.settings import (
    GraphConfig,
    ShockConfig,
    MutationConfig,
    UncertaintyConfig,
    BiasConfig,
    AnswerGuardConfig,
    MigraphConfig,
)

settings = Dynaconf(
    envvar_prefix="MIGRAPH",
    load_dotenv=True,
    settings_files=[],
)
settings.update(DEFAULTS)

def _parse_csv(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return None


@dataclass(frozen=True)
class AppConfig:
    # ---------------- App ----------------
    app_name: str = settings.get("APP_NAME", "migraph-backend")
    api_prefix: str = settings.get("API_PREFIX", "")

    # ---------------- LLM ----------------
    llm_model: str = settings.get(
        "LLM_MODEL",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    hf_token: str = settings.get("HF_TOKEN")
    embedding_model: str = settings.get(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    embedding_device: str = settings.get("EMBEDDING_DEVICE", "cpu")

    max_new_tokens: int = settings.get("LLM_MAX_NEW_TOKENS", 512)
    temperature: float = settings.get("LLM_TEMPERATURE", 0.2)
    top_p: float = settings.get("LLM_TOP_P", 0.9)
    repetition_penalty: float = settings.get("LLM_REPETITION_PENALTY", 1.15)
    no_repeat_ngram_size: int = settings.get("LLM_NO_REPEAT_NGRAM_SIZE", 4)

    # ---------------- Migraph Policy ----------------
    migraph: MigraphConfig = MigraphConfig(
        graph=GraphConfig(
            max_hops=settings.get("GRAPH_MAX_HOPS", 2),
            allow_weak_edges=settings.get("GRAPH_ALLOW_WEAK_EDGES", False),
            min_edge_confidence=settings.get("GRAPH_MIN_EDGE_CONFIDENCE", 0.3),
            temporal_reasoning=settings.get("GRAPH_TEMPORAL_REASONING", False),
            retrieval_timeout_ms=settings.get("RETRIEVAL_TIMEOUT_MS", 0.0),
        ),
        shock=ShockConfig(
            enabled=settings.get("SHOCK_ENABLED", True),
            shock_threshold=settings.get("SHOCK_THRESHOLD", 0.4),
            novelty_window=settings.get("SHOCK_NOVELTY_WINDOW", 10),
            escalation_policy=settings.get("SHOCK_ESCALATION_POLICY", "mutate"),
        ),
        mutation=MutationConfig(
            enabled=settings.get("MUTATION_ENABLED", True),
            max_variants=settings.get("MUTATION_MAX_VARIANTS", 6),
            edge_decay_factor=settings.get("MUTATION_EDGE_DECAY_FACTOR", 0.85),
            allow_edge_removal=settings.get("MUTATION_ALLOW_EDGE_REMOVAL", True),
            allow_hypothetical_edges=settings.get("MUTATION_ALLOW_HYPOTHETICAL", True),
            mutation_strategy=settings.get("MUTATION_STRATEGY", "adaptive"),
        ),
        uncertainty=UncertaintyConfig(
            enabled=settings.get("UNCERTAINTY_ENABLED", True),
            min_agreement_ratio=settings.get("UNCERTAINTY_MIN_AGREEMENT_RATIO", 0.6),
            entropy_threshold=settings.get("UNCERTAINTY_ENTROPY_THRESHOLD", 0.8),
            confidence_mode=settings.get("CONFIDENCE_MODE", "hybrid"),
        ),
        bias=BiasConfig(
            enabled=settings.get("BIAS_ENABLED", False),
            profile=settings.get("BIAS_PROFILE", "neutral"),
            amplification_factor=settings.get("BIAS_AMPLIFICATION_FACTOR", 1.0),
        ),
        guard=AnswerGuardConfig(
            enabled=settings.get("ANSWER_GUARD_ENABLED", True),
            min_context_nodes=settings.get("ANSWER_GUARD_MIN_CONTEXT_NODES", 3),
            min_context_edges=settings.get("ANSWER_GUARD_MIN_CONTEXT_EDGES", 1),
            min_retrieval_density=settings.get("ANSWER_GUARD_MIN_RETRIEVAL_DENSITY", 0.0),
            min_query_coverage=settings.get("ANSWER_GUARD_MIN_QUERY_COVERAGE", 0.2),
            shock_threshold=settings.get("ANSWER_GUARD_SHOCK_THRESHOLD", 0.5),
            min_semantic_novelty=settings.get("ANSWER_GUARD_MIN_SEMANTIC_NOVELTY", 0.6),
            min_token_novelty=settings.get("ANSWER_GUARD_MIN_TOKEN_NOVELTY", 0.6),
            min_sentence_overlap=settings.get("ANSWER_GUARD_MIN_SENTENCE_OVERLAP", 0.2),
            stop_words=_parse_csv(
                settings.get(
                    "ANSWER_GUARD_STOP_WORDS",
                    [
                        "again",
                        "timeline",
                        "explain",
                        "impact",
                        "affect",
                        "effect",
                        "how",
                        "what",
                        "why",
                        "when",
                        "which",
                        "about",
                        "compare",
                        "cause",
                        "caused",
                        "because",
                        "due",
                        "during",
                        "after",
                        "before",
                        "between",
                        "increase",
                        "increased",
                        "decrease",
                        "decreased",
                        "recovery",
                        "recovered",
                    ],
                )
            ),
            min_type_overlap=settings.get("ANSWER_GUARD_MIN_TYPE_OVERLAP", 1),
        ),
        entity_max_entities=settings.get("ENTITY_MAX_ENTITIES", 8),
    )

    # ---------------- Data Paths ----------------
    data_dir: str = settings.get("DATA_DIR", "data")
    processed_dir: str = settings.get("PROCESSED_DIR", "data/processed")
    embeddings_path: str = settings.get(
        "EMBEDDINGS_PATH",
        "data/processed/embeddings.parquet",
    )

    # ---------------- Loader/Embedding Flags ----------------
    embed_edges: bool = settings.get("EMBED_EDGES", True)
    persist_embeddings: bool = settings.get("PERSIST_EMBEDDINGS", False)
    persist_embeddings_if_missing: bool = settings.get(
        "PERSIST_EMBEDDINGS_IF_MISSING",
        True,
    )
    rebuild_embeddings_on_mismatch: bool = settings.get(
        "REBUILD_EMBEDDINGS_ON_MISMATCH",
        False,
    )
