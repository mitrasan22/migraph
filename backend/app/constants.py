DEFAULTS = {
    # LLM output length limit (tokens)
    "LLM_MAX_NEW_TOKENS": 180,
    # LLM sampling temperature
    "LLM_TEMPERATURE": 0.2,
    # Nucleus sampling threshold
    "LLM_TOP_P": 0.9,
    # Penalize repetition in generation
    "LLM_REPETITION_PENALTY": 1.2,
    # Prevent repeating n-grams of this size
    "LLM_NO_REPEAT_NGRAM_SIZE": 5,
    # Enable/disable answer guardrails
    "ANSWER_GUARD_ENABLED": True,
    # Minimum nodes required in dominant context
    "ANSWER_GUARD_MIN_CONTEXT_NODES": 3,
    # Minimum edges required in dominant context
    "ANSWER_GUARD_MIN_CONTEXT_EDGES": 1,
    # Minimum retrieval density threshold
    "ANSWER_GUARD_MIN_RETRIEVAL_DENSITY": 0.0,
    # Minimum fraction of query tokens covered by context
    "ANSWER_GUARD_MIN_QUERY_COVERAGE": 0.2,
    # Shock threshold for guard-based refusal
    "ANSWER_GUARD_SHOCK_THRESHOLD": 0.5,
    # Minimum semantic novelty for answer acceptance
    "ANSWER_GUARD_MIN_SEMANTIC_NOVELTY": 0.6,
    # Minimum token novelty for answer acceptance
    "ANSWER_GUARD_MIN_TOKEN_NOVELTY": 0.6,
    # Minimum token overlap to keep a sentence during grounding
    "ANSWER_GUARD_MIN_SENTENCE_OVERLAP": 0.08,
    # Stop words ignored when detecting unsupported terms
    "ANSWER_GUARD_STOP_WORDS": [
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
    # Minimum type overlap between query and entity type/label
    "ANSWER_GUARD_MIN_TYPE_OVERLAP": 1,
    # Max inferred entities to seed retrieval
    "ENTITY_MAX_ENTITIES": 8,
    # Retrieval timeout in milliseconds (0 = no timeout)
    "RETRIEVAL_TIMEOUT_MS": 5000,
    # Whether to embed edges in the embedding index
    "EMBED_EDGES": True,
    # Persist embeddings after computing
    "PERSIST_EMBEDDINGS": False,
    # Persist embeddings if missing on disk
    "PERSIST_EMBEDDINGS_IF_MISSING": True,
    # Rebuild embeddings if dimension/model mismatch detected
    "REBUILD_EMBEDDINGS_ON_MISMATCH": False,
    # Default backend API URL for UI
    "API_URL": "http://localhost:8000",
}
