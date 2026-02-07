from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
import time

from migraph.graph.graph_store import GraphStore
from migraph.graph.graph_mutator import GraphMutator
from migraph.shock import ShockDetector, ShockScore
from migraph.config.settings import MigraphConfig

from migraph.embeddings.encoder import EmbeddingEncoder
from migraph.memory.replay import EpisodicRecall

from migraph.agents import (
    ConservativeAgent,
    ExplorerAgent,
    CausalAgent,
    SkepticAgent,
    JudgeAgent,
)

from migraph.rag.retriever import GraphRetriever
from migraph.rag.context_builder import ContextBuilder
from migraph.rag.synthesizer import AnswerSynthesizer
from migraph.rag.generator import Generator
from migraph.rag.entity_inferer import EntityInferer, EntityInferenceConfig
from migraph.rag.retriever import RetrievedContext
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import logging
import inspect

from migraph.memory.episodic import EpisodicMemory, Episode

from migraph.bias.bias_profiles import (
    NeutralBias,
    RiskAverseBias,
    OptimisticBias,
    SkepticalBias,
    BiasApplier,
)


class GraphRAGService:
    """
    Policy-aware orchestration layer for migraph.

    This is the ONLY place where:
    - config is interpreted
    - policies are enforced
    - subsystems are wired
    """

    def __init__(
        self,
        *,
        base_graph: GraphStore,
        memory: EpisodicMemory,
        generator: Generator,
        embedding_encoder: EmbeddingEncoder,
        config: MigraphConfig,
        embeddings_path: Path | None = None,
    ) -> None:

        self.base_graph = base_graph
        self.memory = memory
        self.generator = generator
        self.encoder = embedding_encoder
        self.config = config
        self.embeddings_path = embeddings_path

        # ---------------- Recall ----------------

        self.recall = EpisodicRecall(
            encoder=self.encoder,
        )

        # ---------------- Shock ----------------

        self.shock_detector = ShockDetector(
            graph=base_graph,
            encoder=self.encoder,
            config=config.shock,
            embeddings_path=embeddings_path,
        )

        # ---------------- Mutation ----------------

        self.mutator = GraphMutator(
            base_graph=base_graph,
            config=config.mutation,
        )

        # ---------------- Retrieval ----------------

        self.retriever = GraphRetriever(
            graph_config=config.graph,
            encoder=self.encoder,
            embeddings_path=embeddings_path,
        )

        self.context_builder = ContextBuilder()

        self.entity_inferer = EntityInferer(
            graph=base_graph,
            encoder=self.encoder,
            embeddings_path=embeddings_path,
            config=EntityInferenceConfig(
                max_entities=config.entity_max_entities
            ),
        )

        # ---------------- Agents ----------------

        self.judge = JudgeAgent(
            agents=[
                ConservativeAgent(),
                ExplorerAgent(),
                CausalAgent(),
                SkepticAgent(),
            ]
        )

        # ---------------- Synthesis ----------------

        self.synthesizer = AnswerSynthesizer(
            generator=self.generator,
            uncertainty_config=config.uncertainty,
        )

    def _serialize_recalled(self, recalled):
        if not recalled:
            return []
        serialized = []
        for item in recalled:
            if hasattr(item, "to_dict"):
                serialized.append(item.to_dict())
            else:
                serialized.append(item)
        return serialized

    def _query_entity_coverage(
        self,
        query: str,
        entities: List[str],
        context_entities: List[Dict[str, Any]] | None = None,
    ) -> float:
        import re

        def _alpha_tokens(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z]+", text.lower())
                if len(t) >= 3
            ]

        query_tokens = set(_alpha_tokens(query))
        if not query_tokens:
            return 0.0

        if context_entities:
            parts: List[str] = []
            for e in context_entities:
                parts.append(str(e.get("label") or ""))
                parts.append(str(e.get("type") or ""))
                parts.append(str(e.get("id") or ""))
            entity_tokens = set(_alpha_tokens(" ".join(parts)))
        else:
            entity_tokens = set(_alpha_tokens(" ".join(entities)))

        if not entity_tokens:
            return 0.0

        return len(query_tokens & entity_tokens) / max(len(query_tokens), 1)

    def _select_dominant_context(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not contexts:
            return {}
        def _score(ctx: Dict[str, Any]) -> tuple[float, float]:
            stats = ctx.get("retrieval_stats") or {}
            return (
                float(stats.get("edge_count") or 0),
                float(stats.get("node_count") or 0),
            )
        return max(contexts, key=_score)

    def _unsupported_terms(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> List[str]:
        import re

        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "from",
            "by",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "who",
            "whom",
            "does",
            "do",
            "did",
            "can",
            "could",
            "should",
            "would",
        }
        stop.update([s.lower() for s in (self.config.guard.stop_words or [])])

        def _tokens(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z]+", text.lower())
                if len(t) >= 3 and t not in stop
            ]

        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []

        parts: List[str] = []
        for e in context.get("entities") or []:
            parts.append(str(e.get("label") or ""))
            parts.append(str(e.get("type") or ""))
            parts.append(str(e.get("id") or ""))
        for r in context.get("relations") or []:
            parts.append(str(r.get("relation") or ""))
            parts.append(str(r.get("causal_type") or ""))
            parts.append(str(r.get("provenance") or ""))
        context_tokens = set(_tokens(" ".join(parts)))

        if not context_tokens:
            return sorted(query_tokens)

        unsupported = [t for t in query_tokens if t not in context_tokens]
        return sorted(set(unsupported))

    def _build_supported_answer(
        self,
        *,
        unsupported_terms: List[str],
        context: Dict[str, Any],
    ) -> str:
        import re

        sentences: List[str] = []

        if unsupported_terms:
            terms = ", ".join(unsupported_terms[:6])
            sentences.append(
                f"No evidence in the graph for: {terms}."
            )

        entities = context.get("entities") or []
        types = []
        for e in entities:
            t = e.get("type")
            if t and t not in types:
                types.append(t)
            if len(types) >= 3:
                break

        years: List[str] = []
        for e in entities:
            for field in (e.get("id"), e.get("label")):
                if not field:
                    continue
                for y in re.findall(r"\b(19\d{2}|20\d{2})\b", str(field)):
                    years.append(y)
        years = sorted(set(years))

        recovery_years: List[str] = []
        for e in entities:
            et = (e.get("type") or "").lower()
            if "recovery" not in et:
                continue
            for field in (e.get("id"), e.get("label")):
                if not field:
                    continue
                for y in re.findall(r"\b(19\d{2}|20\d{2})\b", str(field)):
                    recovery_years.append(y)
        recovery_years = sorted(set(recovery_years))

        if years:
            years_text = ", ".join(years[:6])
            if types:
                sentences.append(
                    f"Graph evidence includes years {years_text} across node types like {', '.join(types)}."
                )
            else:
                sentences.append(
                    f"Graph evidence includes years {years_text} in retrieved nodes."
                )

        if recovery_years:
            ry = ", ".join(recovery_years[:4])
            sentences.append(f"Recovery nodes indicate years {ry}.")

        if not sentences:
            return "Insufficient graph evidence to answer this question."

        if len(sentences) < 3:
            sentences.append("No additional supported causal links are present in the retrieved edges.")

        return " ".join(sentences[:4])

    def _prefix_unsupported_notice(
        self,
        *,
        answer: str,
        unsupported_terms: List[str],
    ) -> str:
        import re

        if not unsupported_terms:
            return answer
        terms = ", ".join(unsupported_terms[:6])
        prefix = f"No evidence in the graph for: {terms}."

        if answer.strip().startswith(prefix):
            return answer

        combined = f"{prefix} {answer}".strip()

        parts = re.split(r"(?<=[.!?])\s+", combined)
        return " ".join(parts[:4]).strip()

    def _ground_answer(
        self,
        *,
        answer: str,
        context: Dict[str, Any],
    ) -> str:
        import re

        def _tokens(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z]+", text.lower())
                if len(t) >= 3
            ]

        context_parts: List[str] = []
        for e in context.get("entities") or []:
            context_parts.append(str(e.get("label") or ""))
            context_parts.append(str(e.get("type") or ""))
            context_parts.append(str(e.get("id") or ""))
        for r in context.get("relations") or []:
            context_parts.append(str(r.get("relation") or ""))
            context_parts.append(str(r.get("causal_type") or ""))
            context_parts.append(str(r.get("provenance") or ""))

        context_tokens = set(_tokens(" ".join(context_parts)))
        if not context_tokens:
            return "Insufficient graph evidence to answer this question."

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        kept: List[str] = []
        for s in sentences:
            if not s:
                continue
            if s.startswith("No evidence in the graph for:"):
                kept.append(s)
                continue
            tokens = _tokens(s)
            if not tokens:
                continue
            overlap = len(set(tokens) & context_tokens) / max(len(tokens), 1)
            if overlap >= self.config.guard.min_sentence_overlap:
                kept.append(s)

        if not kept:
            return "Insufficient graph evidence to answer this question."

        return " ".join(kept[:4]).strip()

    def _filter_context_by_type_overlap(
        self,
        *,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import re

        query_tokens = set(re.findall(r"[a-z]+", query.lower()))
        if not query_tokens:
            return context

        entities = context.get("entities") or []
        relations = context.get("relations") or []

        type_overlap: Dict[str, int] = {}
        for e in entities:
            etype = str(e.get("type") or "").lower()
            label = str(e.get("label") or "").lower()
            if etype and (etype in query_tokens or any(t in label for t in query_tokens)):
                type_overlap[etype] = type_overlap.get(etype, 0) + 1

        if not type_overlap:
            return context

        min_overlap = int(self.config.guard.min_type_overlap or 1)
        keep_types = {t for t, c in type_overlap.items() if c >= min_overlap}
        if not keep_types:
            return context

        filtered_entities = [
            e for e in entities
            if str(e.get("type") or "").lower() in keep_types
        ]
        keep_ids = {e.get("id") for e in filtered_entities}
        filtered_relations = [
            r for r in relations
            if r.get("source") in keep_ids or r.get("target") in keep_ids
        ]

        new_ctx = dict(context)
        new_ctx["entities"] = filtered_entities
        new_ctx["relations"] = filtered_relations
        return new_ctx

    def _guard_answer(
        self,
        *,
        query: str,
        entities: List[str],
        shock: ShockScore,
        dominant_context: Dict[str, Any],
    ) -> bool:
        guard = self.config.guard
        if not guard.enabled:
            return False
        stats = dominant_context.get("retrieval_stats") or {}
        node_count = int(stats.get("node_count") or 0)
        edge_count = int(stats.get("edge_count") or 0)
        density = float(stats.get("retrieval_density") or 0.0)
        if node_count < guard.min_context_nodes:
            return True
        if edge_count < guard.min_context_edges:
            return True
        if density < guard.min_retrieval_density:
            return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _infer_entities(self, query: str) -> List[str]:
        candidates = self.entity_inferer.infer(query)
        return [c["id"] for c in candidates]

    def _fallback_retrieval(
        self,
        *,
        graphs: List[GraphStore],
        seed_nodes: List[str],
        reason: str,
    ) -> List[RetrievedContext]:
        contexts: List[RetrievedContext] = []
        for graph in graphs:
            edges = graph.get_edges_between(seed_nodes)
            contexts.append(
                RetrievedContext(
                    graph_variant=graph.metadata.get("variant", "unknown"),
                    node_ids=list(seed_nodes),
                    edges=edges,
                    metadata={
                        "node_count": len(seed_nodes),
                        "edge_count": len(edges),
                        "fallback": reason,
                    },
                )
            )
        return contexts

    def run(
        self,
        *,
        query: str,
        entities: List[str],
        bias: str | None = None,
    ) -> Dict[str, Any]:

        inferred_entities = self._infer_entities(query)
        merged_entities = list(dict.fromkeys([*entities, *inferred_entities]))
        entities = merged_entities

        # ===============================================================
        # 1. Episodic recall (semantic)
        # ===============================================================

        recalled = self.recall.recall(
            query=query,
            episodes=self.memory.all(),
        )
        recalled_payload = self._serialize_recalled(recalled)

        # ===============================================================
        # 2. Shock detection
        # ===============================================================

        t_shock = time.perf_counter()
        shock = self.shock_detector.score_query(query, entities)
        shock_ms = (time.perf_counter() - t_shock) * 1000.0

        shock_active = (
            self.config.shock.enabled
            and shock.overall >= self.config.shock.shock_threshold
        )

        # ===============================================================
        # 3. Bias application
        # ===============================================================

        graph = self.base_graph

        bias_profile = bias or (
            self.config.bias.profile if self.config.bias.enabled else None
        )

        if bias_profile is not None:
            bias_map = {
                "neutral": NeutralBias(),
                "risk_averse": RiskAverseBias(),
                "optimistic": OptimisticBias(),
                "skeptical": SkepticalBias(),
            }

            applier = BiasApplier(
                profile=bias_map[bias_profile],
                amplification_factor=self.config.bias.amplification_factor,
            )

            graph = applier.apply(graph)

        # ===============================================================
        # 4. Shock escalation / mutation
        # ===============================================================

        if shock_active:
            policy = self.config.shock.escalation_policy

            if policy == "halt":
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "uncertainty": {
                        "reason": "halted_due_to_shock",
                    },
                    "shock": {
                        "overall": shock.overall,
                        "components": shock.details,
                    },
                    "diagnostics": {
                        "status": "halted",
                        "recalled": recalled_payload,
                    },
                }

            if policy == "mutate" and self.config.mutation.enabled:
                graph_variants = self.mutator.generate_variants(
                    shock_score=shock.overall
                )[: self.config.mutation.max_variants]
            else:
                graph_variants = [graph]
        else:
            graph_variants = [graph]

        # ===============================================================
        # 5. Retrieval
        # ===============================================================

        logging.getLogger("migraph.retrieval").info(
            "retrieval start seeds=%s",
            len(entities),
        )

        timeout_ms = float(self.config.graph.retrieval_timeout_ms or 0)
        if timeout_ms > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(
                    self.retriever.retrieve,
                    graphs=graph_variants,
                    seed_nodes=entities,
                    query_text=query,
                )
                try:
                    retrieved_contexts = fut.result(timeout=timeout_ms / 1000.0)
                except FutureTimeout:
                    retrieved_contexts = self._fallback_retrieval(
                        graphs=graph_variants,
                        seed_nodes=entities,
                        reason="retrieval_timeout",
                    )
        else:
            retrieved_contexts = self.retriever.retrieve(
                graphs=graph_variants,
                seed_nodes=entities,
                query_text=query,
            )

        # ===============================================================
        # 6. Context building
        # ===============================================================

        contexts = [
            self._filter_context_by_type_overlap(
                query=query,
                context=self.context_builder.build(g, r),
            )
            for g, r in zip(graph_variants, retrieved_contexts)
        ]
        for ctx in contexts:
            ctx["unsupported_terms"] = self._unsupported_terms(query, ctx)

        dominant_context = self._select_dominant_context(contexts)
        unsupported_terms = self._unsupported_terms(query, dominant_context)
        if self._guard_answer(
            query=query,
            entities=entities,
            shock=shock,
            dominant_context=dominant_context,
        ):
            dominant_entities = [
                {
                    "id": e.get("id"),
                    "label": e.get("label"),
                    "type": e.get("type"),
                }
                for e in (dominant_context.get("entities") or [])[:25]
            ]
            dominant_edges_raw = [
                {
                    "source": r.get("source"),
                    "target": r.get("target"),
                    "relation": r.get("relation"),
                    "causal_type": r.get("causal_type"),
                    "confidence": (r.get("strength") or {}).get("confidence"),
                    "weight": (r.get("strength") or {}).get("weight"),
                    "provenance": r.get("provenance"),
                }
                for r in (dominant_context.get("relations") or [])
            ]
            dominant_edges = sorted(
                dominant_edges_raw,
                key=lambda e: (e.get("confidence") or 0.0, e.get("weight") or 0.0),
                reverse=True,
            )[:20]
            return {
                "answer": "Insufficient graph evidence to answer this question.",
                "confidence": 0.0,
                "uncertainty": {
                    "reason": "insufficient_graph_evidence",
                },
                "shock": {
                    "overall": shock.overall,
                    "components": shock.details,
                },
                "diagnostics": {
                    "judge": {},
                    "episode_id": "",
                    "recalled": recalled_payload,
                    "dominant_entities": dominant_entities,
                    "dominant_edges": dominant_edges,
                    "dominant_retrieval_stats": dominant_context.get("retrieval_stats", {}),
                    "inferred_entities": inferred_entities,
                    "entity_inference": self.entity_inferer.last_info,
                },
            }

        # ===============================================================
        # 7. Agent evaluation
        # ===============================================================

        judge_output = self.judge.evaluate(
            graphs=graph_variants,
            query_context={
                "query": query,
                "shock": shock,
                "recalled": recalled_payload,
            },
        )

        scores = [r["score"] for r in judge_output["agent_results"]]

        # ===============================================================
        # 8. Synthesis + LLM
        # ===============================================================

        final = self.synthesizer.synthesize(
            query=query,
            contexts=contexts,
            scores=scores,
            bias=bias_profile,
            shock={
                "overall": shock.overall,
                "components": shock.details,
            },
        )

        if unsupported_terms:
            final["answer"] = self._prefix_unsupported_notice(
                answer=final.get("answer", ""),
                unsupported_terms=unsupported_terms,
            )
            final["answer"] = self._ground_answer(
                answer=final.get("answer", ""),
                context=dominant_context,
            )

        dominant_context = final.get("dominant_context", {}) or {}
        dominant_entities = [
            {
                "id": e.get("id"),
                "label": e.get("label"),
                "type": e.get("type"),
            }
            for e in (dominant_context.get("entities") or [])[:25]
        ]
        dominant_edges_raw = [
            {
                "source": r.get("source"),
                "target": r.get("target"),
                "relation": r.get("relation"),
                "causal_type": r.get("causal_type"),
                "confidence": (r.get("strength") or {}).get("confidence"),
                "weight": (r.get("strength") or {}).get("weight"),
                "provenance": r.get("provenance"),
            }
            for r in (dominant_context.get("relations") or [])
        ]
        dominant_edges = sorted(
            dominant_edges_raw,
            key=lambda e: (e.get("confidence") or 0.0, e.get("weight") or 0.0),
            reverse=True,
        )[:20]
        logging.getLogger("migraph.retrieval").info(
            "dominant_entities=%s dominant_edges=%s",
            [e.get("id") for e in dominant_entities],
            [
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "relation": e.get("relation"),
                    "confidence": e.get("confidence"),
                    "weight": e.get("weight"),
                }
                for e in dominant_edges
            ],
        )

        # ===============================================================
        # 9. Episodic memory write
        # ===============================================================

        episode = Episode.create(
            query=query,
            entities=entities,
            answer=final["answer"],
            confidence=final["confidence"],
            uncertainty=final["uncertainty"],
            shock={
                "overall": shock.overall,
                "components": shock.details,
            },
            graph_variants=[
                g.metadata.get("variant", "base")
                for g in graph_variants
            ],
            dominant_agents=judge_output["dominant_agents"],
            bias=bias_profile or "neutral",
            embedding=self.encoder.encode_one(query),
        )

        self.memory.add(episode)

        # ===============================================================
        # 10. Final response
        # ===============================================================

        result = {
            "answer": final["answer"],
            "confidence": final["confidence"],
            "uncertainty": final["uncertainty"],
            "shock": {
                "overall": shock.overall,
                "components": shock.details,
            },
            "diagnostics": {
                "judge": judge_output,
                "episode_id": episode.id,
                "recalled": recalled_payload,
                "dominant_entities": dominant_entities,
                "dominant_edges": dominant_edges,
                "dominant_retrieval_stats": dominant_context.get("retrieval_stats", {}),
                "inferred_entities": inferred_entities,
                "entity_inference": self.entity_inferer.last_info,
            },
        }
        if unsupported_terms:
            result["diagnostics"]["unsupported_terms"] = unsupported_terms
        return result

    def run_stream(
        self,
        *,
        query: str,
        entities: List[str],
        bias: str | None = None,
    ):
        """
        Streaming pipeline that yields token-level chunks and metadata.
        """

        inferred_entities = self._infer_entities(query)
        merged_entities = list(dict.fromkeys([*entities, *inferred_entities]))
        entities = merged_entities

        # ===============================================================
        # 1. Episodic recall (semantic)
        # ===============================================================

        recalled = self.recall.recall(
            query=query,
            episodes=self.memory.all(),
        )
        recalled_payload = self._serialize_recalled(recalled)

        # ===============================================================
        # 2. Shock detection
        # ===============================================================
        logging.getLogger("migraph.recall").info("entities=%s", entities)
        t_shock = time.perf_counter()
        shock = self.shock_detector.score_query(query, entities)
        shock_ms = (time.perf_counter() - t_shock) * 1000.0

        shock_active = (
            self.config.shock.enabled
            and shock.overall >= self.config.shock.shock_threshold
        )
        logging.getLogger("migraph.shock").info(
            "shock detected: overall=%.4f active=%s ms=%.2f",
            shock.overall,
            shock_active,
            shock_ms,
        )

        # ===============================================================
        # 3. Bias application
        # ===============================================================

        graph = self.base_graph

        bias_profile = bias or (
            self.config.bias.profile if self.config.bias.enabled else None
        )

        if bias_profile is not None:
            bias_map = {
                "neutral": NeutralBias(),
                "risk_averse": RiskAverseBias(),
                "optimistic": OptimisticBias(),
                "skeptical": SkepticalBias(),
            }

            applier = BiasApplier(
                profile=bias_map[bias_profile],
                amplification_factor=self.config.bias.amplification_factor,
            )

            graph = applier.apply(graph)

        # ===============================================================
        # 4. Shock escalation / mutation
        # ===============================================================

        if shock_active:
            policy = self.config.shock.escalation_policy

            if policy == "halt":
                yield {
                    "type": "metadata",
                    "data": {
                        "answer": "",
                        "confidence": 0.0,
                        "uncertainty": {"reason": "halted_due_to_shock"},
                        "shock": {
                            "overall": shock.overall,
                            "components": shock.details,
                        },
                        "diagnostics": {
                            "status": "halted",
                            "recalled": recalled_payload,
                        },
                    },
                }
                return
            logging.getLogger("migraph.shock").info(
                "shock active: overall=%.4f policy=%s",
                shock.overall,
                policy,
            )

            if policy == "mutate" and self.config.mutation.enabled:
                graph_variants = self.mutator.generate_variants(
                    shock_score=shock.overall
                )[: self.config.mutation.max_variants]
            else:
                graph_variants = [graph]
            logging.getLogger("migraph.shock").info(
                "generated %s graph variants due to shock escalation",
                len(graph_variants),
            )
        else:
            graph_variants = [graph]
            logging.getLogger("migraph.shock").info(
                "shock not active (overall=%.4f), proceeding without mutation",
                shock.overall,
            )

        # ===============================================================
        # 5. Retrieval
        # ===============================================================

        logging.getLogger("migraph.retrieval").info(
            "retrieval start (stream) seeds=%s",
            len(entities),
        )

        t_retrieval = time.perf_counter()
        timeout_ms = float(self.config.graph.retrieval_timeout_ms or 0)
        if timeout_ms > 0:
            with ThreadPoolExecutor(max_workers=1) as executor:
                fut = executor.submit(
                    self.retriever.retrieve,
                    graphs=graph_variants,
                    seed_nodes=entities,
                    query_text=query,
                )
                try:
                    retrieved_contexts = fut.result(timeout=timeout_ms / 1000.0)
                except FutureTimeout:
                    retrieved_contexts = self._fallback_retrieval(
                        graphs=graph_variants,
                        seed_nodes=entities,
                        reason="retrieval_timeout",
                    )
        else:
            retrieved_contexts = self.retriever.retrieve(
                graphs=graph_variants,
                seed_nodes=entities,
                query_text=query,
            )
        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000.0

        # ===============================================================
        # 6. Context building
        # ===============================================================

        t_context = time.perf_counter()
        contexts = [
            self._filter_context_by_type_overlap(
                query=query,
                context=self.context_builder.build(g, r),
            )
            for g, r in zip(graph_variants, retrieved_contexts)
        ]
        for ctx in contexts:
            ctx["unsupported_terms"] = self._unsupported_terms(query, ctx)
        context_ms = (time.perf_counter() - t_context) * 1000.0

        dominant_context = self._select_dominant_context(contexts)
        unsupported_terms = self._unsupported_terms(query, dominant_context)

        if self._guard_answer(
            query=query,
            entities=entities,
            shock=shock,
            dominant_context=dominant_context,
        ):
            dominant_entities = [
                {
                    "id": e.get("id"),
                    "label": e.get("label"),
                    "type": e.get("type"),
                }
                for e in (dominant_context.get("entities") or [])[:25]
            ]
            dominant_edges_raw = [
                {
                    "source": r.get("source"),
                    "target": r.get("target"),
                    "relation": r.get("relation"),
                    "causal_type": r.get("causal_type"),
                    "confidence": (r.get("strength") or {}).get("confidence"),
                    "weight": (r.get("strength") or {}).get("weight"),
                    "provenance": r.get("provenance"),
                }
                for r in (dominant_context.get("relations") or [])
            ]
            dominant_edges = sorted(
                dominant_edges_raw,
                key=lambda e: (e.get("confidence") or 0.0, e.get("weight") or 0.0),
                reverse=True,
            )[:20]
            yield {
                "type": "metadata",
                "data": {
                    "answer": "Insufficient graph evidence to answer this question.",
                    "confidence": 0.0,
                    "uncertainty": {"reason": "insufficient_graph_evidence"},
                    "shock": {
                        "overall": shock.overall,
                        "components": shock.details,
                    },
                    "diagnostics": {
                        "judge": {},
                        "episode_id": "",
                        "recalled": recalled_payload,
                        "dominant_entities": dominant_entities,
                        "dominant_edges": dominant_edges,
                        "dominant_retrieval_stats": dominant_context.get("retrieval_stats", {}),
                        "inferred_entities": inferred_entities,
                        "entity_inference": self.entity_inferer.last_info,
                    },
                },
            }
            return

        # ===============================================================
        # 7. Agent evaluation
        # ===============================================================

        t_judge = time.perf_counter()
        judge_output = self.judge.evaluate(
            graphs=graph_variants,
            query_context={
                "query": query,
                "shock": shock,
                "recalled": recalled_payload,
            },
        )
        judge_ms = (time.perf_counter() - t_judge) * 1000.0

        scores = [r["score"] for r in judge_output["agent_results"]]

        # ===============================================================
        # 8. Prepare synthesis
        # ===============================================================

        t_prepare = time.perf_counter()
        prepared = self.synthesizer.prepare(
            query=query,
            contexts=contexts,
            scores=scores,
            bias=bias_profile,
            shock={
                "overall": shock.overall,
                "components": shock.details,
            },
        )
        prepare_ms = (time.perf_counter() - t_prepare) * 1000.0

        if not prepared.get("contexts"):
            yield {
                "type": "metadata",
                "data": {
                    "answer": "",
                    "confidence": 0.0,
                    "uncertainty": {},
                    "shock": {
                        "overall": shock.overall,
                        "components": shock.details,
                    },
                    "diagnostics": {
                        "judge": judge_output,
                        "episode_id": "",
                        "recalled": recalled_payload,
                        "dominant_entities": [],
                        "dominant_edges": [],
                        "dominant_retrieval_stats": {},
                        "inferred_entities": inferred_entities,
                        "entity_inference": self.entity_inferer.last_info,
                    },
                },
            }
            return

        dominant_context = prepared.get("dominant_context", {})
        if hasattr(self.generator, "extract_evidence"):
            context_text = str(dominant_context)
            if len(context_text) <= 1000:
                evidence = self.generator.extract_evidence(
                    query=query,
                    context=dominant_context,
                    uncertainty=prepared.get("uncertainty", {}),
                    bias=bias_profile,
                    shock={
                        "overall": shock.overall,
                        "components": shock.details,
                    },
                )
                if evidence:
                    dominant_context = dict(dominant_context)
                    dominant_context["evidence"] = evidence
        dominant_entities = [
            {
                "id": e.get("id"),
                "label": e.get("label"),
                "type": e.get("type"),
            }
            for e in (dominant_context.get("entities") or [])[:25]
        ]
        dominant_edges_raw = [
            {
                "source": r.get("source"),
                "target": r.get("target"),
                "relation": r.get("relation"),
                "causal_type": r.get("causal_type"),
                "confidence": (r.get("strength") or {}).get("confidence"),
                "weight": (r.get("strength") or {}).get("weight"),
                "provenance": r.get("provenance"),
            }
            for r in (dominant_context.get("relations") or [])
        ]
        dominant_edges = sorted(
            dominant_edges_raw,
            key=lambda e: (e.get("confidence") or 0.0, e.get("weight") or 0.0),
            reverse=True,
        )[:20]
        logging.getLogger("migraph.retrieval").info(
            "dominant_entities=%s dominant_edges=%s",
            [e.get("id") for e in dominant_entities],
            [
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "relation": e.get("relation"),
                    "confidence": e.get("confidence"),
                    "weight": e.get("weight"),
                }
                for e in dominant_edges
            ],
        )

        # ===============================================================
        # 9. Streaming generation
        # ===============================================================

        answer_parts: List[str] = []
        stream = self.generator.stream_generate(
            query=query,
            context=dominant_context,
            uncertainty=prepared.get("uncertainty", {}),
            bias=bias_profile,
            shock={
                "overall": shock.overall,
                "components": shock.details,
            },
        )

        for chunk in stream:
            answer_parts.append(chunk)
            yield {"type": "chunk", "data": chunk}

        answer_text = "".join(answer_parts)
        if unsupported_terms:
            answer_text = self._prefix_unsupported_notice(
                answer=answer_text,
                unsupported_terms=unsupported_terms,
            )
            answer_text = self._ground_answer(
                answer=answer_text,
                context=dominant_context,
            )

        # ===============================================================
        # 10. Episodic memory write
        # ===============================================================

        episode = Episode.create(
            query=query,
            entities=entities,
            answer=answer_text,
            confidence=prepared.get("confidence", 0.0),
            uncertainty=prepared.get("uncertainty", {}),
            shock={
                "overall": shock.overall,
                "components": shock.details,
            },
            graph_variants=[
                g.metadata.get("variant", "base")
                for g in graph_variants
            ],
            dominant_agents=judge_output["dominant_agents"],
            bias=bias_profile or "neutral",
            embedding=self.encoder.encode_one(query),
        )

        self.memory.add(episode)

        # ===============================================================
        # 11. Final metadata event
        # ===============================================================

        def _sanitize(value):
            if isinstance(value, list):
                return [_sanitize(v) for v in value]
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            if hasattr(value, "__dict__"):
                return _sanitize(value.__dict__)
            return value

        diagnostics = {
            "judge": judge_output,
            "episode_id": episode.id,
            "recalled": recalled_payload,
            "dominant_entities": dominant_entities,
            "dominant_edges": dominant_edges,
            "dominant_retrieval_stats": dominant_context.get("retrieval_stats", {}),
            "inferred_entities": inferred_entities,
            "entity_inference": self.entity_inferer.last_info,
        }
        if unsupported_terms:
            diagnostics["unsupported_terms"] = unsupported_terms

        yield {
            "type": "metadata",
            "data": _sanitize(
                {
                    "answer": answer_text,
                    "confidence": prepared.get("confidence", 0.0),
                    "uncertainty": prepared.get("uncertainty", {}),
                    "shock": {
                        "overall": shock.overall,
                        "components": shock.details,
                    },
                    "diagnostics": diagnostics,
                }
            ),
        }
