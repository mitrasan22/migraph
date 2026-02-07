from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Callable


class GenerationBackend(Protocol):
    """
    Mandatory text generation backend.

    Any implementation MUST:
    - accept a prompt string
    - return generated text
    """

    def generate(self, prompt: str) -> str: ...


class Generator(ABC):
    """
    Abstract generation interface.

    This layer converts epistemic state
    into a final natural-language answer
    using an LLM backend.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context: Dict[str, Any],
        *,
        uncertainty: Dict[str, Any],
        bias: str,
        shock: Dict[str, Any],
    ) -> str:
        raise NotImplementedError

    def stream_generate(
        self,
        query: str,
        context: Dict[str, Any],
        *,
        uncertainty: Dict[str, Any],
        bias: str,
        shock: Dict[str, Any],
    ):
        """
        Optional streaming generation interface.
        """
        raise NotImplementedError


PromptBuilder = Callable[
    [
        str,  # query
        Dict[str, Any],  # context
        Dict[str, Any],  # uncertainty
        str,  # bias
        Dict[str, Any],  # shock
    ],
    str,
]


class LLMGenerator(Generator):
    """
    LLM-backed generator.

    """

    def __init__(
        self,
        backend: GenerationBackend,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        if backend is None:
            raise ValueError("LLM backend must be provided")

        self.backend = backend
        self.prompt_builder = prompt_builder

    def generate(
        self,
        query: str,
        context: Dict[str, Any],
        *,
        uncertainty: Dict[str, Any],
        bias: str,
        shock: Dict[str, Any],
    ) -> str:

        prompt = (
            self.prompt_builder(
                query,
                context,
                uncertainty,
                bias,
                shock,
            )
            if self.prompt_builder is not None
            else self._default_prompt(
                query=query,
                context=context,
                uncertainty=uncertainty,
                bias=bias,
                shock=shock,
            )
        )

        return self.backend.generate(prompt)

    def extract_evidence(
        self,
        query: str,
        context: Dict[str, Any],
        *,
        uncertainty: Dict[str, Any],
        bias: str,
        shock: Dict[str, Any],
    ) -> str:
        prompt = f"""
You are extracting ONLY evidence statements from a graph context.

Rules:
- Output 5-8 short bullet points.
- Each bullet must be directly supported by the context.
- Do not invent facts.
- If evidence is missing, output "No direct evidence in context."

Question:
{query}

Graph-derived context:
{context}

Evidence bullets:
""".strip()
        return self.backend.generate(prompt)

    def stream_generate(
        self,
        query: str,
        context: Dict[str, Any],
        *,
        uncertainty: Dict[str, Any],
        bias: str,
        shock: Dict[str, Any],
    ):
        prompt = (
            self.prompt_builder(
                query,
                context,
                uncertainty,
                bias,
                shock,
            )
            if self.prompt_builder is not None
            else self._default_prompt(
                query=query,
                context=context,
                uncertainty=uncertainty,
                bias=bias,
                shock=shock,
            )
        )

        if hasattr(self.backend, "generate_stream"):
            return self.backend.generate_stream(prompt)

        # Fallback: chunk the full answer
        text = self.backend.generate(prompt)
        return (text[i : i + 24] for i in range(0, len(text), 24))

    def _default_prompt(
        self,
        *,
        query: str,
        context: Dict[str, Any],
        uncertainty: Dict[str, Any],
        bias: str,
        shock: Dict[str, Any],
    ) -> str:

        # Build a structured, compact context so the model stays within max length.
        max_context_chars = 700

        def compact_entities(entities: list[dict]) -> list[str]:
            lines = []
            allow_keys = {
                "cases_per_m",
                "deaths_per_m",
                "total_cases",
                "total_deaths",
                "peak_year",
                "year",
                "value",
            }
            for e in entities[:15]:
                attrs = e.get("attributes") or {}
                keep = {k: attrs.get(k) for k in allow_keys if k in attrs}
                lines.append(
                    f"- id={e.get('id')}, label={e.get('label')}, type={e.get('type')}, attrs={keep}"
                )
            return lines

        def compact_relations(relations: list[dict]) -> list[str]:
            lines = []
            for r in relations[:20]:
                strength = r.get("strength") or {}
                lines.append(
                    f"- {r.get('source')} --{r.get('relation')}--> {r.get('target')} "
                    f"(type={r.get('causal_type')}, conf={strength.get('confidence')}, "
                    f"weight={strength.get('weight')})"
                )
            return lines

        if isinstance(context, list):
            blocks = []
            for i, ctx in enumerate(context, start=1):
                entities = ctx.get("entities") or []
                relations = ctx.get("relations") or []
                stats = ctx.get("retrieval_stats") or {}
                blocks.append(f"Context[{i}]: graph_variant={ctx.get('graph_variant')}")
                blocks.append("Entities:\n" + "\n".join(compact_entities(entities)))
                if relations:
                    blocks.append("Relations:\n" + "\n".join(compact_relations(relations)))
                if stats:
                    blocks.append(f"Stats: {stats}")
            context_text = "\n".join(blocks)
        else:
            context_text = str(context)

        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars] + "\n[TRUNCATED CONTEXT]"

        unsupported_terms = []
        if isinstance(context, dict):
            unsupported_terms = context.get("unsupported_terms") or []
        evidence_text = ""
        if isinstance(context, dict):
            evidence_text = context.get("evidence") or ""
        if evidence_text and len(context_text) > 500:
            evidence_text = ""
        unsupported_text = (
            f"Unsupported terms: {', '.join(unsupported_terms)}"
            if unsupported_terms else "Unsupported terms: none"
        )

        return f"""
You are generating an answer using a knowledge graph and prior reasoning.

STRICT RULES:
- Use ONLY the provided context.
- Do NOT invent facts.
- Reflect uncertainty explicitly.
- Respect the specified bias.
- If shock is high, be conservative.
- If the question asks "when", "year", or about trends, include specific years and values.
- If the context does not contain the needed years/values, say "Insufficient temporal data in context."
- Write exactly 3-4 sentences.
- Cite at least two indicators or node types from the context (e.g., covid_year, gdp_year, inflation_year, recovery nodes).
- When comparing, mention which recovered first and the time gap.
- Do NOT repeat the question or use tautologies.
- If context lacks any causal or comparative evidence, explicitly say "Insufficient evidence in graph context." and explain what is missing.
- If unsupported terms are present, explicitly say there is no evidence for those terms, then answer only the supported part of the question.
- Prefer concrete, specific statements grounded in entity attributes (e.g., deaths_per_m, cases_per_m, recovery year).
- If the retrieved context has 0 relations, say the graph lacks linking edges and decline to over-interpret.
- Do not list more than 3 items unless explicitly asked.
- Ignore entities not explicitly mentioned in the question unless they are required to connect two mentioned entities.
- If evidence bullets are provided, ONLY use those bullets as your factual basis.

Question:
{query}

Graph-derived context (authoritative):
{context_text}

{unsupported_text}

Evidence bullets:
{evidence_text}

Uncertainty metrics:
{uncertainty}

Bias profile:
{bias}

Shock signal:
{shock}

Final answer:
""".strip()
