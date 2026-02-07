from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from migraph.uncertainty.stability import StabilityAnalyzer
from migraph.uncertainty.entropy import EntropyAnalyzer
from migraph.rag.generator import Generator
from migraph.config.settings import UncertaintyConfig


class AnswerSynthesizer:
    """
    Synthesizes epistemic state across graph variants
    and produces the final answer via LLM generation.
    Incorporates uncertainty metrics into confidence scoring.
    """

    def __init__(
        self,
        *,
        generator: Generator,
        uncertainty_config: UncertaintyConfig,
    ) -> None:
        self.generator = generator
        self.config = uncertainty_config
        self.stability_analyzer = StabilityAnalyzer()
        self.entropy_analyzer = EntropyAnalyzer()

    def synthesize(
        self,
        *,
        query: str,
        contexts: List[Dict[str, Any]],
        scores: List[float],
        bias: str | None = None,
        shock: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        prepared = self.prepare(
            query=query,
            contexts=contexts,
            scores=scores,
            bias=bias,
            shock=shock,
        )

        if not prepared["contexts"]:
            return {
                "answer": "",
                "confidence": 0.0,
                "uncertainty": {},
                "diagnostics": {},
                "variants": [],
            }

        dominant_context = prepared["dominant_context"]
        if hasattr(self.generator, "extract_evidence"):
            context_text = str(dominant_context)
            if len(context_text) <= 1000:
                evidence = self.generator.extract_evidence(
                    query=query,
                    context=dominant_context,
                    uncertainty=prepared["uncertainty"],
                    bias=bias,
                    shock=shock or {},
                )
                if evidence:
                    dominant_context = dict(dominant_context)
                    dominant_context["evidence"] = evidence

        answer_text = self.generator.generate(
            query=query,
            context=dominant_context,
            uncertainty=prepared["uncertainty"],
            bias=bias,
            shock=shock or {},
        )

        return {
            "answer": answer_text,
            "confidence": prepared["confidence"],
            "uncertainty": prepared["uncertainty"],
            "diagnostics": prepared["diagnostics"],
            "variants": prepared["contexts"],
            "dominant_context": dominant_context,
        }

    def prepare(
        self,
        *,
        query: str,
        contexts: List[Dict[str, Any]],
        scores: List[float],
        bias: str | None = None,
        shock: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        if not contexts or not scores or len(scores) == 0:
            return {
                "confidence": 0.0,
                "uncertainty": {},
                "diagnostics": {},
                "dominant_context": {},
                "contexts": [],
            }
        if len(scores) < len(contexts):
            contexts = contexts[: len(scores)]
        elif len(scores) > len(contexts):
            scores = scores[: len(contexts)]

        # -------------------------------------------------
        # Uncertainty computation
        # -------------------------------------------------

        if self.config.enabled:
            stability = self.stability_analyzer.compute(
                [
                    {
                        "score": score,
                        "variant": ctx.get("graph_variant", "unknown"),
                    }
                    for score, ctx in zip(scores, contexts)
                ]
            )

            entropy = self.entropy_analyzer.compute(scores)
            agreement = self._agreement(scores)
        else:
            stability = {"stability": float(np.mean(scores))}
            entropy = 0.0
            agreement = 1.0

        # -------------------------------------------------
        # Confidence base (mode-driven)
        # -------------------------------------------------

        if self.config.confidence_mode == "stability":
            confidence = stability["stability"]
        elif self.config.confidence_mode == "entropy":
            confidence = max(1.0 - entropy, 0.0)
        else:  # hybrid
            confidence = max(
                0.5 * stability["stability"] + 0.5 * agreement,
                0.0,
            )

        penalties: List[str] = []

        if agreement < self.config.min_agreement_ratio:
            confidence *= agreement
            penalties.append("low_agent_agreement")

        if entropy > self.config.entropy_threshold:
            confidence *= max(1.0 - entropy, 0.0)
            penalties.append("high_variant_entropy")

        confidence = max(confidence, 0.0)

        # -------------------------------------------------
        # Dominant context selection
        # -------------------------------------------------

        best_idx = int(np.argmax(scores))
        dominant_context = contexts[best_idx]

        # -------------------------------------------------
        # LLM generation
        # -------------------------------------------------

        uncertainty = {
            "stability": stability,
            "entropy": entropy,
            "agreement": agreement,
            "penalties": penalties,
        }

        diagnostics = {
            "confidence_mode": self.config.confidence_mode,
            "min_agreement_ratio": self.config.min_agreement_ratio,
            "entropy_threshold": self.config.entropy_threshold,
            "selected_variant": dominant_context.get("graph_variant"),
            "score_distribution": scores,
        }

        return {
            "confidence": confidence,
            "uncertainty": uncertainty,
            "diagnostics": diagnostics,
            "dominant_context": dominant_context,
            "contexts": contexts,
        }

    def _agreement(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        return max(1.0 - float(np.std(scores)), 0.0)
