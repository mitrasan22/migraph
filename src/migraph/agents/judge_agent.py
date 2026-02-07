from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from migraph.agents.base_agent import AgentResult, BaseAgent
from migraph.graph.graph_store import GraphStore
from migraph.uncertainty.stability import StabilityAnalyzer
from migraph.uncertainty.entropy import EntropyAnalyzer


class JudgeAgent:
    """
    Aggregates and reconciles agent evaluations across graph variants.

    Principles:
    - Epistemic confidence comes from stability across graph variants
    - Agent agreement is diagnostic, not authoritative
    - Rationales and signals are preserved for transparency
    """

    def __init__(self, agents: List[BaseAgent]) -> None:
        self.agents = agents
        self.stability_analyzer = StabilityAnalyzer()
        self.entropy_analyzer = EntropyAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        graphs: List[GraphStore],
        query_context: Dict[str, Any],
    ) -> Dict[str, Any]:

        results: List[AgentResult] = []

        for graph in graphs:
            for agent in self.agents:
                results.append(agent.evaluate(graph, query_context))

        scores = [r.score for r in results]

        stability = self.stability_analyzer.compute(
            [
                {
                    "score": r.score,
                    "variant": r.graph_variant,
                }
                for r in results
            ]
        )

        entropy = self.entropy_analyzer.compute(scores)

        agent_confidence = self._confidence(scores)
        agent_agreement = self._agreement(scores)

        agent_outputs = [
            {
                "agent": r.agent,
                "graph_variant": r.graph_variant,
                "score": r.score,
                "rationale": r.rationale,
                "signals": r.signals,
            }
            for r in results
        ]

        return {
            "confidence": stability["stability"],
            "uncertainty": {
                "stability": stability,
                "entropy": entropy,
            },
            "diagnostics": {
                "agent_confidence": agent_confidence,
                "agent_agreement": agent_agreement,
            },
            "agent_results": agent_outputs,
            "dominant_agents": self._dominant_agents(results),
        }

    def _confidence(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        return max(float(np.mean(scores) - np.std(scores)), 0.0)

    def _agreement(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        return max(1.0 - float(np.std(scores)), 0.0)

    def _dominant_agents(
        self,
        results: List[AgentResult],
    ) -> List[str]:

        if not results:
            return []

        grouped: Dict[str, List[float]] = {}

        for r in results:
            grouped.setdefault(r.agent, []).append(r.score)

        avg_scores = {
            agent: float(np.mean(scores)) for agent, scores in grouped.items()
        }

        max_score = max(avg_scores.values())
        return [agent for agent, score in avg_scores.items() if score == max_score]
