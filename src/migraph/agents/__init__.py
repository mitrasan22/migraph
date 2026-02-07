"""
Agent subsystem for migraph.

Agents represent distinct epistemic strategies applied to
graph-based reasoning. Each agent evaluates graph variants
from a different perspective.

The judge agent aggregates and reconciles agent outputs.
"""

from migraph.agents.base_agent import BaseAgent, AgentResult
from migraph.agents.conservative_agent import ConservativeAgent
from migraph.agents.explorer_agent import ExplorerAgent
from migraph.agents.causal_agent import CausalAgent
from migraph.agents.skeptic_agent import SkepticAgent
from migraph.agents.judge_agent import JudgeAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "ConservativeAgent",
    "ExplorerAgent",
    "CausalAgent",
    "SkepticAgent",
    "JudgeAgent",
]
