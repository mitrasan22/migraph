"""
Configuration layer for migraph.

This module defines the authoritative configuration contracts that control
graph construction, inference-time mutation, agent orchestration, shock
handling, uncertainty computation, and bias conditioning.

Configuration in migraph is:
- Explicit (passed, not global)
- Typed (validated at construction time)
- Stable (backward compatible by design)
"""

from migraph.config.settings import (
    GraphConfig,
    ShockConfig,
    MutationConfig,
    UncertaintyConfig,
    BiasConfig,
    MigraphConfig,
)

__all__ = [
    "GraphConfig",
    "ShockConfig",
    "MutationConfig",
    "UncertaintyConfig",
    "BiasConfig",
    "MigraphConfig",
]
