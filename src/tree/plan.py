# src/core/plan.py
"""
Pydantic schema for the JSON plan produced by the LLM.

The schema mirrors the tree DSL described in the architecture diagram:
- ActionNode, ConditionNode, CompositeNode (sequence/fallback), DecoratorNode.
- All nodes inherit from the union type ``PlanNode``.
- ``Plan`` wraps a root node and retains the original English goal.
"""

from __future__ import annotations

from typing import Literal, Any, List, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------


class ActionNode(BaseModel):
    """JSON action invocation; the action may expose a leaf or native subtree."""
    type: Literal["action"] = "action"
    id: str = ""  # Optional identifier for tracking
    action_type: str = ""  # Type of action (for routing)
    name: str  # Must match a registered action key
    params: dict[str, Any] = Field(default_factory=dict)


class ConditionNode(BaseModel):
    """Leaf representing a boolean condition."""
    type: Literal["condition"] = "condition"
    name: str  # e.g. "object_in_hand", "at_location"
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Composite nodes (sequence / fallback)
# ---------------------------------------------------------------------------


class CompositeNode(BaseModel):
    """Container for ordered children (sequence) or fallback)."""
    type: Literal["sequence", "selector", "parallel", "fallback"]  # "selector" is an alias for "fallback"
    id: str = ""
    node_type: str = ""  # "sequence", "selector", "parallel"
    name: str = ""
    children: List[AbstractPlanNode] = Field(default_factory=list)  # recursive children


# ---------------------------------------------------------------------------
# Decorator nodes (retry / timeout / inverter, etc.)
# ---------------------------------------------------------------------------


class DecoratorNode(BaseModel):
    """Wraps a single child with extra behaviour."""
    type: Literal["retry", "timeout", "inverter", "repeat"]
    decorator_type: str = ""  # "retry", "timeout", "inverter", "repeat"
    id: str = ""
    name: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    child: AbstractPlanNode
    parameters: dict[str, Any] = Field(default_factory=dict)  # alias for params


# ---------------------------------------------------------------------------
# Union of all node types (used for recursive typing)
# ---------------------------------------------------------------------------

AbstractPlanNode = Union[ActionNode, ConditionNode, CompositeNode, DecoratorNode]


# ---------------------------------------------------------------------------
# Top‑level plan
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """Full plan output by the LLM.

    Attributes
    ----------
    goal: str
        Original English instruction (kept for repro‑planning).
    root: AbstractPlanNode
        Root of the behavior‑tree description.
    """

    goal: str = ""
    root: AbstractPlanNode


# Export public symbols for convenient imports
__all__ = [
    "ActionNode",
    "ConditionNode",
    "CompositeNode",
    "DecoratorNode",
    "AbstractPlanNode",
    "Plan",
]
