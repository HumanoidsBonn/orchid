"""Planner interfaces for manipulation planning backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from src.core.failure import FailureInfo


@dataclass
class PlannerResult:
    """Structured result returned by manipulation planners."""

    success: bool
    reason: str | None = None
    trajectory: Any | None = None
    joint_positions: dict[str, float] | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    failure: FailureInfo | None = None


    def __post_init__(self):
        if not self.success and self.failure is None:
            self.failure = FailureInfo()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "trajectory": self.trajectory,
            "joint_positions": self.joint_positions,
            "score": self.score,
            "metadata": self.metadata,
            "failure": {"code": self.failure.code.value, "retryable": self.failure.retryable, "backend_code": self.failure.backend_code} if self.failure else None,
        }


class BaseManipulationPlanner:
    """Base interface for manipulation planning backends.

    Planners produce feasibility checks or trajectories. Controllers remain
    responsible for execution.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    def check_ik(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        return PlannerResult(success=False, reason="ik_check_not_supported")

    def check_reachability(self, request: Any, *, context: Any | None = None) -> PlannerResult:
        return PlannerResult(success=False, reason="reachability_not_supported")

    def plan_pose(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        return PlannerResult(success=False, reason="pose_planning_not_supported")

    def plan_trajectory(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        return PlannerResult(success=False, reason="trajectory_planning_not_supported")
