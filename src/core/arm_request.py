"""Explicit pose-operation inputs, separate from measured invocation state."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from src.core.manipulation_types import BasePose2D
if TYPE_CHECKING:
    from src.core.controllers import Pose3D


@dataclass(frozen=True)
class CartesianLine:
    axis: tuple[float, float, float] | None = None
    distance: float | None = None
    max_step: float | None = None
    min_fraction: float | None = None
    jump_threshold: float | None = None
    avoid_collisions: bool | None = None


@dataclass(frozen=True)
class MoveItPoseSettings:
    group_name: str | None = None
    planning_strategy: str | None = None
    goal_mode: str | None = None
    ee_link: str | None = None
    constrain_base: bool | None = None
    base_pose: BasePose2D | None = None
    base_constraint_tolerance: dict[str, float] | None = None
    max_base_path_drift: dict[str, float] | None = None
    collision_policy: Literal["full", "target_relaxed"] = "full"


@dataclass(frozen=True)
class ArmPoseRequest:
    pose: Pose3D
    segment: str = "pose"
    scope: Literal["whole_body", "arm_only"] = "whole_body"
    motion: Literal["pose", "cartesian_line"] = "pose"
    line: CartesianLine = field(default_factory=CartesianLine)
    looking_hand_constraint: bool = False
    moveit: MoveItPoseSettings = field(default_factory=MoveItPoseSettings)

    def __post_init__(self):
        if self.scope not in {"whole_body", "arm_only"}:
            raise ValueError(f"Invalid planning_scope '{self.scope}'")
        if self.motion not in {"pose", "cartesian_line"}:
            raise ValueError(f"Invalid planning_motion '{self.motion}'")
        if self.motion == "cartesian_line" and (self.line.axis is None or len(self.line.axis) != 3 or self.line.distance is None):
            raise ValueError("cartesian_line requires an axis of three values and a distance")
