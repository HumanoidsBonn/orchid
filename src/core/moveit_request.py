"""Resolved MoveIt execution inputs. Backend defaults are captured during setup."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from src.core.manipulation_types import BasePose2D


@dataclass(frozen=True)
class MoveItPlanningSettings:
    group_name: str = "whole_body_weighted"
    ee_link: str = "hand_palm_link"
    planning_strategy: str = "auto"
    base_constraint_tolerance: dict[str, float] | None = None
    pipeline_id: str = "move_group"
    planner_id: str | None = None
    planning_attempts: int = 10
    allowed_planning_time: float = 5.0
    max_velocity_scaling_factor: float = 0.8
    max_acceleration_scaling_factor: float = 0.8
    goal_position_tolerance: float = 0.01
    goal_orientation_tolerance: float = 0.05
    joint_goal_tolerance: float = 0.01
    start_state_complete: bool = False
    workspace: Any = None
    collision_relaxation_links: tuple[str, ...] = ()
    collision_relaxation_objects: tuple[str, ...] = ()
    collision_relaxation_pairs: tuple[tuple[str, str], ...] = ()
    cartesian_max_step: float = 0.01
    cartesian_min_fraction: float = 0.99
    cartesian_jump_threshold: float = 0.0
    avoid_collisions: bool = True

    @classmethod
    def from_config(cls, config):
        from dataclasses import fields
        from copy import deepcopy
        values = {item.name: deepcopy(config[item.name]) for item in fields(cls) if item.name in config}
        values["base_constraint_tolerance"] = deepcopy(config.get("base_drift_tolerance"))
        values["collision_relaxation_links"] = tuple(config.get("collision_relaxation_links", config.get("moveit_collision_relaxation_links", ())))
        values["collision_relaxation_objects"] = tuple(config.get("collision_relaxation_objects", config.get("target_collision_objects", ())))
        values["avoid_collisions"] = bool(config.get("cartesian_avoid_collisions", True))
        return cls(**values)


@dataclass(frozen=True)
class MoveItPlanRequest:
    group_name: str
    goal_mode: str
    pose: Any = None
    joint_positions: dict[str, float] | None = None
    ee_link: str | None = None
    constrain_base: bool = False
    base_pose: BasePose2D | None = None
    base_constraint_tolerance: dict[str, float] | None = None
    collision_policy: str = "full"
    max_base_path_drift: dict[str, float] | None = None
    start_state_seed: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None
    settings: MoveItPlanningSettings | None = None
