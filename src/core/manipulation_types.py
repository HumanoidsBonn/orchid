"""Shared data contracts for mobile manipulation planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import math

from src.core.serialization import to_json_safe


CollisionPolicy = Literal["full", "target_relaxed"]
ReachabilityMode = Literal["pregrasp_region", "exact_grasp", "position"]


@dataclass(frozen=True)
class BasePose2D:
    """2D mobile-base pose in a global frame."""

    x: float
    y: float
    yaw: float
    frame_id: str = "map"

    def distance_to(self, point_xy: tuple[float, float]) -> float:
        return float(math.hypot(self.x - point_xy[0], self.y - point_xy[1]))

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw, "frame_id": self.frame_id}

    @classmethod
    def from_mapping(cls, value: Any, *, default_frame_id: str = "map") -> "BasePose2D":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("BasePose2D requires a mapping or BasePose2D")
        return cls(
            x=float(value["x"]),
            y=float(value["y"]),
            yaw=float(value["yaw"]),
            frame_id=str(value.get("frame_id") or default_frame_id),
        )


@dataclass(frozen=True)
class ReachTarget:
    """End-effector reach target used for candidate-base feasibility checks."""

    x: float
    y: float
    z: float
    frame_id: str
    source: str | None = None

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "frame_id": self.frame_id,
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, value: Any, *, default_frame_id: str = "map") -> "ReachTarget":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("ReachTarget requires a mapping or ReachTarget")
        return cls(
            x=float(value["x"]),
            y=float(value["y"]),
            z=float(value["z"]),
            frame_id=str(value.get("frame_id") or value.get("target_position_frame") or default_frame_id),
            source=value.get("source"),
        )


@dataclass(frozen=True)
class BaseDriftTolerance:
    """Allowed whole-body base drift around a candidate base pose."""

    x: float = 0.30
    y: float = 0.30
    yaw: float = 0.50

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


@dataclass(frozen=True)
class MoveItReachabilityRequest:
    """Typed request for MoveIt-backed candidate-base reachability."""

    reach_target: ReachTarget
    candidate_base: BasePose2D
    joint_state_seed: Any | None = None
    group_name: str = "whole_body_weighted"
    ee_link: str | None = None
    mode: ReachabilityMode = "pregrasp_region"
    collision_policy: CollisionPolicy = "full"
    validate_plan: bool = True
    avoid_collisions_for_ik: bool = True
    avoid_collisions_for_plan: bool = True
    position_tolerance_m: float | None = None
    orientation_tolerance_rad: float | None = None
    base_drift_tolerance: BaseDriftTolerance | dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def base_drift_tolerance_dict(self) -> dict[str, float]:
        if isinstance(self.base_drift_tolerance, BaseDriftTolerance):
            return self.base_drift_tolerance.to_dict()
        if isinstance(self.base_drift_tolerance, dict):
            return {
                "x": float(self.base_drift_tolerance.get("x", 0.30)),
                "y": float(self.base_drift_tolerance.get("y", 0.30)),
                "yaw": float(self.base_drift_tolerance.get("yaw", 0.50)),
            }
        return BaseDriftTolerance().to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reach_target": self.reach_target.to_dict(),
            "candidate_base": self.candidate_base.to_dict(),
            "group_name": self.group_name,
            "ee_link": self.ee_link,
            "mode": self.mode,
            "collision_policy": self.collision_policy,
            "validate_plan": self.validate_plan,
            "avoid_collisions_for_ik": self.avoid_collisions_for_ik,
            "avoid_collisions_for_plan": self.avoid_collisions_for_plan,
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "base_drift_tolerance": self.base_drift_tolerance_dict(),
            "metadata": to_json_safe(self.metadata),
        }


@dataclass(frozen=True)
class FeasibilityResult:
    """Common typed feasibility result used by selector callbacks."""

    success: bool
    reason: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "score": self.score,
            "metadata": to_json_safe(self.metadata),
        }

    @classmethod
    def from_legacy(cls, result: Any) -> "FeasibilityResult":
        if isinstance(result, cls):
            return result
        if isinstance(result, dict):
            metadata = {
                key: value
                for key, value in result.items()
                if key not in {"success", "feasible", "reason", "score"}
            }
            score = result.get("score")
            return cls(
                success=bool(result.get("success", result.get("feasible", False))),
                reason=str(result["reason"]) if result.get("reason") else None,
                score=float(score) if score is not None else None,
                metadata=metadata,
            )
        if isinstance(result, tuple):
            score = result[2] if len(result) > 2 else None
            return cls(
                success=bool(result[0]),
                reason=str(result[1]) if len(result) > 1 and result[1] else None,
                score=float(score) if score is not None else None,
            )
        return cls(success=bool(result))
