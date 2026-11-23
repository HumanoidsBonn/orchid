"""Heuristic manipulation base-pose selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.core.geometry import wrap_angle
from src.core.manipulation_types import BasePose2D, FeasibilityResult
from src.core.serialization import to_json_safe
from src.core.logger import log


@dataclass
class BasePoseSelectorConfig:
    """Config for coarse-to-fine manipulation base-pose search."""

    min_radius: float = 0.45
    max_radius: float = 1.05
    radius_step: float = 0.15
    angle_step_rad: float = math.radians(30.0)
    preferred_radius: float = 0.4 # 0.75
    yaw_offset_rad: float = 0.0
    refine_top_k: int = 5
    refine_radius_step: float = 0.5 # 0.05
    refine_angle_step_rad: float = math.radians(20.0) # 10.0
    refine_iterations: int = 4 # 1
    distance_weight: float = 1.0
    yaw_weight: float = 0.25
    clearance_weight: float = 1.0
    visibility_weight: float = 1.0
    robot_distance_weight: float = 0.0
    robot_yaw_change_weight: float = 0.0
    base_drift_weight: float = 1.0
    base_yaw_drift_weight: float = 0.25
    ik_bonus: float = 2.0
    require_ik: bool = True
    evaluate_ik_on_nav_rejected: bool = False


@dataclass
class BasePoseScore:
    """Scored manipulation base pose with reasons."""

    pose: BasePose2D
    score: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pose": self.pose.to_dict(),
            "score": self.score,
            "accepted": self.accepted,
            "reasons": self.reasons,
            "metadata": to_json_safe(self.metadata),
        }


@dataclass
class BasePoseSelectionResult:
    """Result of manipulation base-pose search."""

    selected: BasePoseScore | None
    accepted: list[BasePoseScore]
    rejected: list[BasePoseScore]

    def to_dict(self) -> dict:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "accepted": [item.to_dict() for item in self.accepted],
            "rejected": [item.to_dict() for item in self.rejected],
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
        }


FeasibilityFn = Callable[[BasePose2D], FeasibilityResult]
BatchFeasibilityFn = Callable[[list[BasePose2D]], list[FeasibilityResult]]
ScoreFn = Callable[[BasePose2D], float | tuple[float, str] | dict]


class ManipulationBasePoseSelector:
    """Coarse-to-fine base-pose selector for mobile manipulation."""

    def __init__(
        self,
        config: BasePoseSelectorConfig | None = None,
        *,
        nav_feasible: FeasibilityFn | None = None,
        nav_feasible_batch: BatchFeasibilityFn | None = None,
        ik_feasible: FeasibilityFn | None = None,
        clearance_score: ScoreFn | None = None,
        visibility_score: ScoreFn | None = None,
        current_robot_pose: BasePose2D | None = None,
    ) -> None:
        self.config = config or BasePoseSelectorConfig()
        self.nav_feasible = nav_feasible
        self.nav_feasible_batch = nav_feasible_batch
        self.ik_feasible = ik_feasible
        self.clearance_score = clearance_score
        self.visibility_score = visibility_score
        self.current_robot_pose = current_robot_pose

    def select(
        self,
        object_xy: tuple[float, float],
        *,
        frame_id: str = "map",
        preferred_angles: list[float] | None = None,
    ) -> BasePoseSelectionResult:
        coarse = self.generate_ring_candidates(
            object_xy,
            frame_id=frame_id,
            preferred_angles=preferred_angles,
        )
        scored = self.score_candidates(coarse, object_xy)
        accepted = [item for item in scored if item.accepted]
        accepted.sort(key=lambda item: item.score)

        refined: list[BasePoseScore] = []
        seeds = accepted[: max(0, int(self.config.refine_top_k))]
        for seed in seeds:
            refined.extend(self._refine(seed.pose, object_xy))

        all_scored = scored + refined
        # log.error(all_scored)

        accepted = [item for item in all_scored if item.accepted]
        rejected = [item for item in all_scored if not item.accepted]
        accepted.sort(key=lambda item: item.score)
        rejected.sort(key=lambda item: item.score)
        return BasePoseSelectionResult(
            selected=accepted[0] if accepted else None,
            accepted=accepted,
            rejected=rejected,
        )

    def generate_ring_candidates(
        self,
        object_xy: tuple[float, float],
        *,
        frame_id: str = "map",
        preferred_angles: list[float] | None = None,
    ) -> list[BasePose2D]:
        radii = np.arange(
            float(self.config.min_radius),
            float(self.config.max_radius) + 1e-6,
            float(self.config.radius_step),
        )
        angles = preferred_angles or list(
            np.arange(-math.pi, math.pi, float(self.config.angle_step_rad))
        )
        candidates: list[BasePose2D] = []
        for radius in radii:
            for angle in angles:
                x = object_xy[0] + float(radius) * math.cos(float(angle))
                y = object_xy[1] + float(radius) * math.sin(float(angle))
                yaw = wrap_angle(math.atan2(object_xy[1] - y, object_xy[0] - x) + self.config.yaw_offset_rad)
                candidates.append(BasePose2D(x=x, y=y, yaw=yaw, frame_id=frame_id))
        return candidates

    def score_candidates(self, candidates: list[BasePose2D], object_xy: tuple[float, float]) -> list[BasePoseScore]:
        nav_results = self._nav_results_for(candidates)
        return [
            self.score_candidate(candidate, object_xy, nav_result=nav_results[index])
            for index, candidate in enumerate(candidates)
        ]

    def score_candidate(
        self,
        candidate: BasePose2D,
        object_xy: tuple[float, float],
        *,
        nav_result: FeasibilityResult | None = None,
    ) -> BasePoseScore:
        reasons: list[str] = []
        metadata: dict = {}
        accepted = True
        distance = candidate.distance_to(object_xy)
        score = float(self.config.distance_weight) * abs(distance - float(self.config.preferred_radius))

        desired_yaw = math.atan2(object_xy[1] - candidate.y, object_xy[0] - candidate.x) + self.config.yaw_offset_rad
        yaw_error = abs(wrap_angle(candidate.yaw - desired_yaw))
        score += float(self.config.yaw_weight) * yaw_error
        metadata["distance"] = distance
        metadata["yaw_error"] = yaw_error

        if self.current_robot_pose is not None:
            robot_distance = candidate.distance_to((self.current_robot_pose.x, self.current_robot_pose.y))
            robot_yaw_change = abs(wrap_angle(candidate.yaw - self.current_robot_pose.yaw))
            score += float(self.config.robot_distance_weight) * robot_distance
            score += float(self.config.robot_yaw_change_weight) * robot_yaw_change
            metadata["robot_distance"] = robot_distance
            metadata["robot_yaw_change"] = robot_yaw_change

        if nav_result is None:
            nav = _call_feasible_details(self.nav_feasible, candidate)
        else:
            nav = nav_result
        nav_ok = bool(nav.success)
        nav_reason = nav.reason
        nav_metadata = dict(nav.metadata)
        metadata["nav_feasible"] = nav_ok
        if nav_reason:
            metadata["nav_reason"] = nav_reason
        if nav_metadata:
            metadata["nav_metadata"] = nav_metadata
        if not nav_ok:
            accepted = False
            reasons.append(nav_reason or "nav_infeasible")

        hard_nav_missing = bool(
            nav_metadata.get("hard_nav_missing")
            or (isinstance(nav_metadata.get("metadata"), dict) and nav_metadata["metadata"].get("hard_nav_missing"))
        )
        should_check_ik = nav_ok or (bool(self.config.evaluate_ik_on_nav_rejected) and not hard_nav_missing)
        ik = (
            _call_feasible_details(self.ik_feasible, candidate)
            if should_check_ik
            else FeasibilityResult(False, "ik_skipped_nav_rejected")
        )
        ik_ok = bool(ik.success)
        ik_reason = ik.reason
        ik_metadata = dict(ik.metadata)
        metadata["ik_feasible"] = ik_ok
        if ik_reason:
            metadata["ik_reason"] = ik_reason
        if ik_metadata:
            metadata["ik_metadata"] = ik_metadata
        if ik_ok:
            score -= float(self.config.ik_bonus)
            drift_xy_m, drift_yaw_rad = _base_drift_from_feasibility_metadata(ik_metadata)
            if drift_xy_m is not None:
                score += float(self.config.base_drift_weight) * drift_xy_m
                metadata["base_drift_xy_m"] = drift_xy_m
            if drift_yaw_rad is not None:
                score += float(self.config.base_yaw_drift_weight) * drift_yaw_rad
                metadata["base_drift_yaw_rad"] = drift_yaw_rad
        elif self.config.require_ik:
            accepted = False
            reasons.append(ik_reason or "ik_infeasible")

        clearance, clearance_reason = _call_score(self.clearance_score, candidate)
        if clearance is not None:
            score += float(self.config.clearance_weight) * clearance
            metadata["clearance_score"] = clearance
            if clearance_reason:
                reasons.append(clearance_reason)

        visibility, visibility_reason = _call_score(self.visibility_score, candidate)
        if visibility is not None:
            score += float(self.config.visibility_weight) * visibility
            metadata["visibility_score"] = visibility
            if visibility_reason:
                reasons.append(visibility_reason)

        return BasePoseScore(
            pose=candidate,
            score=float(score),
            accepted=accepted,
            reasons=reasons,
            metadata=metadata,
        )

    def _nav_results_for(self, candidates: list[BasePose2D]) -> list[FeasibilityResult | None]:
        if not candidates:
            return []
        if self.nav_feasible_batch is None:
            return [None] * len(candidates)
        try:
            results = list(self.nav_feasible_batch(candidates))
        except Exception as exc:  # noqa: BLE001
            log.exception("Batched nav feasibility check failed: %s", exc)
            return [FeasibilityResult(False, f"nav_batch_exception: {exc}") for _ in candidates]
        if len(results) != len(candidates):
            return [
                FeasibilityResult(
                    False,
                    "nav_batch_result_count_mismatch",
                    metadata={"expected": len(candidates), "actual": len(results)},
                )
                for _ in candidates
            ]
        return results

    def _refine(self, seed: BasePose2D, object_xy: tuple[float, float]) -> list[BasePoseScore]:
        current = [seed]
        scored: list[BasePoseScore] = []
        for iteration in range(max(0, int(self.config.refine_iterations))):
            next_candidates: list[BasePose2D] = []
            radius_delta = float(self.config.refine_radius_step) / float(iteration + 1)
            angle_delta = float(self.config.refine_angle_step_rad) / float(iteration + 1)
            for pose in current:
                radius = pose.distance_to(object_xy)
                angle = math.atan2(pose.y - object_xy[1], pose.x - object_xy[0])
                for dr in (-radius_delta, 0.0, radius_delta):
                    for da in (-angle_delta, 0.0, angle_delta):
                        new_radius = min(
                            float(self.config.max_radius),
                            max(float(self.config.min_radius), radius + dr),
                        )
                        new_angle = angle + da
                        x = object_xy[0] + new_radius * math.cos(new_angle)
                        y = object_xy[1] + new_radius * math.sin(new_angle)
                        yaw = wrap_angle(math.atan2(object_xy[1] - y, object_xy[0] - x) + self.config.yaw_offset_rad)
                        next_candidates.append(BasePose2D(x=x, y=y, yaw=yaw, frame_id=pose.frame_id))
            round_scored = self.score_candidates(next_candidates, object_xy)
            scored.extend(round_scored)
            accepted = [item.pose for item in round_scored if item.accepted]
            accepted_scores = [item for item in round_scored if item.accepted]
            accepted_scores.sort(key=lambda item: item.score)
            current = [item.pose for item in accepted_scores[: max(1, int(self.config.refine_top_k))]]
            if not accepted:
                break
        return scored


def _call_feasible_details(fn: FeasibilityFn | None, pose: BasePose2D) -> FeasibilityResult:
    if fn is None:
        return FeasibilityResult(True)
    result = fn(pose)
    if not isinstance(result, FeasibilityResult):
        raise TypeError(f"feasibility callback returned {type(result).__name__}, expected FeasibilityResult")
    return result


def _base_drift_from_feasibility_metadata(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    planner_metadata = metadata.get("planner_metadata") if isinstance(metadata.get("planner_metadata"), dict) else metadata
    xy = planner_metadata.get("base_drift_xy_m")
    yaw = planner_metadata.get("base_drift_yaw_rad")
    if xy is None and isinstance(planner_metadata.get("base_drift"), dict):
        drift = planner_metadata["base_drift"]
        dx = drift.get("odom_x") if drift.get("odom_x") is not None else drift.get("x")
        dy = drift.get("odom_y") if drift.get("odom_y") is not None else drift.get("y")
        if dx is not None and dy is not None:
            xy = math.hypot(float(dx), float(dy))
        yaw = yaw if yaw is not None else drift.get("odom_t", drift.get("yaw"))
    return (
        float(xy) if xy is not None else None,
        abs(float(yaw)) if yaw is not None else None,
    )


def _call_score(fn: ScoreFn | None, pose: BasePose2D) -> tuple[float | None, str | None]:
    if fn is None:
        return None, None
    result = fn(pose)
    if isinstance(result, dict):
        value = result.get("score")
        return (float(value) if value is not None else None), result.get("reason")
    if isinstance(result, tuple):
        return float(result[0]), str(result[1]) if len(result) > 1 and result[1] else None
    return float(result), None
