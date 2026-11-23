"""Grasp candidate filtering and ranking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GraspCandidate:
    """Single grasp candidate in matrix form."""

    index: int
    tf_matrix: np.ndarray
    score: float
    width: float

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.tf_matrix[:3, 3], dtype=np.float32)

    @property
    def approach(self) -> np.ndarray:
        return _normalize(np.asarray(self.tf_matrix[:3, 0], dtype=np.float32))

    def to_dict(self, *, final_score: float | None = None, reasons: list[str] | None = None) -> dict[str, Any]:
        return {
            "index": self.index,
            "tf_matrix": self.tf_matrix.tolist(),
            "score": self.score,
            "final_score": final_score if final_score is not None else self.score,
            "width": self.width,
            "position": self.position.tolist(),
            "approach": self.approach.tolist(),
            "reasons": reasons or [],
        }


@dataclass
class GraspFilterConfig:
    """Config for robot/task-aware grasp filtering."""

    min_score: float | None = None
    min_width: float = 0.0
    max_width: float = 0.2
    enable_support_filter: bool = False
    support_z: float | None = None
    support_clearance: float = 0.0
    camera_position: list[float] | None = None
    approach_direction: list[float] | None = None
    approach_threshold_rad: float | None = None
    score_weight: float = 1.0
    camera_side_weight: float = 0.25
    approach_weight: float = 0.25


@dataclass
class GraspFilterRecord:
    """Filter/ranking result for one candidate."""

    candidate: GraspCandidate
    accepted: bool
    final_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = self.candidate.to_dict(final_score=self.final_score, reasons=self.reasons)
        data["accepted"] = self.accepted
        return data


@dataclass
class GraspFilterResult:
    """Full candidate filtering result."""

    accepted: list[GraspFilterRecord]
    rejected: list[GraspFilterRecord]

    @property
    def best(self) -> GraspFilterRecord | None:
        return self.accepted[0] if self.accepted else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [record.to_dict() for record in self.accepted],
            "rejected": [record.to_dict() for record in self.rejected],
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "best_index": self.best.candidate.index if self.best else None,
        }


def filter_and_rank_grasps(
    tf_matrices: list[np.ndarray],
    scores: np.ndarray,
    widths: np.ndarray,
    config: GraspFilterConfig,
) -> GraspFilterResult:
    """Filter and rank grasp candidates with explicit rejection reasons."""

    accepted: list[GraspFilterRecord] = []
    rejected: list[GraspFilterRecord] = []
    for index, tf_matrix in enumerate(tf_matrices):
        candidate = GraspCandidate(
            index=index,
            tf_matrix=np.asarray(tf_matrix, dtype=np.float32),
            score=float(scores[index]) if index < len(scores) else 0.0,
            width=float(widths[index]) if index < len(widths) else 0.0,
        )
        record = _score_candidate(candidate, config)
        if record.accepted:
            accepted.append(record)
        else:
            rejected.append(record)

    accepted.sort(key=lambda item: item.final_score, reverse=True)
    rejected.sort(key=lambda item: item.final_score, reverse=True)
    return GraspFilterResult(accepted=accepted, rejected=rejected)


def infer_support_z(object_points: Any, surround_points: Any = None) -> float | None:
    """Infer a conservative support height from observed object/surround points."""

    object_array = _as_points(object_points)
    if object_array.size == 0:
        return None
    return float(np.nanmin(object_array[:, 2]))


def _score_candidate(candidate: GraspCandidate, config: GraspFilterConfig) -> GraspFilterRecord:
    reasons: list[str] = []
    hard_reject = False
    if config.min_score is not None and candidate.score < float(config.min_score):
        reasons.append("score_below_min")
        hard_reject = True
    if candidate.width < float(config.min_width):
        reasons.append("width_below_min")
        hard_reject = True
    if candidate.width > float(config.max_width):
        reasons.append("width_above_max")
        hard_reject = True
    if config.enable_support_filter and config.support_z is not None:
        min_z = float(config.support_z) + float(config.support_clearance)
        if candidate.position[2] < min_z:
            reasons.append("below_support_clearance")
            hard_reject = True

    final_score = float(config.score_weight) * candidate.score
    camera_alignment = _camera_side_score(candidate, config.camera_position)
    if camera_alignment is not None:
        final_score += float(config.camera_side_weight) * camera_alignment
        if camera_alignment < -0.2:
            reasons.append("camera_opposite_side")

    approach_alignment = _approach_score(candidate, config.approach_direction)
    if approach_alignment is not None:
        final_score += float(config.approach_weight) * approach_alignment
        if config.approach_threshold_rad is not None:
            threshold_cos = float(np.cos(float(config.approach_threshold_rad)))
            if approach_alignment < threshold_cos:
                reasons.append("approach_direction_outside_threshold")
                hard_reject = True

    return GraspFilterRecord(
        candidate=candidate,
        accepted=not hard_reject,
        final_score=final_score,
        reasons=reasons,
    )


def _camera_side_score(candidate: GraspCandidate, camera_position: list[float] | None) -> float | None:
    if camera_position is None:
        return None
    camera = np.asarray(camera_position, dtype=np.float32)
    if camera.shape[0] < 3:
        return None
    from_camera_to_grasp = _normalize(candidate.position - camera[:3])
    if np.linalg.norm(from_camera_to_grasp) < 1e-6:
        return None
    return float(np.dot(candidate.approach, from_camera_to_grasp))


def _approach_score(candidate: GraspCandidate, approach_direction: list[float] | None) -> float | None:
    if approach_direction is None:
        return None
    direction = _normalize(np.asarray(approach_direction, dtype=np.float32))
    if np.linalg.norm(direction) < 1e-6:
        return None
    return float(np.dot(candidate.approach, direction))


def _as_points(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return array[:, :3]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros_like(vector, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)
