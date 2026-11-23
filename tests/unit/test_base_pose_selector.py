"""Tests for heuristic manipulation base-pose selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.manipulation_types import FeasibilityResult
from src.plugin.base_pose_selector import (
    BasePose2D,
    BasePoseSelectorConfig,
    ManipulationBasePoseSelector,
)


def test_ring_candidates_face_object():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            angle_step_rad=math.pi / 2.0,
            refine_iterations=0,
        )
    )

    candidates = selector.generate_ring_candidates((0.0, 0.0), preferred_angles=[0.0])

    assert len(candidates) == 1
    assert candidates[0].x == pytest.approx(1.0)
    assert candidates[0].y == pytest.approx(0.0)
    assert abs(candidates[0].yaw) == pytest.approx(math.pi)


def test_selector_requires_ik_when_configured():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=True,
        ),
        ik_feasible=lambda pose: FeasibilityResult(pose.x < 0.0, "ik_failed"),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0, math.pi])

    assert result.selected is not None
    assert result.selected.pose.x == pytest.approx(-1.0)
    assert all(item.pose.x < 0.0 for item in result.accepted)
    assert any("ik_failed" in item.reasons for item in result.rejected)


def test_selector_records_soft_ik_failure_metadata_when_not_required():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=False,
        ),
        ik_feasible=lambda pose: FeasibilityResult(
            False,
            "curobo_ik_not_configured",
            metadata={"solver_trace": np.array([1, 2, 3])},
        ),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0])

    assert result.selected is not None
    assert result.selected.accepted is True
    assert result.selected.metadata["ik_feasible"] is False
    assert result.selected.metadata["ik_reason"] == "curobo_ik_not_configured"
    assert result.to_dict()["selected"]["metadata"]["ik_metadata"]["solver_trace"] == [1, 2, 3]


def test_selector_refines_around_best_candidate():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=0.5,
            max_radius=1.0,
            radius_step=0.5,
            preferred_radius=0.8,
            refine_top_k=1,
            refine_radius_step=0.1,
            refine_angle_step_rad=0.0,
            refine_iterations=2,
            require_ik=False,
        )
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0])

    assert result.selected is not None
    assert result.selected.pose.distance_to((0.0, 0.0)) == pytest.approx(0.8, abs=0.05)


def test_selector_uses_nav_and_soft_scores():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=False,
            visibility_weight=10.0,
        ),
        nav_feasible=lambda pose: FeasibilityResult(pose.y >= 0.0, "blocked"),
        visibility_score=lambda pose: 0.0 if pose.x < 0.0 else 1.0,
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0, math.pi / 2.0, math.pi, -math.pi / 2.0])

    assert result.selected is not None
    assert result.selected.pose.x < 0.0
    assert any("blocked" in item.reasons for item in result.rejected)


def test_selector_skips_ik_on_nav_rejected_by_default():
    calls = []
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=True,
        ),
        nav_feasible=lambda pose: FeasibilityResult(False, "nav_blocked"),
        ik_feasible=lambda pose: calls.append(pose) or FeasibilityResult(True),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0])

    assert result.selected is None
    assert calls == []
    assert result.rejected[0].metadata["ik_reason"] == "ik_skipped_nav_rejected"


def test_selector_can_evaluate_ik_on_nav_rejected_for_diagnostics():
    calls = []
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=True,
            evaluate_ik_on_nav_rejected=True,
        ),
        nav_feasible=lambda pose: FeasibilityResult(False, "nav_blocked"),
        ik_feasible=lambda pose: calls.append(pose) or FeasibilityResult(True),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0])

    assert result.selected is None
    assert len(calls) == 1
    assert result.rejected[0].metadata["ik_feasible"] is True


def test_selector_uses_batched_nav_feasibility_once():
    scalar_calls = []
    batch_calls = []

    def batch(candidates):
        batch_calls.append(list(candidates))
        return [FeasibilityResult(pose.x < 0.0, "nav_blocked") for pose in candidates]

    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=False,
        ),
        nav_feasible=lambda pose: scalar_calls.append(pose) or FeasibilityResult(True),
        nav_feasible_batch=batch,
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0, math.pi])

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 2
    assert scalar_calls == []
    assert result.selected is not None
    assert result.selected.pose.x < 0.0
    assert any(item.metadata["nav_reason"] == "nav_blocked" for item in result.rejected)


def test_nav_rejected_candidates_are_not_selected_even_with_diagnostic_ik():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=True,
            evaluate_ik_on_nav_rejected=True,
        ),
        nav_feasible_batch=lambda candidates: [FeasibilityResult(False, "nav_blocked") for _ in candidates],
        ik_feasible=lambda pose: FeasibilityResult(True),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0])

    assert result.selected is None
    assert result.rejected[0].metadata["ik_feasible"] is True
    assert result.rejected[0].metadata["nav_feasible"] is False


def test_hard_missing_nav_data_skips_ik_even_in_diagnostic_mode():
    calls = []
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=True,
            evaluate_ik_on_nav_rejected=True,
        ),
        nav_feasible_batch=lambda candidates: [
            FeasibilityResult(
                False,
                "missing_local_costmap,missing_map",
                metadata={"metadata": {"hard_nav_missing": True}},
            )
            for _ in candidates
        ],
        ik_feasible=lambda pose: calls.append(pose) or FeasibilityResult(True),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0])

    assert result.selected is None
    assert calls == []
    assert result.rejected[0].metadata["ik_reason"] == "ik_skipped_nav_rejected"


def test_selector_scores_distance_and_yaw_change_from_current_robot_pose():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=False,
            distance_weight=0.0,
            yaw_weight=0.0,
            robot_distance_weight=1.0,
            robot_yaw_change_weight=0.5,
        ),
        current_robot_pose=BasePose2D(1.0, 0.0, math.pi, "map"),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0, math.pi])

    assert result.selected is not None
    assert result.selected.pose.x == pytest.approx(1.0)
    assert result.selected.metadata["robot_distance"] == pytest.approx(0.0)
    assert result.selected.metadata["robot_yaw_change"] == pytest.approx(0.0)


def test_selector_penalizes_successful_ik_base_drift():
    selector = ManipulationBasePoseSelector(
        BasePoseSelectorConfig(
            min_radius=1.0,
            max_radius=1.0,
            radius_step=1.0,
            refine_iterations=0,
            require_ik=True,
            ik_bonus=0.0,
            base_drift_weight=10.0,
            base_yaw_drift_weight=2.0,
        ),
        ik_feasible=lambda pose: FeasibilityResult(
            True,
            metadata={
                "planner_metadata": {
                    "base_drift": {
                        "odom_x": 0.03 if pose.x < 0.0 else 0.20,
                        "odom_y": 0.04 if pose.x < 0.0 else 0.00,
                        "odom_t": 0.10 if pose.x < 0.0 else 0.20,
                    }
                }
            },
        ),
    )

    result = selector.select((0.0, 0.0), preferred_angles=[0.0, math.pi])

    assert result.selected is not None
    assert result.selected.pose.x < 0.0
    assert result.selected.metadata["base_drift_xy_m"] == pytest.approx(0.05)
    assert result.selected.metadata["base_drift_yaw_rad"] == pytest.approx(0.10)


def test_base_pose_serializes():
    assert BasePose2D(1.0, 2.0, 0.5, "odom").to_dict() == {
        "x": 1.0,
        "y": 2.0,
        "yaw": 0.5,
        "frame_id": "odom",
    }
