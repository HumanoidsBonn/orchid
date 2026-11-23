"""Tests for shared manipulation planning contracts."""

from __future__ import annotations

import pytest

from src.core.manipulation_types import BasePose2D, FeasibilityResult, ReachTarget


def test_base_pose_2d_from_mapping_requires_pose_fields():
    pose = BasePose2D.from_mapping({"x": 1, "y": 2, "yaw": 0.5, "frame_id": "odom"})

    assert pose == BasePose2D(1.0, 2.0, 0.5, "odom")
    with pytest.raises(KeyError):
        BasePose2D.from_mapping({"x": 1, "y": 2})


def test_reach_target_from_mapping_accepts_transitional_frame_key():
    target = ReachTarget.from_mapping({"x": 1, "y": 2, "z": 3, "target_position_frame": "odom"})

    assert target == ReachTarget(1.0, 2.0, 3.0, "odom", None)


def test_feasibility_result_from_legacy_shapes():
    assert FeasibilityResult.from_legacy(True).success is True
    assert FeasibilityResult.from_legacy((False, "blocked")).reason == "blocked"

    result = FeasibilityResult.from_legacy({"success": False, "reason": "ik_failed", "trace": [1, 2]})

    assert result.success is False
    assert result.reason == "ik_failed"
    assert result.metadata == {"trace": [1, 2]}
