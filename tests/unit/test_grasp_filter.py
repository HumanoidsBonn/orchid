"""Tests for grasp filtering and ranking."""

from __future__ import annotations

import numpy as np
import pytest

from src.plugin.grasp_filter import GraspFilterConfig, filter_and_rank_grasps, infer_support_z


def _tf(approach, position):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = np.asarray(approach, dtype=np.float32)
    matrix[:3, 3] = np.asarray(position, dtype=np.float32)
    return matrix


def test_filter_rejects_width_and_score_with_support_filter_disabled_by_default():
    result = filter_and_rank_grasps(
        [
            _tf([1, 0, 0], [0, 0, 0.2]),
            _tf([1, 0, 0], [0, 0, 0.0]),
            _tf([1, 0, 0], [0, 0, 0.2]),
        ],
        np.array([0.9, 0.9, 0.1], dtype=np.float32),
        np.array([0.05, 0.3, 0.05], dtype=np.float32),
        GraspFilterConfig(
            min_score=0.5,
            min_width=0.02,
            max_width=0.1,
            support_z=0.05,
            support_clearance=0.02,
        ),
    )

    assert [record.candidate.index for record in result.accepted] == [0]
    rejected_reasons = {
        reason
        for record in result.rejected
        for reason in record.reasons
    }
    assert "width_above_max" in rejected_reasons
    assert "score_below_min" in rejected_reasons
    assert "below_support_clearance" not in rejected_reasons


def test_filter_rejects_support_clearance_when_explicitly_enabled():
    result = filter_and_rank_grasps(
        [
            _tf([1, 0, 0], [0, 0, 0.2]),
            _tf([1, 0, 0], [0, 0, 0.0]),
        ],
        np.array([0.9, 0.9], dtype=np.float32),
        np.array([0.05, 0.05], dtype=np.float32),
        GraspFilterConfig(
            min_score=0.5,
            min_width=0.02,
            max_width=0.1,
            enable_support_filter=True,
            support_z=0.05,
            support_clearance=0.02,
        ),
    )

    assert [record.candidate.index for record in result.accepted] == [0]
    assert result.rejected[0].candidate.index == 1
    assert "below_support_clearance" in result.rejected[0].reasons


def test_filter_ranks_with_camera_side_and_approach_bias():
    result = filter_and_rank_grasps(
        [
            _tf([1, 0, 0], [1, 0, 0]),
            _tf([-1, 0, 0], [1, 0, 0]),
        ],
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.05, 0.05], dtype=np.float32),
        GraspFilterConfig(
            camera_position=[0, 0, 0],
            approach_direction=[1, 0, 0],
            camera_side_weight=0.5,
            approach_weight=0.5,
        ),
    )

    assert [record.candidate.index for record in result.accepted] == [0, 1]
    assert result.accepted[0].final_score > result.accepted[1].final_score
    assert "camera_opposite_side" in result.accepted[1].reasons


def test_infer_support_z_uses_object_bottom():
    assert infer_support_z(np.array([[0, 0, 0.3], [0, 0, 0.1]], dtype=np.float32)) == pytest.approx(0.1)
