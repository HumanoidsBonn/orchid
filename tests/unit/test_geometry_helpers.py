"""Tests for shared geometry helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.geometry import (
    point_relative_to_planar_base,
    pose_matrix_relative_to_planar_base,
    rpy_to_quaternion,
    rpy_to_rotation_matrix,
    wrap_angle,
    yaw_delta,
)
from src.core.manipulation_types import BasePose2D


def test_wrap_angle_and_yaw_delta_use_shortest_path():
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    assert yaw_delta(-math.pi + 0.1, math.pi - 0.1) == pytest.approx(0.2)


def test_rpy_quaternion_and_rotation_matrix_identity():
    assert rpy_to_quaternion(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert rpy_to_rotation_matrix(0.0, 0.0, 0.0) == pytest.approx(np.eye(3, dtype=np.float32))


def test_point_and_pose_relative_to_planar_base():
    base = BasePose2D(x=1.0, y=2.0, yaw=math.pi / 2.0, frame_id="odom")
    point = point_relative_to_planar_base((1.0, 3.0, 0.5), base)

    assert point == pytest.approx((1.0, 0.0, 0.5), abs=1e-6)

    target = np.eye(4, dtype=np.float32)
    target[:3, 3] = [1.0, 3.0, 0.5]
    relative = pose_matrix_relative_to_planar_base(target, base)

    assert relative[:3, 3] == pytest.approx((1.0, 0.0, 0.5), abs=1e-6)
