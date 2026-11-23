"""Tests for grasp_collision's gripper-envelope collision checks."""

from __future__ import annotations

import numpy as np

from src.core.grasp_collision import get_gripper_bbox, gripper_envelope_clears


def test_get_gripper_bbox_loads_real_wizard_calibration():
    bbox_min, bbox_max = get_gripper_bbox()
    assert bbox_min.shape == (3,)
    assert bbox_max.shape == (3,)
    assert np.all(bbox_min < bbox_max)


def test_no_points_always_clears():
    tf = np.eye(4)
    assert gripper_envelope_clears(tf, None) is True
    assert gripper_envelope_clears(tf, np.empty((0, 3))) is True


def test_identity_pose_clears_far_away_points():
    tf = np.eye(4)
    far_points = np.array([[5.0, 5.0, 5.0], [-3.0, 2.0, 1.0]])
    assert gripper_envelope_clears(tf, far_points) is True


def test_identity_pose_does_not_clear_point_inside_bbox():
    tf = np.eye(4)
    bbox_min, bbox_max = get_gripper_bbox()
    center = (bbox_min + bbox_max) / 2.0
    inside_point = np.array([center])
    assert gripper_envelope_clears(tf, inside_point) is False


def test_shelf_below_gripper_at_safe_height_clears():
    """A gripper positioned well above a flat shelf plane should clear it."""
    tf = np.eye(4)
    tf[:3, 3] = [0.0, 0.0, 0.5]  # gripper origin 0.5m above the shelf plane
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 20))
    shelf_points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    assert gripper_envelope_clears(tf, shelf_points) is True


def test_shelf_intersecting_gripper_envelope_does_not_clear():
    """Descending the gripper down to the shelf plane should collide."""
    tf = np.eye(4)
    tf[:3, 3] = [0.0, 0.0, 0.01]  # gripper origin just above the shelf plane
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.5, 0.5, 20))
    shelf_points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    assert gripper_envelope_clears(tf, shelf_points) is False


def test_translated_pose_moves_the_collision_region():
    bbox_min, bbox_max = get_gripper_bbox()
    center_local = (bbox_min + bbox_max) / 2.0

    tf = np.eye(4)
    tf[:3, 3] = [2.0, 0.0, 0.0]
    world_point = np.array([center_local + tf[:3, 3]])
    assert gripper_envelope_clears(tf, world_point) is False
    # Same world point, checked against the identity-pose gripper far away.
    assert gripper_envelope_clears(np.eye(4), world_point) is True


def test_rotated_pose_reorients_the_collision_region():
    """A 90-degree rotation about Z should reorient the local bbox in world
    space, not just translate it - a point that collides with the rotated
    pose (placed via R @ corner) should not collide at the identity pose,
    since the bbox's X/Y extents are asymmetric.
    """
    _, bbox_max = get_gripper_bbox()
    theta = np.pi / 2
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated_tf = np.eye(4)
    rotated_tf[:3, :3] = rotation

    world_point = np.array([rotation @ bbox_max])
    assert gripper_envelope_clears(rotated_tf, world_point) is False
    assert gripper_envelope_clears(np.eye(4), world_point) is True
