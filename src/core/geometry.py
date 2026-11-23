"""Shared geometry helpers for transforms, poses, and point arrays."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_delta(a: float, b: float) -> float:
    """Return shortest signed planar yaw delta ``a - b``."""
    return wrap_angle(float(a) - float(b))


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for quaternion ``x,y,z,w``."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Return quaternion ``x,y,z,w`` for a 3x3 rotation matrix."""
    rot = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot[2, 1] - rot[1, 2]) * s
        qy = (rot[0, 2] - rot[2, 0]) * s
        qz = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]))
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]))
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]))
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Return quaternion ``x,y,z,w`` for fixed-axis roll/pitch/yaw."""
    cy = math.cos(float(yaw) * 0.5)
    sy = math.sin(float(yaw) * 0.5)
    cp = math.cos(float(pitch) * 0.5)
    sp = math.sin(float(pitch) * 0.5)
    cr = math.cos(float(roll) * 0.5)
    sr = math.sin(float(roll) * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for fixed-axis roll/pitch/yaw."""
    cr, sr = np.cos(float(roll)), np.sin(float(roll))
    cp, sp = np.cos(float(pitch)), np.sin(float(pitch))
    cy, sy = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def planar_pose_matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """Return a 4x4 transform matrix for a planar pose."""
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]
    matrix[:3, 3] = [float(x), float(y), 0.0]
    return matrix


def pose_matrix_relative_to_planar_base(target_matrix: np.ndarray, base_pose: Any) -> np.ndarray:
    """Express a world-frame pose matrix relative to a planar base pose."""
    bx = float(getattr(base_pose, "x", base_pose.get("x", 0.0) if isinstance(base_pose, dict) else 0.0))
    by = float(getattr(base_pose, "y", base_pose.get("y", 0.0) if isinstance(base_pose, dict) else 0.0))
    byaw = float(getattr(base_pose, "yaw", base_pose.get("yaw", 0.0) if isinstance(base_pose, dict) else 0.0))
    return np.linalg.inv(planar_pose_matrix(bx, by, byaw).astype(np.float64)) @ np.asarray(target_matrix, dtype=np.float64)


def point_relative_to_planar_base(target_position: Any, base_pose: Any) -> np.ndarray:
    """Express a world-frame XYZ point relative to a planar base pose."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(target_position, dtype=np.float64)[:3]
    return pose_matrix_relative_to_planar_base(matrix, base_pose)[:3, 3]


def transform_to_matrix(transform: Any) -> np.ndarray:
    """Return a 4x4 matrix from an object with ROS Transform-like fields."""
    translation = transform.translation
    rotation = transform.rotation
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quaternion_to_rotation_matrix(
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
        float(rotation.w),
    )
    mat[:3, 3] = [
        float(translation.x),
        float(translation.y),
        float(translation.z),
    ]
    return mat


def pose_dict_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    """Return a 4x4 matrix from ``SensorDataCenter.get_pose_at`` output."""
    position = pose.get("position") or {}
    orientation = pose.get("orientation") or {}
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quaternion_to_rotation_matrix(
        float(orientation.get("x", 0.0)),
        float(orientation.get("y", 0.0)),
        float(orientation.get("z", 0.0)),
        float(orientation.get("w", 1.0)),
    )
    mat[:3, 3] = [
        float(position.get("x", 0.0)),
        float(position.get("y", 0.0)),
        float(position.get("z", 0.0)),
    ]
    return mat


def matrix_to_xyz_rpy(tf_matrix: np.ndarray) -> list[float]:
    """Convert a 4x4 matrix to ``[x, y, z, roll, pitch, yaw]``."""
    tf_matrix = np.asarray(tf_matrix, dtype=np.float64)
    x = float(tf_matrix[0, 3])
    y = float(tf_matrix[1, 3])
    z = float(tf_matrix[2, 3])
    rot = tf_matrix[:3, :3]
    if rot[2, 0] < 1:
        if rot[2, 0] > -1:
            pitch = np.arcsin(-rot[2, 0])
            roll = np.arctan2(rot[2, 1], rot[2, 2])
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
        else:
            pitch = np.pi / 2.0
            roll = -np.arctan2(-rot[0, 1], rot[0, 2])
            yaw = 0.0
    else:
        pitch = -np.pi / 2.0
        roll = np.arctan2(-rot[0, 1], rot[0, 2])
        yaw = 0.0
    return [x, y, z, float(roll), float(pitch), float(yaw)]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from quaternion ``x,y,z,w``."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_points(points: Any, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an ``Nx3`` point array."""
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Invalid point array shape {array.shape}")
    xyz = np.ascontiguousarray(array[:, :3], dtype=np.float32)
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    transformed = (np.asarray(matrix, dtype=np.float32) @ np.hstack((xyz, ones)).T).T
    return np.ascontiguousarray(transformed[:, :3], dtype=np.float32)


def transform_pose_matrix(pose_matrix: np.ndarray, source_to_target: np.ndarray) -> np.ndarray:
    """Apply ``source_to_target`` to a homogeneous pose matrix."""
    return np.asarray(source_to_target, dtype=np.float32) @ np.asarray(pose_matrix, dtype=np.float32)


def transform_direction(vector: Any, source_to_target: np.ndarray) -> np.ndarray:
    """Rotate a 3D direction by the rotational part of a transform matrix."""
    direction = np.asarray(vector, dtype=np.float32)
    if direction.shape[0] < 3:
        raise ValueError(f"Invalid direction shape {direction.shape}")
    transformed = np.asarray(source_to_target, dtype=np.float32)[:3, :3] @ direction[:3]
    norm = float(np.linalg.norm(transformed))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return np.ascontiguousarray(transformed / norm, dtype=np.float32)
