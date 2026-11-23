"""Gripper-geometry collision checks using the real wizard-calibrated envelope.

Loads the HSR gripper's exterior bounding box from GraspGenX's own gripper
wizard config (``ext/gripper_descriptions/.../hsr_gripper/config.json`` under
the ``GGX`` submodule) - the same geometry, in the same pose-frame convention
(``approach=+Z``, ``closing=+X``), that GraspGenX itself was calibrated
against - and tests whether a candidate grasp pose's gripper envelope
actually clears real sensed points.

This exists because GraspGenX only ever sees the isolated object cloud (see
``estimate_pick_pose_ggx.py``), so nothing in its own candidate generation
checks a pose against the surrounding environment (shelf, table, background)
the object sits in - unlike GPD, which searches the combined object+surround
cloud and is collision-aware by construction. This module lets orchid do
that check after the fact, using the real gripper shape instead of a
fitted constant or a single-point distance proxy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_GRIPPER_CONFIG_ROOT = (
    Path(__file__).resolve().parents[2]
    / "submodules" / "GGX" / "GraspGenX" / "ext" / "gripper_descriptions"
    / "gripper_descriptions" / "assets" / "x_grippers"
)

# Matches SensorDataCenter.pcd_to_mesh's voxel_size=0.02 -> +-1cm inflation
# of any collision mesh built from the same raw points, so the envelope
# check stays consistent with what the real collision-mesh-based planner
# would already be inflating by.
_DEFAULT_MARGIN_M = 0.01


@lru_cache(maxsize=None)
def _load_gripper_config(gripper_name: str) -> dict[str, Any]:
    config_path = _GRIPPER_CONFIG_ROOT / gripper_name / "config.json"
    with open(config_path) as f:
        return json.load(f)


def get_gripper_bbox(gripper_name: str = "hsr_gripper") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bbox_min, bbox_max)``: the gripper's full exterior envelope
    in its own local pose frame, as calibrated by GraspGenX's gripper
    wizard. The pose origin sits at the gripper's leading/contact edge
    (bbox max Z ~= 0), with the body extending back along -Z as the
    gripper approaches in +Z.
    """
    config = _load_gripper_config(gripper_name)
    bbox_min, bbox_max = config["bbox"]
    return np.asarray(bbox_min, dtype=np.float64), np.asarray(bbox_max, dtype=np.float64)


def gripper_envelope_clears(
    tf: np.ndarray,
    points: np.ndarray | None,
    *,
    gripper_name: str = "hsr_gripper",
    margin_m: float = _DEFAULT_MARGIN_M,
) -> bool:
    """Check whether the gripper's real exterior envelope, placed at pose
    ``tf`` (a 4x4 transform in the same frame/convention as GraspGenX's
    returned poses), contains none of ``points`` (a world-frame Nx3+ array).

    Works by transforming ``points`` into the gripper's own local frame
    (inverse of the rigid transform ``tf``) and testing axis-aligned
    containment against the wizard-calibrated ``bbox`` - equivalent to, but
    cheaper than, transforming the box into world frame and testing there.
    """
    if points is None:
        return True
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    if pts.shape[0] == 0:
        return True

    tf = np.asarray(tf, dtype=np.float64)
    rotation = tf[:3, :3]
    translation = tf[:3, 3]
    # rotation is orthonormal, so rotation^-1 == rotation^T; for row-vector
    # points, (p - t) @ rotation gives the same per-row result as
    # rotation^T @ (p - t) in column-vector form.
    local_points = (pts - translation) @ rotation

    bbox_min, bbox_max = get_gripper_bbox(gripper_name)
    inside = np.all(
        (local_points >= (bbox_min - margin_m)) & (local_points <= (bbox_max + margin_m)),
        axis=1,
    )
    return not bool(np.any(inside))
