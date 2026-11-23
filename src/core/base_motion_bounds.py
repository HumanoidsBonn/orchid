"""Validate the base coordinates that will actually be sent to controllers."""
from __future__ import annotations

import math
from typing import Any


def validate_base_motion(trajectory: Any, reference: dict, *, joint_names,
                         endpoint_tolerance=None, path_limits=None,
                         path_tolerance=None, require_base=True) -> dict:
    """Return diagnostics, including ``violation`` on any uncheckable/invalid path.

    Yaw coordinates are commanded values, not wrapped headings. In particular,
    a waypoint commanding an extra full turn must not pass a small yaw bound.
    """
    def invalid(reason, **details):
        return {"base_validation": "rejected", "violation": {"reason": reason, **details}}

    names = list(getattr(trajectory, "joint_names", []) or [])
    present = [name in names for name in joint_names]
    if not any(present) and not require_base:
        return {"base_validation": "stationary"}
    if not all(present):
        return invalid("base_trajectory_joints_missing", joint_names=names)
    points = list(getattr(trajectory, "points", []) or [])
    if not points:
        return invalid("base_trajectory_points_missing")
    try:
        anchor = [float(reference[key]) for key in ("x", "y", "yaw")]
        indices = [names.index(name) for name in joint_names]
        bounds = {}
        for kind, limits in (("endpoint", endpoint_tolerance), ("path", path_limits), ("window", path_tolerance)):
            bounds[kind] = {str(key): float(value) for key, value in (limits or {}).items()}
            if any(not math.isfinite(value) or value < 0 for value in bounds[kind].values()):
                return invalid("base_limits_invalid", limits=bounds[kind])
        if not all(math.isfinite(value) for value in anchor):
            return invalid("base_reference_invalid")
        rows = []
        for index, point in enumerate(points):
            values = list(getattr(point, "positions", []) or [])
            if len(values) != len(names):
                return invalid("base_trajectory_positions_invalid", waypoint=index)
            row = [float(values[i]) for i in indices]
            if not all(math.isfinite(value) for value in row):
                return invalid("base_trajectory_nonfinite", waypoint=index)
            rows.append(row)
        max_xy = max_yaw = 0.0
        for index, row in enumerate(rows):
            delta = [abs(value - origin) for value, origin in zip(row, anchor)]
            xy = math.hypot(delta[0], delta[1])
            max_xy, max_yaw = max(max_xy, xy), max(max_yaw, delta[2])
            checks = [("path", "xy", xy, bounds["path"].get("xy", bounds["path"].get("xy_m"))),
                      ("path", "yaw", delta[2], bounds["path"].get("yaw", bounds["path"].get("yaw_rad")))]
            for axis, value, name in zip(("x", "y", "yaw"), delta, joint_names):
                checks.append(("window", axis, value, bounds["window"].get(axis, bounds["window"].get(name))))
                if index == len(rows) - 1:
                    checks.append(("endpoint", axis, value, bounds["endpoint"].get(axis, bounds["endpoint"].get(name))))
            for kind, axis, value, limit in checks:
                if limit is not None and value > limit + 1e-9:
                    return invalid("base_motion_limit_exceeded", kind=kind, axis=axis,
                                   waypoint=index, value=value, limit=limit, reference=dict(reference))
        return {"base_validation": "passed", "base_waypoint_count": len(rows),
                "max_base_path_drift_xy_m": max_xy, "max_base_path_drift_yaw_rad": max_yaw}
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        return invalid("base_trajectory_uncheckable", error=str(exc))
