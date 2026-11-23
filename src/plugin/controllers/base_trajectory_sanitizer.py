"""Helpers for controller-ready HSR omni-base virtual joint trajectories."""

from __future__ import annotations

import math
from typing import Any

import src.core.logger as log


BASE_JOINT_NAMES = ("odom_x", "odom_y", "odom_t")


def is_base_trajectory_config(controller_config: dict[str, Any], trajectory: Any) -> bool:
    if str(controller_config.get("group_type") or "") == "base_trajectory":
        return True
    joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
    return all(name in joint_names for name in BASE_JOINT_NAMES)


def sanitize_base_trajectory(
    trajectory: Any,
    *,
    current_positions: dict[str, float],
    controller_config: dict[str, Any],
    log_prefix: str = "Base",
) -> Any:
    from trajectory_msgs.msg import JointTrajectoryPoint

    joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
    points = list(getattr(trajectory, "points", []) or [])
    if joint_names != list(BASE_JOINT_NAMES) or not points:
        return trajectory

    min_dt = _positive_scale(controller_config.get("base_trajectory_min_dt_s"), default=0.12)
    max_vx = _positive_scale(controller_config.get("base_trajectory_max_vx_mps"), default=0.20)
    max_vy = _positive_scale(controller_config.get("base_trajectory_max_vy_mps"), default=0.20)
    max_wz = _positive_scale(controller_config.get("base_trajectory_max_wz_radps"), default=0.35)
    extra_time_scale = _positive_scale(controller_config.get("base_trajectory_time_scale"), default=1.0)

    first_positions = list(getattr(points[0], "positions", []) or [])
    current = [
        float(current_positions.get("odom_x", first_positions[0] if len(first_positions) > 0 else 0.0)),
        float(current_positions.get("odom_y", first_positions[1] if len(first_positions) > 1 else 0.0)),
        float(current_positions.get("odom_t", first_positions[2] if len(first_positions) > 2 else 0.0)),
    ]

    raw_summary = base_trajectory_diagnostics(trajectory, limits=(max_vx, max_vy, max_wz))
    sanitized = type(trajectory)()
    sanitized.header = getattr(trajectory, "header", sanitized.header)
    sanitized.joint_names = list(joint_names)
    sanitized.points = []

    source_rows: list[tuple[list[float], float]] = [(current, 0.0)]
    previous_yaw = current[2]
    last_time = 0.0
    for point in points:
        positions = list(getattr(point, "positions", []) or [])
        if len(positions) < 3:
            continue
        pos = [float(positions[0]), float(positions[1]), float(positions[2])]
        pos[2] = unwrap_angle_near(pos[2], previous_yaw)
        previous_yaw = pos[2]
        source_time = max(duration_to_sec(getattr(point, "time_from_start", None)), last_time + min_dt)
        last_time = source_time
        source_rows.append((pos, source_time))

    output_times = [0.0]
    for index in range(1, len(source_rows)):
        previous_pos, previous_time = source_rows[index - 1]
        pos, source_time = source_rows[index]
        source_dt = max(min_dt, float(source_time) - float(previous_time))
        required_dt = required_base_segment_dt(
            previous_pos,
            pos,
            min_dt=min_dt,
            max_vx=max_vx,
            max_vy=max_vy,
            max_wz=max_wz,
        )
        output_times.append(output_times[-1] + max(source_dt * extra_time_scale, required_dt))

    for (pos, _source_time), output_time in zip(source_rows, output_times, strict=False):
        point = JointTrajectoryPoint()
        point.positions = [float(pos[0]), float(pos[1]), float(pos[2])]
        point.velocities = [0.0, 0.0, 0.0]
        point.accelerations = [0.0, 0.0, 0.0]
        point.time_from_start = sec_to_duration(output_time)
        sanitized.points.append(point)

    sanitized_summary = base_trajectory_diagnostics(sanitized, limits=(max_vx, max_vy, max_wz))
    diagnostics = {
        "raw": raw_summary,
        "sanitized": sanitized_summary,
        "limits": {
            "max_vx_mps": max_vx,
            "max_vy_mps": max_vy,
            "max_wz_radps": max_wz,
            "min_dt_s": min_dt,
            "time_scale": extra_time_scale,
        },
        "current_state_inserted": True,
        "velocities_stripped": True,
        "accelerations_stripped": True,
    }
    try:
        setattr(sanitized, "_orchid_base_trajectory_diagnostics", diagnostics)
    except Exception:
        pass
    log.info("%s base trajectory diagnostics: %s", log_prefix, diagnostics)
    return sanitized


def base_trajectory_diagnostics(
    trajectory: Any,
    *,
    limits: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    joint_names = [str(name) for name in getattr(trajectory, "joint_names", []) or []]
    points = list(getattr(trajectory, "points", []) or [])
    if not points or not all(name in joint_names for name in BASE_JOINT_NAMES):
        return {"point_count": len(points), "reason": "not_base_trajectory"}
    indices = [joint_names.index(name) for name in BASE_JOINT_NAMES]
    rows: list[tuple[list[float], float]] = []
    for point in points:
        positions = list(getattr(point, "positions", []) or [])
        if max(indices) >= len(positions):
            continue
        rows.append((
            [float(positions[index]) for index in indices],
            duration_to_sec(getattr(point, "time_from_start", None)),
        ))
    if not rows:
        return {"point_count": len(points), "reason": "missing_positions"}
    summary: dict[str, Any] = {
        "point_count": len(rows),
        "duration_s": float(rows[-1][1]),
        "first": rows[0][0],
        "last": rows[-1][0],
        "min_dt_s": None,
        "max_abs_dx_m": 0.0,
        "max_abs_dy_m": 0.0,
        "max_abs_dyaw_rad": 0.0,
        "max_abs_vx_mps": 0.0,
        "max_abs_vy_mps": 0.0,
        "max_abs_wz_radps": 0.0,
        "limit_violations": [],
    }
    min_dt: float | None = None
    for index in range(1, len(rows)):
        previous, previous_time = rows[index - 1]
        current, current_time = rows[index]
        dt = float(current_time) - float(previous_time)
        if min_dt is None or dt < min_dt:
            min_dt = dt
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        dyaw = current[2] - previous[2]
        summary["max_abs_dx_m"] = max(float(summary["max_abs_dx_m"]), abs(dx))
        summary["max_abs_dy_m"] = max(float(summary["max_abs_dy_m"]), abs(dy))
        summary["max_abs_dyaw_rad"] = max(float(summary["max_abs_dyaw_rad"]), abs(dyaw))
        if dt > 0.0:
            summary["max_abs_vx_mps"] = max(float(summary["max_abs_vx_mps"]), abs(dx / dt))
            summary["max_abs_vy_mps"] = max(float(summary["max_abs_vy_mps"]), abs(dy / dt))
            summary["max_abs_wz_radps"] = max(float(summary["max_abs_wz_radps"]), abs(dyaw / dt))
        else:
            summary["limit_violations"].append({"segment": index, "reason": "non_positive_dt", "dt_s": dt})
    summary["min_dt_s"] = min_dt
    if limits is not None:
        max_vx, max_vy, max_wz = limits
        if float(summary["max_abs_vx_mps"]) > max_vx:
            summary["limit_violations"].append({"joint": "odom_x", "value": summary["max_abs_vx_mps"], "limit": max_vx})
        if float(summary["max_abs_vy_mps"]) > max_vy:
            summary["limit_violations"].append({"joint": "odom_y", "value": summary["max_abs_vy_mps"], "limit": max_vy})
        if float(summary["max_abs_wz_radps"]) > max_wz:
            summary["limit_violations"].append({"joint": "odom_t", "value": summary["max_abs_wz_radps"], "limit": max_wz})
    return summary


def required_base_segment_dt(
    previous: list[float],
    current: list[float],
    *,
    min_dt: float,
    max_vx: float,
    max_vy: float,
    max_wz: float,
) -> float:
    return max(
        float(min_dt),
        abs(float(current[0]) - float(previous[0])) / max(max_vx, 1e-6),
        abs(float(current[1]) - float(previous[1])) / max(max_vy, 1e-6),
        abs(float(current[2]) - float(previous[2])) / max(max_wz, 1e-6),
    )


def unwrap_angle_near(angle: float, reference: float) -> float:
    value = float(angle)
    reference = float(reference)
    while value - reference > math.pi:
        value -= 2.0 * math.pi
    while value - reference < -math.pi:
        value += 2.0 * math.pi
    return value


def duration_to_sec(duration: Any) -> float:
    if duration is None:
        return 0.0
    return float(getattr(duration, "sec", 0)) + float(getattr(duration, "nanosec", 0)) / 1_000_000_000.0


def sec_to_duration(seconds: float) -> Any:
    from builtin_interfaces.msg import Duration

    total_ns = max(0, int(round(float(seconds) * 1_000_000_000)))
    return Duration(sec=total_ns // 1_000_000_000, nanosec=total_ns % 1_000_000_000)


def _positive_scale(value: Any, *, default: float) -> float:
    try:
        scale = float(value)
        return scale if scale > 0.0 else default
    except (TypeError, ValueError):
        return default
