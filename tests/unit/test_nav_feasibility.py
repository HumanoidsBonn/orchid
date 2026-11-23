"""Tests for batched navigation feasibility checks."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.core.sensor_data_center import SensorMessage
from src.plugin.base_pose_selector import BasePose2D
from src.plugin.nav_feasibility import NavFeasibilityConfig, NavigationFeasibilityChecker


def _grid(width=10, height=10, resolution=0.1, occupied=None, frame_id="map"):
    data = [0] * (width * height)
    for u, v, value in occupied or []:
        data[v * width + u] = value
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
        ),
        data=data,
    )


def _footprint(points):
    return SimpleNamespace(
        polygon=SimpleNamespace(
            points=[SimpleNamespace(x=x, y=y) for x, y in points],
        )
    )


def _fake_sdc(grid_msg=None, *, local=True, footprint_msg=None):
    sdc = Mock()

    def latest(sensor_name, timeout_s=0.2):
        if sensor_name == "local_costmap" and local and grid_msg is not None:
            return SensorMessage("/local_costmap", grid_msg, 1.0, grid_msg.header.frame_id)
        if sensor_name == "map" and (not local) and grid_msg is not None:
            return SensorMessage("/map", grid_msg, 1.0, grid_msg.header.frame_id)
        if sensor_name == "local_footprint" and footprint_msg is not None:
            return SensorMessage("/footprint", footprint_msg, 1.0, "base_footprint")
        return None

    sdc.get_latest = Mock(side_effect=latest)
    sdc.get_pose_at = Mock(return_value=None)
    return sdc


def test_uses_map_fallback_and_fetches_sources_once():
    grid = _grid()
    sdc = _fake_sdc(grid, local=False)
    checker = NavigationFeasibilityChecker(sdc, NavFeasibilityConfig(robot_radius=0.05))

    results = checker.check_candidates(
        [
            BasePose2D(0.5, 0.5, 0.0, "map"),
            BasePose2D(0.6, 0.5, 0.0, "map"),
        ]
    )

    assert all(result["success"] for result in results)
    assert all(result["source"] == "map" for result in results)
    assert [call.args[0] for call in sdc.get_latest.call_args_list] == [
        "local_costmap",
        "map",
        "local_footprint",
    ]


def test_rejects_candidate_when_any_footprint_cell_is_occupied():
    grid = _grid(occupied=[(5, 5, 100)])
    sdc = _fake_sdc(grid, footprint_msg=_footprint([(0.0, 0.0), (0.05, 0.0), (0.0, 0.05)]))
    checker = NavigationFeasibilityChecker(sdc, NavFeasibilityConfig(robot_radius=0.05))

    result = checker.check_candidates([BasePose2D(0.5, 0.5, 0.0, "map")])[0]

    assert result["success"] is False
    assert result["reason"] == "footprint_costmap_blocked"
    assert result["metadata"]["blocked_count"] >= 1
    assert result["metadata"]["footprint_blocked_count"] >= 1
    assert result["metadata"]["near_goal_blocked_count"] == 0


def test_near_goal_circle_uses_separate_threshold_and_reason():
    grid = _grid(occupied=[(5, 5, 80)])
    sdc = _fake_sdc(grid, footprint_msg=_footprint([(-0.04, -0.04), (0.04, -0.04), (0.04, 0.04), (-0.04, 0.04)]))
    checker = NavigationFeasibilityChecker(
        sdc,
            NavFeasibilityConfig(
                robot_radius=0.05,
                occupied_threshold=100,
                near_goal_radius_m=0.08,
                near_goal_occupied_threshold=50,
            ),
        )

    result = checker.check_candidates([BasePose2D(0.5, 0.5, 0.0, "map")])[0]

    assert result["success"] is False
    assert result["reason"] == "near_goal_costmap_blocked"
    assert result["metadata"]["footprint_blocked_count"] == 0
    assert result["metadata"]["near_goal_blocked_count"] >= 1


def test_published_footprint_inflation_expands_checked_polygon():
    grid = _grid(occupied=[(5, 4, 100)])
    footprint = _footprint([(-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)])
    without_inflation = NavigationFeasibilityChecker(
        _fake_sdc(grid, footprint_msg=footprint),
        NavFeasibilityConfig(footprint_inflation_m=0.0),
    ).check_candidates([BasePose2D(0.45, 0.45, 0.0, "map")])[0]
    with_inflation = NavigationFeasibilityChecker(
        _fake_sdc(grid, footprint_msg=footprint),
        NavFeasibilityConfig(footprint_inflation_m=0.14),
    ).check_candidates([BasePose2D(0.45, 0.45, 0.0, "map")])[0]

    assert without_inflation["success"] is True
    assert with_inflation["success"] is False
    assert with_inflation["reason"] == "footprint_costmap_blocked"


def test_unknown_cells_obey_unknown_policy():
    grid = _grid(occupied=[(5, 5, -1)])
    sdc = _fake_sdc(grid)

    allowed = NavigationFeasibilityChecker(
        sdc,
        NavFeasibilityConfig(robot_radius=0.05, unknown_is_blocked=False),
    ).check_candidates([BasePose2D(0.5, 0.5, 0.0, "map")])[0]
    blocked = NavigationFeasibilityChecker(
        sdc,
        NavFeasibilityConfig(robot_radius=0.05, unknown_is_blocked=True),
    ).check_candidates([BasePose2D(0.5, 0.5, 0.0, "map")])[0]

    assert allowed["success"] is True
    assert blocked["success"] is False
    assert blocked["reason"] == "candidate_unknown_grid"


def test_out_of_bounds_footprint_cell_rejects_candidate():
    sdc = _fake_sdc(_grid())
    checker = NavigationFeasibilityChecker(sdc, NavFeasibilityConfig(robot_radius=0.05))

    result = checker.check_candidates([BasePose2D(-0.1, 0.5, 0.0, "map")])[0]

    assert result["success"] is False
    assert result["reason"] == "candidate_out_of_grid"


def test_batch_transform_uses_one_tf_lookup_for_candidate_frame():
    grid = _grid(frame_id="map")
    sdc = _fake_sdc(grid)
    tf = np.eye(4, dtype=np.float32)
    tf[0, 3] = 0.1
    sdc.get_pose_at = Mock(
        return_value={
            "mat": tf,
            "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(0.25), "w": math.cos(0.25)},
        }
    )
    checker = NavigationFeasibilityChecker(sdc, NavFeasibilityConfig(robot_radius=0.05))

    results = checker.check_candidates(
        [
            BasePose2D(0.4, 0.5, 0.0, "odom"),
            BasePose2D(0.5, 0.5, 0.0, "odom"),
        ]
    )

    assert all(result["success"] for result in results)
    sdc.get_pose_at.assert_called_once_with("odom", "map")
    assert results[0]["metadata"]["candidate_in_grid"]["x"] == np.float32(0.5)


def test_missing_sources_can_reject_all_candidates():
    sdc = _fake_sdc(None)
    checker = NavigationFeasibilityChecker(sdc, NavFeasibilityConfig(missing_policy="reject"))

    result = checker.check_candidates([BasePose2D(0.5, 0.5, 0.0, "map")])[0]

    assert result["success"] is False
    assert "missing_local_costmap" in result["reason"]
    assert result["metadata"]["hard_nav_missing"] is True
