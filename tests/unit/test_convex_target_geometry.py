"""Focused target geometry and scene-integration regressions; no robot execution."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.core.collision_geometry import inflated_convex_hull, write_collision_mesh
from src.core.context import Context
from src.core.sensor_data_center import SensorDataCenter
from src.plugin.actions.manipulation import PickAtPose, PickTarget, _AddCollisionMeshAction
from src.plugin.actions.grasp_alignment import PickTargetWithGraspAlignment
from src.plugin.controllers.moveit import MoveItArmController


@pytest.fixture
def cube():
    return np.array([[x, y, z] for x in (2., 4.) for y in (3., 5.) for z in (6., 8.)])


@pytest.mark.parametrize('factor', [1., 1.1, 1.5])
def test_closed_hull_uniform_inflation_and_cleaning(cube, factor):
    points = np.vstack([cube, cube[0], [np.nan, 0, 0], [0, np.inf, 0]])
    hull = inflated_convex_hull(points, factor)
    assert hull.is_watertight()
    assert hull.is_edge_manifold(allow_boundary_edges=False)
    assert hull.get_volume() == pytest.approx(8 * factor ** 3)
    np.testing.assert_allclose(hull.get_max_bound() - hull.get_min_bound(), [2 * factor] * 3)
    np.testing.assert_allclose(hull.get_center(), [3, 4, 7])


@pytest.mark.parametrize('action', [PickTarget, PickAtPose, PickTargetWithGraspAlignment])
@pytest.mark.parametrize('factor', [0., .99, float('nan'), float('inf'), -float('inf')])
def test_pick_factor_validation(action, factor):
    with pytest.raises(ValueError, match='finite and >= 1'):
        action(collision_hull_inflation_factor=factor)


@pytest.mark.parametrize('points', [[], [[0, 0, 0]], [[x,y,0] for x in (0,1) for y in (0,1)], [[x,x,x] for x in range(5)], [[np.nan]*3]*5])
def test_degenerate_cloud_rejected(points):
    with pytest.raises(ValueError):
        inflated_convex_hull(points)


def test_sensor_route_and_distinct_files(cube):
    sdc = SensorDataCenter.__new__(SensorDataCenter)
    first = sdc.pcd_to_mesh(cube, convex_hull=True, inflation_factor=1.1)
    second = sdc.pcd_to_mesh(cube, convex_hull=True, inflation_factor=1.2)
    assert first != second
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(first)
    mesh.remove_duplicated_vertices()  # STL stores per-triangle vertices.
    assert mesh.is_watertight()
    np.testing.assert_allclose(mesh.get_max_bound()-mesh.get_min_bound(), [2.2]*3, atol=1e-6)


def controller():
    arm = MoveItArmController.__new__(MoveItArmController)
    arm.hardware_config = {}
    arm._ros_node = None
    arm._owned_collision_objects = {}
    arm._context = SimpleNamespace(sensor_data_center=None)
    proxy = Mock()
    arm._planner = SimpleNamespace(config={}, _module_proxy=lambda context: proxy)
    return arm, proxy


def response(obj):
    from moveit_msgs.srv import GetPlanningScene
    result = GetPlanningScene.Response()
    result.scene.world.collision_objects = [obj]
    return result


def test_moveit_frame_geometry_and_readback(cube):
    arm, proxy = controller()
    path = write_collision_mesh(inflated_convex_hull(cube))
    expected = arm._collision_object_from_mesh(path, 'pick_target_collision', frame_id='map')
    assert expected.header.frame_id == 'map'
    assert expected.id == 'pick_target_collision'
    np.testing.assert_allclose([expected.meshes[0].vertices[0].x, expected.meshes[0].vertices[0].y, expected.meshes[0].vertices[0].z], np.asarray(__import__('open3d').io.read_triangle_mesh(path).vertices)[0])
    proxy.call.return_value = response(expected)
    arm._apply_collision_object = Mock(return_value=True)
    assert arm.add_collision_mesh(path, 'pick_target_collision', frame_id='map')
    assert arm._owned_collision_objects['pick_target_collision'].header.frame_id == 'map'
    assert proxy.call.call_args.args[1]['request'].components.components == 16
    for mismatch in ('geometry', 'frame', 'missing'):
        observed = copy.deepcopy(expected)
        if mismatch == 'geometry': observed.meshes[0].vertices[0].z += .1
        if mismatch == 'frame': observed.header.frame_id = 'unknown'
        if mismatch == 'missing': observed.id = 'other'
        proxy.call.return_value = response(observed)
        assert not arm.add_collision_mesh(path, 'pick_target_collision', frame_id='map')
    arm._apply_collision_object.return_value = False
    proxy.call.reset_mock()
    assert not arm.add_collision_mesh(path, frame_id='map')
    proxy.call.assert_not_called()


def test_readback_equivalent_geometry_transformed_to_scene_frame(cube):
    arm, proxy = controller()
    expected = arm._collision_object_from_mesh(write_collision_mesh(inflated_convex_hull(cube)), 'target', frame_id='map')
    observed = copy.deepcopy(expected)
    observed.header.frame_id = 'odom'
    observed.mesh_poses[0].position.z = 2.
    arm._context.sensor_data_center = SimpleNamespace(get_pose_at=Mock(return_value={'position': {'z':2.}, 'orientation': {'w':1.}}))
    proxy.call.return_value = response(observed)
    assert arm._verify_collision_object(expected)
    arm._context.sensor_data_center.get_pose_at.assert_called_once_with('map', 'odom')


def synchronous_context():
    ctx = Context()
    ctx.arm = SimpleNamespace(add_collision_mesh=Mock(return_value=True))
    ctx.run_in_context_loop = lambda fn, callback=None, error_callback=None: callback(fn())
    return ctx


@pytest.mark.parametrize('failure', ['mesh', 'add', 'exception'])
@pytest.mark.parametrize('convex', [True, False])
def test_target_add_failures_propagate(failure, convex, monkeypatch, cube):
    import src.plugin.actions.manipulation as manipulation
    ctx = synchronous_context()
    if failure == 'mesh':
        monkeypatch.setattr(manipulation, '_create_mesh_for_collision_world', Mock(side_effect=ValueError('degenerate')))
    else:
        monkeypatch.setattr(manipulation, '_create_mesh_for_collision_world', Mock(return_value='target.stl'))
        ctx.arm.add_collision_mesh = Mock(return_value=False) if failure == 'add' else Mock(side_effect=RuntimeError('service failed'))
    leaf = _AddCollisionMeshAction(cube, name='target', convex_hull=convex, frame_id='map', required=True)
    leaf.start(ctx)
    result = leaf.check_result(ctx)
    assert result.status == 'failure' and not result.failure.retryable
    assert failure != 'mesh' or not ctx.arm.add_collision_mesh.called


def test_target_snapshot_and_stale_callback(monkeypatch, cube):
    import src.plugin.actions.manipulation as manipulation
    mesh = Mock(return_value='target.stl')
    monkeypatch.setattr(manipulation, '_create_mesh_for_collision_world', mesh)
    ctx = synchronous_context()
    queued = []
    ctx.run_in_context_loop = lambda fn, callback=None, error_callback=None: queued.append((fn, callback))
    leaf = _AddCollisionMeshAction(cube, name='target', convex_hull=True, frame_id='map')
    leaf.start(ctx)
    cube[:] = 0
    leaf.cancel(ctx)
    queued[0][1](queued[0][0]())
    ctx.arm.add_collision_mesh.assert_not_called()
    leaf.reset(); leaf.start(ctx)
    queued[0][1](SimpleNamespace(status='success'))
    assert leaf.check_result(ctx).status == 'running'
    queued[1][1](queued[1][0]())
    assert leaf.check_result(ctx).status == 'success'
    assert np.any(mesh.call_args_list[0].args[1] != 0)
    ctx.arm.add_collision_mesh.assert_called_once_with('target.stl', name='target', frame_id='map')


def test_workflow_factor_forwarding_and_failure_stops_dispatch(monkeypatch):
    from tests.unit.test_action_subtrees import _pick_context, _finish
    from src.tree.bt_node_wrapper import ActionScope
    ctx = _pick_context(monkeypatch, {})
    pick = PickTarget('jar', collision_hull_inflation_factor=1.4)
    root = pick.build_node(ctx, 'Pick')
    assert isinstance(root, ActionScope)
    assert pick.param_dict()['collision_hull_inflation_factor'] == 1.4
    assert pick._pick._target_mesh.inflation_factor == 1.4
    direct = PickAtPose(collision_hull_inflation_factor=1.2)
    ctx.arm.add_collision_mesh = Mock(return_value=False)
    result = _finish(direct, ctx)
    assert result.status == 'failure' and 'scene readback' in result.reason
    assert not ctx.arm.pose_requests and ctx.gripper.last_close_force is None
    aligned = PickTargetWithGraspAlignment(collision_hull_inflation_factor=1.3)
    aligned.build_node(ctx, 'Aligned')
    assert aligned._object_mesh.convex_hull and aligned._object_mesh.inflation_factor == 1.3
    assert not aligned._surround_mesh.convex_hull


def test_hsr_mesh_frame_and_failure(cube):
    from src.plugin.controllers.hsrb_arm import HsrbArmController
    arm = HsrbArmController.__new__(HsrbArmController)
    arm._collision_world = SimpleNamespace(add_mesh=Mock())
    assert arm.add_collision_mesh('target.stl', 'target', frame_id='map')
    arm._collision_world.add_mesh.assert_called_once_with(filename='target.stl', frame_id='map', name='target')
    arm._collision_world.add_mesh.side_effect = RuntimeError('backend failed')
    assert not arm.add_collision_mesh('target.stl')


def test_alignment_target_failure_propagates_before_fk_or_motion(monkeypatch, cube):
    from tests.unit.test_action_subtrees import _pick_context, _finish
    ctx = _pick_context(monkeypatch, {})
    ctx.world_state.set('jar_object_pcd', cube)
    ctx.world_state.set('jar_pcd_frame', 'map')
    ctx.world_state.set('last_pick_start_base_pose', {'x':0., 'y':0., 'yaw':0., 'frame_id':'map'})
    ctx.arm.add_collision_mesh = Mock(return_value=False)
    action = PickTargetWithGraspAlignment('jar')
    result = _finish(action, ctx)
    assert result.status == 'failure' and 'scene readback' in result.reason
    assert action._object_mesh.frame_id == 'map'
    assert not ctx.arm.pose_requests and ctx.gripper.last_close_force is None
    ctx.arm.add_collision_mesh.assert_called_once_with('target.ply', name='pick_target_collision', frame_id='map')


def test_scene_readback_service_failures(cube):
    arm, proxy = controller()
    expected = arm._collision_object_from_mesh(write_collision_mesh(inflated_convex_hull(cube)), 'target')
    arm._collision_object_from_mesh = Mock(return_value=expected)
    arm._apply_collision_object = Mock(return_value=True)
    for value in (None, {'error': 'service timeout'}, SimpleNamespace(scene=None)):
        proxy.call.return_value = value
        assert not arm.add_collision_mesh('target.stl')
    proxy.call.side_effect = RuntimeError('transport unavailable')
    assert not arm.add_collision_mesh('target.stl')


def test_dense_target_cleanup_rejects_cloud_without_neighbours():
    points = np.random.default_rng(3).uniform(-100, 100, (60, 3))
    with pytest.raises(ValueError, match='after outlier cleanup'):
        inflated_convex_hull(points)
