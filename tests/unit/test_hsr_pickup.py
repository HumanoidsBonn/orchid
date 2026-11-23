"""Focused pickup geometry, transport, lifecycle and execution regressions."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from src.core.context import Context
from src.plugin.actions.hsr_pickup.model import (PICKUP_JOINTS, PickupBackendSettings, PickupInputs, PickupPlan, PickupRequest, PreparedMotion, pickup_geometry)
from src.plugin.actions.hsr_pickup import HsrPickup, PlanHsrPickup
from src.plugin.actions.hsr_pickup.planning import (PickupPlanningClient, add_pickup_groups, pickup_observation, trajectory_message, import_simple_grasp)
from src.plugin.actions.trajectory import ExecuteJointTrajectory, trajectory_subset
from src.plugin.controllers.ros2_control_trajectory import Ros2ControlController, _ControllerGoal

def geometry_inputs():
    pose = np.eye(4)
    # Hand +z points along world +x.
    pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    pose[:3, 3] = [.6, .1, .75]
    points = pose[:3, 3] + np.array([[x, y, z] for x in (-.03, .03) for y in (-.02, .02) for z in (-.03, .03)])
    return pose, points


@pytest.mark.parametrize("mode", ["base_only", "arm_only", "coordinated"])
def test_geometry_keeps_grasp_and_resolves_clearance(mode):
    pose, points = geometry_inputs()
    original = pose.copy()
    geometry = pickup_geometry(PickupRequest("jar", approach_mode=mode), pose, points, "odom")
    assert np.array_equal(geometry.grasp, original)
    assert geometry.distances == (.25, .06, .60)
    assert np.allclose(geometry.pregrasp(.12)[:3, 3], [.48, .1, .75])
    assert np.array_equal(pose, original)
    assert geometry.frame_id == "odom"


def test_vertical_and_tilted_geometry_require_compatible_mode():
    pose = np.eye(4)
    points = np.array([[0, 0, -.02], [0, 0, .02]])
    with pytest.raises(ValueError, match="horizontal insertion"):
        pickup_geometry(PickupRequest("jar", "base_only"), pose, points, "odom")
    geometry = pickup_geometry(PickupRequest("jar", "arm_only"), pose, points, "odom")
    assert geometry.direction == (0, 0, 1)
    assert geometry.pregrasp(.12)[2, 3] == -.12
    with pytest.raises(ValueError, match="horizontal projection"):
        pickup_geometry(PickupRequest("jar", "arm_only", "horizontal_projection"), pose, points, "odom")


def test_explicit_projection_does_not_change_grasp():
    from scipy.spatial.transform import Rotation
    pose, points = geometry_inputs()
    pose[:3, :3] = Rotation.from_rotvec([0, .2, 0]).as_matrix() @ pose[:3, :3]
    geometry = pickup_geometry(PickupRequest("jar", backoff_direction="horizontal_projection"), pose, points, "odom")
    assert geometry.direction[2] == 0
    assert np.array_equal(geometry.grasp, pose)


def test_impossible_clearance_and_invalid_geometry_fail():
    pose, points = geometry_inputs()
    points[:, 0] -= .9
    with pytest.raises(ValueError, match="clearance"):
        pickup_geometry(PickupRequest("jar"), pose, points, "odom")
    with pytest.raises(ValueError, match="empty target"):
        pickup_geometry(PickupRequest("jar"), pose, np.full((5, 3), np.nan), "odom")
    pose[0, 0] = 3
    with pytest.raises(ValueError, match="rotation"):
        pickup_geometry(PickupRequest("jar"), pose, points, "odom")


@pytest.mark.parametrize("values", [{"approach_mode": "typo"}, {"min_insertion_m": .8}, {"clearance_m": float("nan")}, {"max_candidates": 9}, {"collision_hull_inflation_factor": .5}, {"target_collision_geometry": "typo"}])
def test_explicit_request_validation(values):
    with pytest.raises(ValueError):
        PickupRequest("jar", **values)


def test_new_groups_do_not_modify_existing_groups():
    result = add_pickup_groups('<robot name="hsrc"><group name="arm"><chain base_link="base_link" tip_link="hand_palm_link"/></group></robot>')
    assert 'name="pickup_arm"' in result and 'name="arm_lift_joint"' in result
    assert 'name="pickup_arm_yaw"' in result and 'name="odom_t"' in result
    assert "wrist_ft_sensor_frame_joint" not in result
    assert 'name="arm"' in result
    with pytest.raises(ValueError, match="reserved"):
        add_pickup_groups(result)


def test_observation_is_copied_at_entry():
    context = Context()
    pose, points = geometry_inputs()
    context.world_state.set("jar_estimated_pick_pose_mat", pose)
    context.world_state.set("jar_object_pcd", points)
    context.world_state.set("jar_pcd_frame", "odom")
    observation = pickup_observation(context, "jar")
    pose[2, 3] = 8
    points[:] = 0
    assert observation.grasp[2, 3] == .75
    assert observation.target_points.any()


def test_import_simple_side_frame_maps_up_to_palm_x_and_opening_to_horizontal():
    pose, _ = geometry_inputs()
    pose[:3, 0] *= -1
    fixed, corrected = import_simple_grasp(pose, "+z")
    assert corrected and np.linalg.det(fixed[:3, :3]) == pytest.approx(1.)
    assert np.array_equal(fixed[:3, 3], pose[:3, 3])
    assert np.array_equal(fixed[:3, 2], pose[:3, 2])
    assert np.array_equal(fixed[:3, 0], pose[:3, 1])
    assert abs(fixed[2, 1]) < 1e-8


def test_trajectory_transport_checks_timing_and_dimensions():
    with pytest.raises(ValueError, match="timing"):
        trajectory_message({"joint_names": ["j"], "points": [{"positions": [0.], "time_from_start": 0.}] * 2})
    with pytest.raises(ValueError, match="invalid prepared"):
        trajectory_message({"joint_names": ["j"], "points": [{"positions": [0., 1.], "time_from_start": 0.}]})


def done(value):
    future = Future()
    future.set_result(value)
    return future


def sync_client(accepted=True):
    from rclpy.time import Time
    client = Ros2ControlController({"ros2_control": {"controllers": [
        {"name": "arm", "action": "/arm", "joints": ["a"]},
        {"name": "base", "action": "/base", "joints": ["b"]}]}, "goal_response_retry_settle_s": 0.})
    clock = Mock()
    clock.now.return_value = Time(seconds=10)
    client._node = Mock()
    client._node.get_clock.return_value = clock
    client._reset_progress_tracking = Mock()
    handles = [Mock(accepted=True), Mock(accepted=accepted)]
    for handle in handles:
        handle.get_result_async.return_value = Future()
    clients = {}
    for name, handle in zip(("/arm", "/base"), handles):
        clients[name] = Mock()
        clients[name].send_goal_async.return_value = done(handle)
    client._client = lambda name: clients[name]
    return client, clients, handles


def test_synchronized_dispatch_preserves_shared_stamp_and_timing():
    client, clients, handles = sync_client()
    trajectory = trajectory_message({"joint_names": ["a", "b"], "points": [
        {"positions": [0., 0.], "time_from_start": 0.}, {"positions": [1., 1.], "time_from_start": 2.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    assert client.get_status() == "in_progress"
    arm = clients["/arm"].send_goal_async.call_args.args[0].trajectory
    base = clients["/base"].send_goal_async.call_args.args[0].trajectory
    assert arm.header.stamp == base.header.stamp and arm.header.stamp.sec == 10
    assert arm.points[-1].time_from_start == base.points[-1].time_from_start
    assert trajectory.header.stamp.sec == 0
    client.cancel_goal(); client.cancel_goal()
    for handle in handles:
        handle.cancel_goal_async.assert_called_once()


def test_synchronized_rejection_cancels_peers_without_retry():
    client, clients, handles = sync_client(accepted=False)
    trajectory = trajectory_message({"joint_names": ["a", "b"], "points": [{"positions": [0., 0.], "time_from_start": 0.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    assert client.get_status() == "failed"
    handles[0].cancel_goal_async.assert_called_once()
    assert all(c.send_goal_async.call_count == 1 for c in clients.values())


def test_synchronized_acceptance_after_three_seconds_uses_five_second_hold(monkeypatch):
    from rclpy.time import Time
    client, clients, handles = sync_client()
    elapsed = [0.]
    futures = [Future(), Future()]
    for name, future in zip(('/arm', '/base'), futures):
        clients[name].send_goal_async.return_value = future
    monkeypatch.setattr('src.plugin.controllers.ros2_control_trajectory.time.monotonic', lambda: elapsed[0])
    client._node.get_clock.return_value.now.side_effect = lambda: Time(seconds=10. + elapsed[0])
    def advance(_timeout):
        elapsed[0] += 1.
        if elapsed[0] == 3.:
            for future, handle in zip(futures, handles):
                future.set_result(handle)
    client._spin_once = advance
    trajectory = trajectory_message({'joint_names': ['a', 'b'], 'points': [
        {'positions': [0., 0.], 'time_from_start': 0.},
        {'positions': [1., 1.], 'time_from_start': 2.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    assert client.get_status() == 'in_progress'
    for action in clients.values():
        sent = action.send_goal_async.call_args.args[0].trajectory
        assert sent.header.stamp.sec == 10
        assert sent.points[0].time_from_start.sec == 0
        assert sent.points[1].time_from_start.sec == 5
        assert sent.points[-1].time_from_start.sec == 7
        assert action.send_goal_async.call_count == 1


def test_pending_goal_is_cancelled_if_acceptance_arrives_after_cancellation():
    client, _, _ = sync_client()
    pending = Future()
    goal = _ControllerGoal("late", "/late", ["a"], goal_handle_future=pending)
    client._active_goals = [goal]
    client.cancel_goal(); client.cancel_goal()
    handle = Mock(accepted=True)
    pending.set_result(handle)
    handle.cancel_goal_async.assert_called_once()


def test_synchronized_timeout_never_retries_a_component(monkeypatch):
    client, clients, handles = sync_client()
    pending = Future()
    clients["/base"].send_goal_async.return_value = pending
    ticks = iter([0., 10., 10., 10.])
    monkeypatch.setattr("src.plugin.controllers.ros2_control_trajectory.time.monotonic", lambda: next(ticks, 10.))
    trajectory = trajectory_message({"joint_names": ["a", "b"], "points": [{"positions": [0., 0.], "time_from_start": 0.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    assert client.get_status() == "failed"
    pending.set_result(handles[1])
    for handle in handles:
        handle.cancel_goal_async.assert_called_once()
    assert all(c.send_goal_async.call_count == 1 for c in clients.values())


def test_synchronized_unowned_joints_fail_before_send():
    client, clients, _ = sync_client()
    trajectory = trajectory_message({"joint_names": ["unknown"], "points": [{"positions": [0.], "time_from_start": 0.}]})
    client.send_synchronized_trajectory_goal(trajectory)
    assert client.get_status() == "failed"
    assert not any(c.send_goal_async.called for c in clients.values())


def test_synchronized_cancel_prevents_any_component_dispatch():
    client, clients, _ = sync_client()
    trajectory = trajectory_message({"joint_names": ["a", "b"], "points": [{"positions": [0., 0.], "time_from_start": 0.}]})
    client.send_synchronized_trajectory_goal(trajectory, cancelled=lambda: True)
    assert client.get_failure().code.value == "cancelled"
    assert not any(c.send_goal_async.called for c in clients.values())


def test_backend_settings_resolve_existing_touch_policy_once():
    hardware = {"planners": {"moveit": {"collision_relaxation_links": ["hand"]}}}
    settings = PickupBackendSettings.from_hardware(hardware)
    hardware["planners"]["moveit"]["collision_relaxation_links"].append("later")
    assert settings.touch_links == ("hand",)
    with pytest.raises(TypeError):
        PickupBackendSettings.from_hardware({"hsr_pickup": {"random_option": 1}})


def test_hsrb_route_keeps_prepared_timeline():
    from src.plugin.controllers.hsrb_arm import HsrbArmController
    controller = HsrbArmController.__new__(HsrbArmController)
    controller._execution_backend = "ros2_control"
    controller._ros2_control = Mock()
    controller._ros2_control.get_status.return_value = "in_progress"
    trajectory = object()
    cancelled = lambda: False
    controller.send_synchronized_trajectory_goal(trajectory, cancelled=cancelled)
    controller._ros2_control.send_synchronized_trajectory_goal.assert_called_once_with(trajectory, cancelled=cancelled)


def test_capture_service_client_is_reused_without_another_executor():
    node = Mock()
    response = object()
    node.create_client.return_value.call_async.side_effect = lambda request: done(response)
    planner = PickupPlanningClient(node, PickupBackendSettings())
    for _ in range(2):
        assert planner._call("/read_only", object, object()) is response
    node.create_client.assert_called_once_with(object, "/read_only")
    assert planner._service_counts == {"/read_only": 2}


def test_service_timeout_is_a_backend_failure_not_collision_rejection():
    from src.core.failure import MotionFailure
    node = Mock()
    pending = Future()
    node.create_client.return_value.call_async.return_value = pending
    planner = PickupPlanningClient(node, PickupBackendSettings())
    with pytest.raises(MotionFailure) as failure:
        planner._call("/read_only", object, object(), timeout=0.)
    assert failure.value.failure.backend_code == "service_timeout"
    assert not failure.value.failure.retryable
    node.create_client.return_value.remove_pending_request.assert_called_once_with(pending)


@pytest.mark.parametrize("yaw", [0., .7, -2.1])
@pytest.mark.parametrize("axis", ["+z", "-z", "+x", "-x"])
def test_legacy_side_basis_maps_to_horizontal_hsr_opening(yaw, axis):
    from src.plugin.actions.grasp_alignment import _hand_rotation_from_yaw
    pose = np.eye(4)
    pose[:3,:3] = _hand_rotation_from_yaw(yaw,axis)
    pose[:3,3] = [.5,.2,.75]
    converted, changed = import_simple_grasp(pose,axis)
    expected = pose[:3,"xyz".index(axis[1])] * (-1 if axis[0] == "-" else 1)
    assert changed
    assert np.allclose(converted[:3,2],expected)
    assert np.allclose(converted[:3,0],[0,0,1])
    assert abs(converted[2,1]) < 1e-9
    assert np.linalg.det(converted[:3,:3]) == pytest.approx(1.)
    assert np.array_equal(converted[:3,3],pose[:3,3])
    again, changed_again = import_simple_grasp(converted,"+z")
    assert not changed_again and np.array_equal(again,converted)


def test_proper_top_down_pose_is_not_reoriented():
    pose = np.diag([1.,-1.,-1.,1.])
    result, changed = import_simple_grasp(pose,"+z")
    assert not changed and np.array_equal(result,pose)


@pytest.mark.parametrize("allow", [True, False])
def test_held_target_octomap_policy_is_explicit(allow):
    settings = PickupBackendSettings.from_hardware({"hsr_pickup":{"allow_held_target_octomap_contact":allow}})
    assert settings.allow_held_target_octomap_contact is allow
    with pytest.raises(ValueError, match="boolean"):
        PickupBackendSettings(allow_held_target_octomap_contact="true")



def test_auto_inputs_and_obsolete_fields():
    assert HsrPickup("jar").approach_mode == PlanHsrPickup("jar").approach_mode == "auto"
    assert PickupRequest("jar").approach_mode == "auto"
    for key in ("snapshot_key", "options", "composition_backend", "retreat_distance_m"):
        with pytest.raises((ValueError,TypeError)):
            PlanHsrPickup.from_params(target="jar",**{key:"obsolete"})


def dummy_plan():
    pose,cloud=geometry_inputs()
    from moveit_msgs.msg import PlanningScene,CollisionObject
    inputs=PickupInputs("odom",dict.fromkeys(PICKUP_JOINTS,0.),pose,cloud,[],"<robot name='hsrc'/>","",{},PlanningScene(),CollisionObject(),{})
    t=trajectory_message({"joint_names":list(PICKUP_JOINTS),"points":[{"positions":[0.]*8,"time_from_start":0.},{"positions":[.01]*8,"time_from_start":1.}]})
    motion=PreparedMotion("insertion",t,inputs.start_joints,dict.fromkeys(PICKUP_JOINTS,.01))
    return PickupPlan("jar","auto","arm_only","base_first",inputs,(motion,),(),{"reason":"pickup_planned"})


def test_planning_ignores_stale_callback_and_does_not_touch_controllers(monkeypatch):
    context=Context(arm=Mock(),gripper=Mock(),nav=Mock())
    plan=dummy_plan();context.world_state.set("jar_estimated_pick_pose_mat",plan.inputs.grasp);context.world_state.set("jar_object_pcd",plan.inputs.target_points);context.world_state.set("jar_pcd_frame","odom")
    jobs=[];context.run_in_context_loop=lambda work,**callbacks:jobs.append((work,callbacks))
    client=Mock();client.collect_inputs.return_value=plan.inputs;client.plan.return_value=plan
    action=PlanHsrPickup("jar");action._client=client
    action.start(context);work,callbacks=jobs[-1];result=work();action.cancel(context);callbacks["callback"](result)
    assert context.world_state.get("jar_pickup_plan") is None
    action.start(context);work,callbacks=jobs[-1];callbacks["callback"](work())
    assert action.check_result(context).status=="success"
    assert context.world_state.get("jar_pickup_plan.mode")=="arm_only"
    assert context.world_state.get("jar_pickup_plan.insertion") is plan.motions[0].trajectory
    assert not context.arm.mock_calls and not context.gripper.mock_calls and not context.nav.mock_calls


def test_trajectory_leaf_binds_fresh_inputs_without_scene_operations():
    context=Context(arm=Mock());context.arm.get_status.return_value="succeeded"
    context.run_in_context_loop=lambda work,**kwargs:work()
    first=dummy_plan().motions[0].trajectory;context.world_state.set("path",first)
    action=ExecuteJointTrajectory("path",joint_names=["arm_lift_joint"])
    action.start(context);sent=context.arm.send_synchronized_trajectory_goal.call_args.args[0]
    assert sent.joint_names==["arm_lift_joint"]
    assert action.check_result(context).status=="success"
    second=trajectory_message({"joint_names":["arm_lift_joint"],"points":[{"positions":[.2],"time_from_start":0.}]})
    context.world_state.set("path",second);action.start(context)
    assert list(context.arm.send_synchronized_trajectory_goal.call_args.args[0].points[0].positions)==[.2]
    context.arm.add_collision_mesh.assert_not_called()


def test_cancel_before_async_dispatch_sends_nothing():
    context=Context(arm=Mock());context.world_state.set("path",dummy_plan().motions[0].trajectory)
    work=[];context.run_in_context_loop=lambda fn,**kwargs:work.append(fn)
    action=ExecuteJointTrajectory("path");action.start(context);action.cancel(context);work[0]()
    assert not context.arm.mock_calls


def test_feedback_prints_contact_and_partial_path(monkeypatch):
    output=[];monkeypatch.setattr("src.plugin.actions.hsr_pickup.planning.log.warn",lambda *args:output.append(args))
    f=SimpleNamespace(event="rejected",mode="arm_only",candidate_id="arm_only-0",operation="insertion",reason="incomplete_cartesian_path",fraction=.12,elapsed_s=.4,contacts=[SimpleNamespace(first="arm_flex_link",second="<octomap>",depth=.001)])
    PickupPlanningClient.feedback(f)
    assert "arm_flex_link" in str(output) and "incomplete_cartesian_path" in str(output)


def test_yaw_request_feedback_is_visible_as_info(monkeypatch):
    output=[]
    monkeypatch.setattr("src.plugin.actions.hsr_pickup.planning.log.info",lambda *args:output.append(args))
    monkeypatch.setattr("src.plugin.actions.hsr_pickup.planning.log.warn",lambda *args:pytest.fail("request diagnostics are not failures"))
    f=SimpleNamespace(event="yaw_constraints",mode="base_only",candidate_id="base_only-0",operation="stage_base",reason='{"start_raw":3.1,"goal_raw":-3.1}',fraction=0.,elapsed_s=.1,contacts=[])
    PickupPlanningClient.feedback(f)
    assert "start_raw" in str(output) and "stage_base" in str(output)


@pytest.mark.parametrize("scale",[0.,.5,float("inf"),float("nan")])
def test_invalid_time_scale_rejected(scale):
    with pytest.raises(ValueError,match="trajectory_time_scale"):
        PickupBackendSettings(trajectory_time_scale=scale)


def test_relative_lift_uses_current_measured_state_and_cancellation(monkeypatch):
    from src.plugin.actions.manipulation import _MoveJointsRelativeAction
    context=Context(arm=Mock());context.arm.get_status.return_value="succeeded"
    monkeypatch.setattr("src.plugin.actions.manipulation._latest_joint_positions",lambda ctx:{"arm_lift_joint":.41})
    jobs=[];context.run_in_context_loop=lambda work,**cb:jobs.append((work,cb))
    action=_MoveJointsRelativeAction({"arm_lift_joint":.05});action.start(context);work,cb=jobs[-1];result=work();cb['callback'](result)
    assert context.arm.send_joint_goal.call_args.args[0]=={"arm_lift_joint":pytest.approx(.46)}
    action.start(context);action.cancel(context);work,cb=jobs[-1];work()
    assert context.arm.send_joint_goal.call_count==1


@pytest.mark.parametrize('failure',[None,'closure','attachment','planning','planning_go','staging','opening','insertion','lift','replan_retreat','retreat','undock','go','escape'])
@pytest.mark.parametrize('allow_target_octomap',[True,False])
@pytest.mark.parametrize('surround',['present','absent','disabled'])
def test_pickup_operation_order_and_postclose_preservation(monkeypatch,failure,allow_target_octomap,surround):
    from tests.fixtures.native_tree import tick_action
    from src.plugin.actions import manipulation
    events=[];arm=Mock();gripper=Mock()
    arm.get_status.return_value=gripper.get_status.return_value='succeeded'
    arm.add_collision_mesh.side_effect=lambda *a,**k:events.append('surround' if k['name'].endswith('_surround') else 'mesh') or True
    arm.allow_collision_pairs.side_effect=lambda *a,**k:events.append('policy') or True
    arm.attach_collision_object.side_effect=lambda *a,**k:events.append('attach') or failure!='attachment'
    def motion(operation):
        events.append(operation)
        arm.get_status.return_value='failed' if failure==operation or (failure=='escape' and operation=='fallback_undock') or (failure=='planning_go' and operation=='go') else 'succeeded'
    from itertools import count
    dispatches=count()
    def prepared(*args,**kwargs):
        index=next(dispatches)
        operation={3:'retreat',4:'undock'}.get(index,'prepared')
        if (failure=='staging' and index==0) or (failure=='insertion' and index==2):operation=failure
        motion(operation)
    arm.send_synchronized_trajectory_goal.side_effect=prepared
    arm.get_error.return_value='motion failed'
    arm.get_failure.return_value=None
    arm.send_joint_goal.side_effect=lambda joints:motion('fallback_undock' if 'odom_x' in joints else 'lift')
    arm.send_base_pose_goal.side_effect=lambda *a,**k:motion('undock')
    arm.send_named_goal.side_effect=lambda *a,**k:motion('fallback_go' if k.get('force_execution') else 'go')
    arm.clear_collision_scene.side_effect=lambda:events.append('clear')
    gripper.open.side_effect=lambda *a,**k:events.append('open') or SimpleNamespace(success=failure!='opening',error='opening failed')
    gripper.close.side_effect=lambda *a,**k:events.append('close') or SimpleNamespace(success=failure!='closure',error='jammed')
    gripper.get_state.return_value='closed';gripper.get_distance.return_value=.04
    context=Context(arm=arm,gripper=gripper)
    context.sensor_data_center=Mock()
    context.sensor_data_center.get_robot_pose_global.return_value={'x':9.,'y':-2.,'yaw':1.2}
    context.hardware['hsr_pickup']={'allow_held_target_octomap_contact':allow_target_octomap}
    def synchronous(work,callback=None,error_callback=None):
        try:
            result=work()
            if callback:callback(result)
        except Exception as exc:
            if error_callback:error_callback(exc)
            else:raise
    context.run_in_context_loop=synchronous
    monkeypatch.setattr(manipulation,'_create_mesh_for_collision_world',lambda *a,**k:'/tmp/mock-mesh')
    monkeypatch.setattr(manipulation,'_latest_joint_positions',lambda ctx:{'arm_lift_joint':.4})
    def clear_octomap(action,ctx):
        events.append('clear_octomap')
        action._done=True
    monkeypatch.setattr(manipulation._ClearOctomapAction,'start',clear_octomap)
    plan=dummy_plan();t=plan.motions[0]
    motions=tuple(replace(t,name=n,hand_path=((.1,.1,.7),(.5,.4,.7)) if n=='insertion' else ()) for n in ('stage_base_first','stage_arm_second','insertion','lift','retreat','undock','go'))
    plan=replace(plan,motions=() if failure in ('planning','planning_go') else motions,diagnostics={'reason':'no_complete_pickup' if failure in ('planning','planning_go') else 'pickup_planned'})
    client=Mock();client.collect_inputs.return_value=plan.inputs
    client.plan.side_effect=lambda *a,**k:events.append('plan') or plan
    def replan_retreat(*args,**kwargs):
        events.append('replan_retreat')
        if failure in ('replan_retreat','escape'):raise RuntimeError('retreat planning failed')
        return replace(plan,motions=tuple(m for m in motions if m.name in ('retreat','undock')))
    client.plan_post_lift.side_effect=replan_retreat
    monkeypatch.setattr('src.plugin.actions.hsr_pickup.PickupPlanningClient',Mock(return_value=client))
    context.world_state.set('jar_estimated_pick_pose_mat',plan.inputs.grasp);context.world_state.set('jar_object_pcd',plan.inputs.target_points);context.world_state.set('jar_pcd_frame','odom')
    context.world_state.set('jar_held_state','commanded')  # Must not block another invocation.
    if surround!='absent':context.world_state.set('jar_surround_pcd',plan.inputs.target_points+1.)
    has_surround=surround=='present'
    action=HsrPickup('jar',include_surround_collision=surround!='disabled')
    for _ in range(40):
        result=tick_action(action,context)
        if result.status!='running':break
    prefix=['clear','mesh']+(['surround'] if has_surround else [])+['clear_octomap','plan']
    assert events[:len(prefix)]==prefix
    assert events[-2:]==['clear','clear_octomap']
    if failure is None:
        assert result.status=='success',result
        undock=next(m for m in motions if m.name=='undock')
        pose=context.world_state.get('jar_pickup_plan.undock_pose')
        assert tuple(pose[n] for n in ('x','y','yaw'))==tuple(undock.end_joints[n] for n in ('odom_x','odom_y','odom_t'))
        assert pose['frame_id']==plan.inputs.frame_id
        arm.send_base_pose_goal.assert_not_called()
        assert set(arm.send_synchronized_trajectory_goal.call_args.args[0].joint_names)=={'odom_x','odom_y','odom_t'}
        assert events==prefix+['prepared','prepared','open','policy']+(['policy'] if has_surround else [])+['prepared','close','attach']+(['policy'] if has_surround else [])+['policy','lift','replan_retreat','retreat','undock','go','clear','clear_octomap']
        policies=[call.args[0] for call in arm.allow_collision_pairs.call_args_list]
        links=PickupBackendSettings.from_hardware(context.hardware).touch_links
        assert set(policies[0])=={(link,obj) for link in links for obj in ('orchid_pickup_jar','<octomap>')}
        expected={(link,obj) for link in links for obj in ('orchid_pickup_jar','<octomap>')}
        if allow_target_octomap:expected.add(('orchid_pickup_jar','<octomap>'))
        assert set(policies[-1])==expected
        if has_surround:
            assert set(policies[1])=={(link,'orchid_pickup_jar_surround') for link in links}
            assert set(policies[2])=={('orchid_pickup_jar','orchid_pickup_jar_surround')}|{(link,'orchid_pickup_jar_surround') for link in links}
    elif failure in ('planning','planning_go'):
        assert result.status=='failure' and result.reason=='no_complete_pickup'
        assert events==prefix+['go','clear','clear_octomap']
        arm.send_base_pose_goal.assert_not_called()
        assert not arm.send_named_goal.call_args.kwargs['force_execution']
    elif failure in ('staging','opening','insertion'):
        assert result.status=='failure'
        assert not any(e in events for e in ('close','fallback_undock','fallback_go'))
        assert events.count('go')==(0 if failure=='insertion' else 1)
    else:
        assert result.status=='failure' and not result.force_replan
        if failure=='escape':
            assert events[-3:]==['fallback_undock','clear','clear_octomap']
            assert 'fallback_go' not in events
        else:assert events[-4:]==['fallback_undock','fallback_go','clear','clear_octomap']
        goal=next(c.args[0] for c in arm.send_joint_goal.call_args_list if 'odom_x' in c.args[0])
        assert goal==pytest.approx({'odom_x':9.-.30*np.cos(1.2),'odom_y':-2.-.30*np.sin(1.2),'odom_t':1.2})
        if failure!='escape':assert arm.send_named_goal.call_args.kwargs['force_execution'] is True
        arm.send_base_pose_goal.assert_not_called()
        if failure in ('replan_retreat','escape'):assert result.reason=='retreat planning failed'
        assert events.count('close')==(2 if failure=='closure' else 1)
        if failure in ('closure','attachment'):assert not any(e in events for e in ('lift','retreat','undock','go'))
        if failure=='lift':assert not any(e in events for e in ('retreat','undock','go'))
        if failure=='retreat':assert not any(e in events for e in ('undock','go'))
        if failure=='undock':assert 'go' not in events
        assert context.world_state.get('jar_held_state')==('uncertain' if failure=='closure' else 'commanded')


def test_upward_lift_never_clamps_to_a_downward_goal(monkeypatch):
    from src.plugin.actions.manipulation import _MoveJointsRelativeAction
    context=Context(arm=Mock())
    monkeypatch.setattr('src.plugin.actions.manipulation._latest_joint_positions',lambda ctx:{'arm_lift_joint':.70})
    def run(work,callback=None,**kwargs):
        result=work()
        if callback:callback(result)
    context.run_in_context_loop=run
    action=_MoveJointsRelativeAction({'arm_lift_joint':.05});action.start(context)
    assert action.check_result(context).status=='failure'
    context.arm.send_joint_goal.assert_not_called()


def test_explicit_undock_pose_is_used_without_pick_start_fallback():
    from src.plugin.actions.navigation import MoveBaseWithWholeBody
    context=Context(arm=Mock())
    action=MoveBaseWithWholeBody(pose_key='jar_pickup_plan.undock_pose')
    context.world_state.set('last_pick_start_base_pose',{'x':.9,'y':0.,'yaw':0.,'frame_id':'odom'})
    with pytest.raises(ValueError,match='base goal missing'):
        action._resolve_base_goal(context)
    context.world_state.set(action.pose_key,{'x':.4,'y':.2,'yaw':4.,'frame_id':'map'})
    goal=action._resolve_base_goal(context)
    assert (goal.x,goal.y,goal.yaw,goal.frame_id)==(.4,.2,4.,'map')


@pytest.mark.parametrize('geometry',['mesh','convex_hull'])
def test_target_geometry_selection_is_explicit_and_shared(geometry):
    context=Context(arm=Mock(),gripper=Mock())
    action=HsrPickup('jar',target_collision_geometry=geometry)
    tree=action.build_node(context,'pick')
    assert action.param_dict()['target_collision_geometry']==geometry
    assert action._mesh.convex_hull==(geometry=='convex_hull')
    assert action._mesh.required
    planner=next(node.action for node in tree.iterate()
                 if isinstance(getattr(node,'action',None),PlanHsrPickup))
    assert planner.target_collision_geometry==geometry
    assert HsrPickup('jar').target_collision_geometry=='mesh'
    assert PickupRequest('jar').target_collision_geometry=='mesh'


@pytest.mark.parametrize('geometry',['mesh','convex_hull'])
def test_planning_geometry_matches_live_mesh_before_frame_transform(geometry):
    import open3d as o3d
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from moveit_msgs.msg import PlanningScene
    from src.core.sensor_data_center import SensorDataCenter
    from src.plugin.actions.hsr_pickup.model import ARM_JOINTS,PickupObservation
    dense=np.stack(np.meshgrid(*([np.linspace(-.035,.035,13)]*3)),axis=-1).reshape(-1,3)+[.6,.1,.75]
    grasp,_=geometry_inputs()
    observation=PickupObservation(grasp,dense,np.empty((0,3)),'map','+z')
    context=Context(arm=Mock())
    context.hardware['arm']={'named_targets':{'go':{'joints':dict.fromkeys(ARM_JOINTS,0.)}}}
    rotation=Rotation.from_euler('z',.4)
    quaternion=rotation.as_quat()
    context.sensor_data_center=SimpleNamespace(get_pose_at=Mock(return_value={
        'position':dict(zip(('x','y','z'),(.2,-.3,.1))),
        'orientation':dict(zip(('x','y','z','w'),quaternion))}))
    scene=PlanningScene();scene.robot_state.joint_state.name=list(PICKUP_JOINTS)
    scene.robot_state.joint_state.position=[0.]*len(PICKUP_JOINTS)
    client=PickupPlanningClient(None,PickupBackendSettings())
    client._call=Mock(side_effect=[
        SimpleNamespace(values=[SimpleNamespace(string_value='<robot name="hsrc"><link name="odom"/></robot>'),
                                SimpleNamespace(string_value='<robot name="hsrc"/>')]),
        SimpleNamespace(result=SimpleNamespace(names=[])),SimpleNamespace(scene=scene)])
    inputs=client.collect_inputs(context,'jar',2.,observation,target_collision_geometry=geometry)
    sdc=SensorDataCenter.__new__(SensorDataCenter)
    path=sdc.pcd_to_mesh(dense,convex_hull=geometry=='convex_hull',inflation_factor=2.)
    try:
        live=o3d.io.read_triangle_mesh(path)
        live_vertices=rotation.apply(np.asarray(live.vertices))+[.2,-.3,.1]
        planned=inputs.target.meshes[0]
        vertices=np.array([[p.x,p.y,p.z] for p in planned.vertices])
        # STL stores float32 and may reorder shared vertices.
        for actual in vertices:
            assert np.min(np.linalg.norm(live_vertices-actual,axis=1))<2e-6
        assert len(planned.triangles)==len(live.triangles)
        assert inputs.target.header.frame_id=='odom'
        assert inputs.target.id=='orchid_pickup_jar'
        context.sensor_data_center.get_pose_at.assert_called_once_with('map','odom')
        if geometry=='mesh':
            assert np.max(np.ptp(np.asarray(live.vertices),axis=0))<.12  # Factor 2 is hull-only.
    finally:
        Path(path).unlink()
