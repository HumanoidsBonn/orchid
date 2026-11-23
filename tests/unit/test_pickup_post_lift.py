"""Measured-state lift/retreat regressions; no robot or live service calls."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PlanningScene, AttachedCollisionObject

from src.core.context import Context
from src.plugin.actions.hsr_pickup import PlanHsrPickupRetreat
from src.plugin.actions.hsr_pickup.planning import PickupPlanningClient
from src.plugin.actions.hsr_pickup.model import PickupBackendSettings
from tests.unit.test_hsr_pickup import dummy_plan


def test_retreat_uses_measured_scene_and_achieved_lift_height(monkeypatch):
    previous = dummy_plan()
    motion = previous.motions[0]
    insertion = replace(motion,name="insertion",hand_path=((.4,.1,.8),(.6,.1,.8)))
    undock = replace(motion,name="undock")
    preview = replace(motion,name="retreat")
    previous = replace(previous,motions=(insertion,preview,undock))
    scene = PlanningScene()
    measured = dict(previous.inputs.start_joints,arm_lift_joint=.47)
    scene.robot_state.joint_state.name = list(measured)
    scene.robot_state.joint_state.position = list(measured.values())
    attachment=AttachedCollisionObject(link_name="hand_palm_link")
    attachment.object.id="orchid_pickup_jar"
    scene.robot_state.attached_collision_objects.append(attachment)
    pose = PoseStamped()
    pose.pose.position.z = .863
    pose.pose.orientation.w = 1.
    original = PoseStamped()
    original.pose.orientation.z = .6
    original.pose.orientation.w = .8
    client = PickupPlanningClient(Mock(),PickupBackendSettings())
    client._call = Mock(side_effect=[SimpleNamespace(scene=scene),SimpleNamespace(error_code=SimpleNamespace(val=1),pose_stamped=[pose]),SimpleNamespace(error_code=SimpleNamespace(val=1),pose_stamped=[original])])
    client.plan = Mock(return_value="planned")
    monkeypatch.setattr("src.plugin.actions.manipulation._latest_joint_positions",lambda ctx:measured)
    assert client.plan_post_lift(Context(),previous)=="planned"
    inputs,request,cancelled = client.plan.call_args.args
    retreat,goal = client.plan.call_args.kwargs["post_lift"]
    assert inputs.scene is scene
    assert inputs.scene.robot_state.attached_collision_objects[0].object.id=="orchid_pickup_jar"
    assert inputs.start_joints["arm_lift_joint"]==.47
    assert np.allclose(retreat[:3,3],[.4,.1,.863])
    from scipy.spatial.transform import Rotation
    assert np.allclose(retreat[:3,:3],Rotation.from_quat([0,0,.6,.8]).as_matrix())
    assert request.approach_mode==previous.selected_mode
    assert goal==undock.end_joints
    assert inputs.scene.robot_state.joint_state.position[list(inputs.scene.robot_state.joint_state.name).index("arm_lift_joint")]==.47


@pytest.mark.parametrize("cancel",[False,True])
@pytest.mark.parametrize("retreat_mode",["arm_only","arm_yaw","coordinated"])
def test_post_lift_leaf_publishes_only_current_invocation(monkeypatch,cancel,retreat_mode):
    context=Context()
    previous=dummy_plan()
    context.world_state.set("plan",previous)
    jobs=[]
    context.run_in_context_loop=lambda work,**callbacks:jobs.append((work,callbacks))
    client=Mock()
    client.plan_post_lift.return_value=replace(previous,selected_mode=retreat_mode)
    monkeypatch.setattr("src.plugin.actions.hsr_pickup.PickupPlanningClient",Mock(return_value=client))
    action=PlanHsrPickupRetreat("plan")
    action.start(context)
    work,callbacks=jobs[0]
    result=work()
    if cancel:action.cancel(context)
    callbacks["callback"](result)
    key="plan."+previous.motions[0].name
    if cancel:
        assert context.world_state.get(key) is None
        client.publish.assert_not_called()
    else:
        assert context.world_state.get(key) is previous.motions[0].trajectory
        assert action.check_result(context).status=="success"
        assert context.world_state.get("plan.retreat_mode")==retreat_mode
        assert context.world_state.get("plan") is previous
