"""Direct base escape contracts; all controller/TF calls are mocked."""
import math
from unittest.mock import Mock

import pytest
from src.core.context import Context
from src.plugin.actions.navigation import MoveBaseWithWholeBody


def setup():
    arm=Mock();arm.get_status.return_value='succeeded'
    sensor=Mock();sensor.get_robot_pose_global.return_value={'x':4.,'y':7.,'yaw':math.pi/2}
    context=Context(arm=arm,sensor_data_center=sensor)
    jobs=[]
    context.run_in_context_loop=lambda work,**callbacks:jobs.append((work,callbacks))
    return context,jobs


def complete(job):
    work,callbacks=job
    try:result=work()
    except Exception as exc:callbacks['error_callback'](exc)
    else:callbacks['callback'](result)


def test_direct_escape_uses_current_odom_and_world_direction_with_fixed_yaw():
    context,jobs=setup()
    action=MoveBaseWithWholeBody(relative_x=-.24,relative_y=-.18,relative_yaw=0.,plan_motion=False,relative_in_base_frame=False)
    context.world_state.set('last_pick_start_base_pose',{'x':-999,'y':-999,'yaw':0})
    action.start(context)
    # Bind the position at actual dispatch, not at mission construction/start.
    context.sensor_data_center.get_robot_pose_global.return_value={'x':9.,'y':-2.,'yaw':1.2}
    complete(jobs[-1])
    context.arm.send_joint_goal.assert_called_once_with({'odom_x':8.76,'odom_y':-2.18,'odom_t':1.2})
    context.arm.send_base_pose_goal.assert_not_called()
    assert action.check_result(context).status=='success'
    context.sensor_data_center.get_robot_pose_global.assert_called_with(global_frame='odom')


@pytest.mark.parametrize('cancel',[True,False])
def test_cancellation_or_execution_control_prevents_escape_dispatch(cancel):
    context,jobs=setup()
    action=MoveBaseWithWholeBody(plan_motion=False)
    action.start(context)
    if cancel:action.cancel(context)
    else:context.execution.control=object()
    complete(jobs[-1])
    context.arm.send_joint_goal.assert_not_called()
    context.arm.send_base_pose_goal.assert_not_called()


def test_direct_base_requires_odom_coordinates():
    context,jobs=setup()
    action=MoveBaseWithWholeBody(frame_id='map',plan_motion=False)
    action.start(context);complete(jobs[-1])
    assert action.check_result(context).status=='failure'
    context.arm.send_joint_goal.assert_not_called()


@pytest.mark.parametrize('mode',['base_backwards','reverse_insertion'])
def test_pickup_escape_modes_hold_yaw_and_use_current_pose(mode):
    from dataclasses import replace
    from src.plugin.actions.hsr_pickup import HsrPickup
    from tests.unit.test_hsr_pickup import dummy_plan
    context,jobs=setup()
    context.gripper=Mock()
    pickup=HsrPickup('jar',failure_undock_mode=mode)
    root=pickup.build_node(context,'Pick')
    plan=dummy_plan();motion=replace(plan.motions[0],hand_path=((0.,0.,.7),(.4,.3,.7)))
    context.world_state.set('jar_pickup_plan',replace(plan,motions=(motion,)))
    recovery=next(n for n in root.iterate() if n.name=='Post-insertion failure escape')
    recovery.initialise()
    assert recovery._entry_failure is None
    leaf=next(n for n in recovery.iterate() if isinstance(getattr(n,'action',None),MoveBaseWithWholeBody))
    leaf.action.start(context);complete(jobs[-1])
    expected=(4.,6.7) if mode=='base_backwards' else (3.76,6.82)
    assert context.arm.send_joint_goal.call_args.args[0]==pytest.approx({'odom_x':expected[0],'odom_y':expected[1],'odom_t':math.pi/2})
    context.arm.send_base_pose_goal.assert_not_called()


def test_default_escape_mode_and_invalid_input():
    from src.plugin.actions.hsr_pickup import HsrPickup
    assert HsrPickup('jar').failure_undock_mode=='base_backwards'
    with pytest.raises(ValueError,match='failure_undock_mode'):
        HsrPickup('jar',failure_undock_mode='unknown')
