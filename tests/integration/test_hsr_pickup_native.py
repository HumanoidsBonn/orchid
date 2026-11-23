"""Native ROS action planning against captured models. No controller/live-scene calls.

Run on an isolated ROS_DOMAIN_ID with ORCHID_PICKUP_SNAPSHOT set. Synthetic scenes
below are feasibility fixtures, not claims about the live robot environment.
"""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from dataclasses import replace
import numpy as np
import pytest
from src.plugin.actions.hsr_pickup.model import ARM_JOINTS,BASE_JOINTS,PickupBackendSettings,PickupRequest
from src.plugin.actions.hsr_pickup.planning import PickupPlanningClient
from src.plugin.actions.hsr_pickup.diagnostics import load_inputs

SOURCE=os.environ.get("ORCHID_PICKUP_SNAPSHOT")
pytestmark=pytest.mark.skipif(not SOURCE,reason="requires captured model and built ROS packages")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    assert os.environ.get("ROS_DOMAIN_ID") not in (None,"5"),"native test server must use an isolated ROS domain"
    log=(tmp_path_factory.mktemp('native-server')/'server.log').open('w')
    process=subprocess.Popen(['ros2','run','orchid_hsr_pickup','pickup_planning_server'],stdout=log,stderr=log,start_new_session=True)
    node=rclpy.create_node('pickup_offline_test');executor=MultiThreadedExecutor();executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    c=PickupPlanningClient(node,PickupBackendSettings(planning_time_s=2.,timeout_s=60.))
    try:yield c
    finally:
        os.killpg(process.pid,signal.SIGINT);process.wait(timeout=10);log.close();executor.shutdown();thread.join(timeout=3);node.destroy_node()


@pytest.fixture(scope="module")
def inputs():
    from moveit_msgs.msg import PlanningSceneWorld,CollisionObject
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    # A known FK-derived pose from the previous native fixture, retained offline.
    fixture=None
    for p in Path(SOURCE).glob('*/*.json'):
        if p.stem not in ('base_only','arm_only','coordinated'):continue
        d=json.loads(p.read_text())
        if 'fixture' in d:fixture=d['fixture'];break
    assert fixture is not None,'capture has no known FK fixture'
    go={n:fixture['joints'][n] for n in ARM_JOINTS}
    data=load_inputs(SOURCE,go);pose=np.array(fixture['grasp']);direction=pose[:3,2]
    scene=copy.deepcopy(data.scene);scene.world=PlanningSceneWorld();scene.robot_state.attached_collision_objects=[]
    joints=dict(fixture['joints']);joints['odom_x']-=.06*direction[0];joints['odom_y']-=.06*direction[1]
    scene.robot_state.joint_state.name=list(joints);scene.robot_state.joint_state.position=list(joints.values())
    centre=pose[:3,3]+direction*.06
    target=CollisionObject(id='orchid_pickup_fixture',operation=CollisionObject.ADD)
    target.header.frame_id=data.frame_id
    primitive=SolidPrimitive(type=SolidPrimitive.BOX,dimensions=[.01,.01,.01]);location=Pose();location.orientation.w=1.;location.position.x,location.position.y,location.position.z=map(float,centre)
    target.primitives=[primitive];target.primitive_poses=[location]
    # The server supports standard MoveIt objects; production supplies the hull.
    return replace(data,start_joints=joints,grasp=pose,scene=scene,target=target,go_joints=go,
        target_points=np.array([[x,y,z] for x in (-.005,.005) for y in (-.005,.005) for z in (-.005,.005)])+centre)


@pytest.fixture(scope="module")
def plans(client,inputs):
    return {mode:client.plan(inputs,PickupRequest('fixture',approach_mode=mode,pregrasp_distance_m=.06,min_insertion_m=.04,max_insertion_m=.06,clearance_m=.0,post_grasp_lift_m=.03)) for mode in ('base_only','arm_only','coordinated','auto')}


@pytest.mark.parametrize('mode',['base_only','arm_only','coordinated','auto'])
def test_complete_connected_pickup(client,inputs,plans,mode):
    plan=plans[mode]
    assert plan.motions,plan.candidates
    assert plan.selected_mode in ('base_only','arm_only','coordinated')
    if mode!='auto':assert plan.selected_mode==mode
    else:assert plan.selected_mode=='coordinated'  # Current configured auto priority.
    names=[m.name for m in plan.motions]
    assert names[-5:]==['insertion','lift','retreat','undock','go']
    assert plan.staging_order in ('base_first','arm_first','coordinated')
    for a,b in zip(plan.motions,plan.motions[1:]):
        assert max(abs(a.end_joints[n]-b.start_joints[n]) for n in (*BASE_JOINTS,*ARM_JOINTS))<2e-5
    lift=next(m for m in plan.motions if m.name=='lift');i=lift.trajectory.joint_names.index('arm_lift_joint');q=[p.positions[i] for p in lift.trajectory.points]
    assert all(b>=a-1e-7 for a,b in zip(q,q[1:]));assert q[-1]-q[0]==pytest.approx(.03,abs=1e-5)
    retreat=next(m for m in plan.motions if m.name=='retreat')
    lift=next(m for m in plan.motions if m.name=='lift')
    insertion=next(m for m in plan.motions if m.name=='insertion')
    assert np.allclose(retreat.hand_path[-1][:2],insertion.hand_path[0][:2],atol=.005)
    expected_z=insertion.hand_path[0][2]+lift.hand_path[-1][2]-insertion.hand_path[-1][2]
    assert retreat.hand_path[-1][2]==pytest.approx(expected_z,abs=.005)
    frozen=ARM_JOINTS+('odom_t',) if mode=='base_only' else BASE_JOINTS if mode=='arm_only' else ()
    for name in frozen:
        i=retreat.trajectory.joint_names.index(name)
        values=np.array([p.positions[i] for p in retreat.trajectory.points])
        assert np.max(np.abs(values-values[0]))<2e-6
    undock=next(m for m in plan.motions if m.name=='undock')
    i=undock.trajectory.joint_names.index('odom_t')
    yaw=np.array([p.positions[i] for p in undock.trajectory.points])
    assert np.max(np.abs(yaw-yaw[0]))<2e-6
    target=inputs.grasp[:2,3]
    insertion=next(m for m in plan.motions if m.name=='insertion')
    choices=[inputs.start_joints,insertion.start_joints]
    farther=max(choices,key=lambda q:np.linalg.norm(np.array([q[n] for n in BASE_JOINTS[:2]])-target))
    assert all(undock.end_joints[n]==pytest.approx(farther[n],abs=2e-5) for n in BASE_JOINTS[:2])
    assert all(abs(plan.motions[-1].end_joints[n]-inputs.go_joints[n])<=.100001 for n in ARM_JOINTS)
    frozen=ARM_JOINTS+('odom_t',) if plan.selected_mode=='base_only' else BASE_JOINTS if plan.selected_mode=='arm_only' else ()
    insertion=next(m for m in plan.motions if m.name=='insertion')
    for n in frozen:
        i=insertion.trajectory.joint_names.index(n);v=[p.positions[i] for p in insertion.trajectory.points];assert max(v)-min(v)<2e-6


@pytest.mark.parametrize('mode',['base_only','coordinated'])
def test_wrapped_yaw_pickup_and_arm_only_go(client,inputs,mode):
    """Rotate the known fixture to an unwrapped heading, reproducing the live bug."""
    from scipy.spatial.transform import Rotation
    yaw=4.-inputs.start_joints['odom_t']
    rotation=Rotation.from_euler('z',yaw)
    transform=np.eye(4);transform[:3,:3]=rotation.as_matrix()
    grasp=transform@inputs.grasp
    state=dict(inputs.start_joints)
    xy=rotation.apply([state['odom_x'],state['odom_y'],0.])
    state.update(odom_x=float(xy[0]),odom_y=float(xy[1]),odom_t=4.)
    scene=copy.deepcopy(inputs.scene)
    scene.robot_state.joint_state.name=list(state)
    scene.robot_state.joint_state.position=list(state.values())
    target=copy.deepcopy(inputs.target)
    for pose in target.primitive_poses:
        xyz=rotation.apply([pose.position.x,pose.position.y,pose.position.z])
        pose.position.x,pose.position.y,pose.position.z=map(float,xyz)
        q=pose.orientation
        quat=(rotation*Rotation.from_quat([q.x,q.y,q.z,q.w])).as_quat()
        q.x,q.y,q.z,q.w=map(float,quat)
    data=replace(inputs,start_joints=state,scene=scene,target=target,grasp=grasp,
                 target_points=rotation.apply(inputs.target_points))
    plan=client.plan(data,PickupRequest('fixture',approach_mode=mode,
        pregrasp_distance_m=.06,min_insertion_m=.04,max_insertion_m=.06,
        clearance_m=0.,post_grasp_lift_m=.03))
    assert plan.motions,plan.candidates
    retreat=next(m for m in plan.motions if m.name=='retreat')
    lift=next(m for m in plan.motions if m.name=='lift')
    insertion=next(m for m in plan.motions if m.name=='insertion')
    assert np.allclose(retreat.hand_path[-1][:2],insertion.hand_path[0][:2],atol=.005)
    expected_z=insertion.hand_path[0][2]+lift.hand_path[-1][2]-insertion.hand_path[-1][2]
    assert retreat.hand_path[-1][2]==pytest.approx(expected_z,abs=.005)
    frozen=ARM_JOINTS+('odom_t',) if mode=='base_only' else BASE_JOINTS if mode=='arm_only' else ()
    for name in frozen:
        i=retreat.trajectory.joint_names.index(name)
        values=np.array([p.positions[i] for p in retreat.trajectory.points])
        assert np.max(np.abs(values-values[0]))<2e-6
    undock=next(m for m in plan.motions if m.name=='undock')
    i=undock.trajectory.joint_names.index('odom_t')
    yaw=np.array([p.positions[i] for p in undock.trajectory.points])
    assert np.max(np.abs(yaw-yaw[0]))<2e-6
    insertion=next(m for m in plan.motions if m.name=='insertion')
    distance=lambda q:np.linalg.norm(np.array([q['odom_x'],q['odom_y']])-grasp[:2,3])
    assert distance(undock.end_joints)==pytest.approx(max(distance(state),distance(insertion.start_joints)),abs=1e-4)
    go=plan.motions[-1]
    for name in BASE_JOINTS:
        delta=np.array([p.positions[go.trajectory.joint_names.index(name)] for p in go.trajectory.points])-go.start_joints[name] if name in go.trajectory.joint_names else np.array([go.end_joints[name]-go.start_joints[name]])
        if name=='odom_t':delta=(delta+np.pi)%(2*np.pi)-np.pi
        assert np.max(np.abs(delta))<1e-6
    assert all(abs(go.end_joints[n]-inputs.go_joints[n])<=.100001 for n in ARM_JOINTS)


def test_unreachable_is_reported(client,inputs):
    pose=inputs.grasp.copy();pose[2,3]=100.
    plan=client.plan(replace(inputs,grasp=pose),PickupRequest('fixture',approach_mode='arm_only'))
    assert not plan.motions
    assert all(c['failure']['reason']=='pregrasp_no_ik' for c in plan.candidates)


def test_postclose_target_surround_contact_uses_grasp_alignment_policy(client,inputs):
    scene=copy.deepcopy(inputs.scene)
    # World/world overlap becomes a collision when the target is attached.
    surrounding=copy.deepcopy(inputs.target)
    surrounding.id=inputs.target.id+'_surround'
    scene.world.collision_objects.append(surrounding)
    plan=client.plan(replace(inputs,scene=scene),PickupRequest('fixture',approach_mode='base_only',
        pregrasp_distance_m=.06,min_insertion_m=.04,max_insertion_m=.06,
        clearance_m=0.,post_grasp_lift_m=.03))
    assert plan.motions,plan.candidates
    assert not scene.robot_state.attached_collision_objects


@pytest.mark.parametrize('environment_id',['<octomap>','surround'])
def test_insertion_environment_contact_policy_is_identical(client,inputs,environment_id):
    from geometry_msgs.msg import Pose
    from shape_msgs.msg import SolidPrimitive
    from moveit_msgs.msg import CollisionObject
    scene=copy.deepcopy(inputs.scene)
    environment=CollisionObject(id='<octomap>' if environment_id=='<octomap>' else inputs.target.id+'_surround',operation=CollisionObject.ADD)
    environment.header.frame_id=inputs.frame_id
    environment.primitives=[SolidPrimitive(type=SolidPrimitive.BOX,dimensions=[.015,.015,.015])]
    pose=Pose();pose.orientation.w=1.
    pose.position.x,pose.position.y,pose.position.z=map(float,inputs.grasp[:3,3])
    environment.primitive_poses=[pose];scene.world.collision_objects.append(environment)
    # Start far enough back that the endpoint obstruction is not a pregrasp
    # collision. This box exercises the ACM object identity, not voxel encoding.
    start=dict(inputs.start_joints);direction=inputs.grasp[:3,2]
    start['odom_x']-=.19*direction[0];start['odom_y']-=.19*direction[1]
    scene.robot_state.joint_state.name=list(start);scene.robot_state.joint_state.position=list(start.values())
    data=replace(inputs,scene=scene,start_joints=start)
    request=PickupRequest('fixture',approach_mode='base_only',pregrasp_distance_m=.25,
        min_insertion_m=.25,max_insertion_m=.25,clearance_m=0.,post_grasp_lift_m=.03)
    blocked_client=PickupPlanningClient(client.node,replace(client.settings,touch_links=()))
    blocked=blocked_client.plan(data,request)
    assert not blocked.motions
    permitted=client.plan(data,request)
    assert permitted.motions,permitted.candidates


def test_client_cancellation_reaches_native_action(client,inputs):
    from src.core.failure import MotionFailure
    calls=0
    def cancelled():
        nonlocal calls
        calls+=1
        return calls>2
    with pytest.raises(MotionFailure,match='cancelled'):
        client.plan(inputs,PickupRequest('fixture',approach_mode='auto'),cancelled)
