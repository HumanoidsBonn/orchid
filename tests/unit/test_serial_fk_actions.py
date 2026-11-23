"""Focused native-action/condition and serial FK regressions; no pose estimation."""
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import threading

import numpy as np
import py_trees
import pytest

from src.core.action import ActionResult
from src.core.context import Context
from src.core.manifest import RobotManifest
from src.core.planners import PlannerResult
from src.core.world_state import WorldState
from src.core.modules.ros2_service_module import ROS2ServiceModule
from src.orchestrator.manifest_listener import load_manifest_from_file, Ros2ManifestListener
from src.plugin.conditions import WorldStateExists
from src.plugin.actions import grasp_alignment as ga
from src.tree.bt_node_wrapper import BtActionLeaf, BtConditionLeaf
from src.tree.builder import TreeBuilder
from src.tree.plan import ConditionNode, Plan


def hand(z=.3, reach=.55):
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, 1, 0]]
    pose[0, 3], pose[2, 3] = reach, z
    return pose


class Planner:
    def __init__(self, invalid_lifts=()):
        self.fk_calls = self.validity_calls = 0
        self.invalid_lifts = invalid_lifts
        self.states = []

    def compute_fk(self, state, **kwargs):
        self.fk_calls += 1
        joints = dict(zip(state.name, state.position))
        return PlannerResult(success=True, metadata={"pose": hand(.3 + joints['arm_lift_joint']).tolist()})

    def check_state_validity(self, state, **kwargs):
        self.validity_calls += 1
        joints = dict(zip(state.name, state.position))
        self.states.append(joints)
        valid = joints['arm_lift_joint'] not in self.invalid_lifts
        return PlannerResult(success=valid, metadata={"valid": valid, "contacts": [] if valid else ['table']})


@pytest.fixture
def fk_context(monkeypatch):
    monkeypatch.setattr(ga, '_joint_state_msg', lambda joints: SimpleNamespace(name=list(joints), position=list(joints.values())))
    ws = WorldState()
    ws.set('mug_pregrasp_pose', hand(.3, .9))
    ws.set('mug_grasp_pose', hand(.3, 1.0))
    ws.set('mug_grasp_motion_metadata', {'grasp_motion_type': 'horizontal', 'approach_axis': '+z',
        'scene': {'min_reach_m': .25, 'max_reach_m': .85}})
    return Context(world_state=ws, planners={'manipulation': Planner()}, sensor_data_center=SimpleNamespace(
        get_pose_at=lambda source, target: {'mat': np.eye(4)}, peek_latest=lambda *a, **k: None))


def action(**kwargs):
    return ga.SelectFkArmPosture(target='mug', execute=False, debug_visualization=False,
        sampling_mode='simple', arm_lift_samples=1, arm_flex_samples=1, **kwargs)


def inline(context):
    def run(work, *args, callback=None, error_callback=None, **kwargs):
        try:
            value = work(*args, **kwargs)
        except Exception as exc:
            error_callback(exc)
        else:
            callback(value)
    context.run_in_context_loop = run


def test_condition_json_validation_and_runtime_binding():
    context = Context(supported_conditions={'world_state_exists': WorldStateExists})
    tree = TreeBuilder(context).build(Plan(root=ConditionNode(name='world_state_exists', params={'key': 'x'})))
    tree.tick()
    assert tree.root.status == py_trees.common.Status.FAILURE
    context.world_state.set('x', False)
    tree.tick()
    assert tree.root.status == py_trees.common.Status.SUCCESS
    with pytest.raises(ValueError, match='Unknown inputs'):
        TreeBuilder(context).build(Plan(root=ConditionNode(name='world_state_exists', params={'key': 'x', 'junk': 1})))
    with pytest.raises(RuntimeError, match='Condition'):
        context.require_condition_type('action_name')


@pytest.mark.parametrize('predicate,diagnostic', [(lambda c: 1, 'TypeError'), (lambda c: 1/0, 'ZeroDivisionError')])
def test_condition_failure_diagnostics(predicate, diagnostic):
    node = BtConditionLeaf(predicate, Context(), name='Check')
    list(node.tick())
    assert node.status == py_trees.common.Status.FAILURE
    assert diagnostic in node.feedback_message


def test_manifest_condition_loaders(tmp_path):
    data = {'robot_id': 'test', 'robot_type': 'test', 'conditions': ['world_state_exists']}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(data))
    assert load_manifest_from_file(path).conditions == ['world_state_exists']
    listener = object.__new__(Ros2ManifestListener)
    assert listener._parse_manifest_data(data).conditions == ['world_state_exists']
    assert RobotManifest('test', 'test').conditions == []


def test_condition_session_wiring_and_unknown_declaration(monkeypatch):
    from src.orchestrator.session import RobotSession
    from src.plugin.registry import PluginRegistry
    registry = PluginRegistry()
    registry.register_condition('world_state_exists', WorldStateExists)
    session = object.__new__(RobotSession)
    session.supported_actions, session.unsupported_actions = {}, []
    session.supported_controllers, session.unsupported_controllers = {}, []
    session.supported_planners, session.unsupported_planners = {}, []
    session.ros2_wrapper, session.action_history_length = None, 5
    monkeypatch.setattr(session, '_load_scene_graph', lambda: None)
    context = session._build_context(RobotManifest('test', 'test', conditions=['world_state_exists']), registry)
    assert context.require_condition_type('world_state_exists') is WorldStateExists
    with pytest.raises(ValueError, match='Unknown manifest condition'):
        session._build_context(RobotManifest('test', 'test', conditions=['missing']), registry)


def test_condition_prompt_exposes_enabled_names():
    from src.core.llm_planner import LLMPlanner
    planner = object.__new__(LLMPlanner)
    planner.context = Context(supported_conditions={'world_state_exists': WorldStateExists})
    assert 'world_state_exists' in planner._build_user_prompt('check x')


def test_disabled_validity_makes_zero_calls(fk_context):
    best = action(check_state_validity=False, enforce_state_validity=False)._select(fk_context)
    assert fk_context.planners['manipulation'].validity_calls == 0
    assert best['gate']['allowed']
    metadata = best['_publication']['mug_fk_posture_metadata']
    assert metadata['counts'] == {'fk_calls': 1, 'validity_calls': 0, 'state_snapshots': 1}
    assert metadata['timings']['fk_s'] >= 0
    with pytest.raises(ValueError, match='requires'):
        action(check_state_validity=False)


def test_diagnostic_validity_records_rejection_without_enforcement(fk_context):
    fk_context.planners['manipulation'].invalid_lifts = (0.0,)
    best = action(enforce_state_validity=False)._select(fk_context)
    assert best['validity']['valid'] is False
    assert best['validity']['contacts'] == ['table']


def test_enforced_unsupported_checker_fails(fk_context):
    planner = fk_context.planners['manipulation']
    fk_context.planners['manipulation'] = SimpleNamespace(compute_fk=planner.compute_fk)
    with pytest.raises(RuntimeError, match='no_valid_candidate') as caught:
        action()._select(fk_context)
    assert caught.value.diagnostics['rejected_candidates'][0]['validity']['error'] == 'unsupported'


@pytest.mark.parametrize('survivor', [None, .02])
def test_upward_offset_cannot_win_outside_z_gate(fk_context, monkeypatch, survivor):
    lifts = [0., .2] if survivor is None else [0., survivor, .2]
    monkeypatch.setattr(ga.SelectFkArmPosture, '_candidate_joint_sets', lambda *a, **k:
        [{'arm_lift_joint': lift, 'arm_flex_joint': -.4} for lift in lifts])
    fk_context.planners['manipulation'].invalid_lifts = (0.,)
    if survivor is None:
        with pytest.raises(RuntimeError, match='no_valid_candidate') as caught:
            action()._select(fk_context)
        rejects = caught.value.diagnostics['rejected_candidates']
        assert any('z_error' in candidate['gate']['reasons'] for candidate in rejects)
    else:
        best = action()._select(fk_context)
        assert best['joint_positions']['arm_lift_joint'] == survivor
        assert best['errors']['signed_z_error_m'] == pytest.approx(.02)
    # The high posture is rejected before a validity request.
    assert fk_context.planners['manipulation'].validity_calls == len(lifts)-1


def test_geometric_rejects_remain_refinement_anchors(fk_context, monkeypatch):
    seen = []
    monkeypatch.setattr(ga.SelectFkArmPosture, '_candidate_joint_sets', lambda *a, **k: [{'arm_lift_joint': .1}])
    def refine(self, anchors, **kwargs):
        seen.extend(anchors)
        return [{'arm_lift_joint': 0.}]
    monkeypatch.setattr(ga.SelectFkArmPosture, '_refined_joint_sets_from_candidates', refine)
    subject = action()
    subject.sampling_mode = 'multires_scene'
    assert subject._select(fk_context)['joint_positions']['arm_lift_joint'] == 0.
    assert seen[0]['gate']['reasons'] == ['z_error']


def test_cost_terms_are_nonnegative_and_shortfalls_do_not_cancel():
    errors = {'z_error_m': 0., 'lateral_error_m': 0., 'approach_error_rad': 0.}
    terms = ga.SelectFkArmPosture._score_terms(hand(), hand(), hand_pose_base=hand(reach=.2), errors=errors,
        joints={'arm_lift_joint': .1}, scene={'min_reach_m': .3}, preferred_lift=.4)
    assert all(value >= 0 for value in terms.values())
    assert terms['reach_shortfall'] > 0 and terms['reach_preference'] == 0
    assert terms['lift_shortfall'] > 0 and terms['lift_preference'] == 0
    assert ga._min_lift_for_reach(.85, {}, arm_lift_max=.55) > .35


def test_shorter_reach_preferred_with_equal_geometry():
    errors = {'z_error_m': 0., 'lateral_error_m': 0., 'approach_error_rad': 0.}
    score = lambda reach: ga.SelectFkArmPosture._score_hand_pose(hand(), hand(),
        hand_pose_base=hand(reach=reach), errors=errors, preferred_lift=0)
    assert score(.4) < score(.7)


class Sensors:
    def __init__(self):
        self.reads = 0
    def peek_latest(self, *args, **kwargs):
        self.reads += 1
        return SimpleNamespace(name=['head_pan_joint'], position=[.2])
    def get_pose_at(self, source, target):
        matrix = np.eye(4)
        if source == 'base_footprint':
            matrix[2, 3] = .05
        elif (source, target) == ('map', 'odom'):
            matrix[0, 3] = -2.
        else:
            return None
        return {'mat': matrix}


def test_state_snapshot_and_frame_height_and_validity_frame(fk_context):
    sensors = Sensors()
    fk_context.sensor_data_center = sensors
    fk_context.world_state.set('mug_pregrasp_pose', hand(.35, .9))
    fk_context.world_state.set('mug_grasp_pose', hand(.35, 1.0))
    best = action(world_frame='map')._select(fk_context)
    assert sensors.reads == 1
    assert best['errors']['signed_z_error_m'] == pytest.approx(0., abs=1e-8)
    state = fk_context.planners['manipulation'].states[0]
    assert state['head_pan_joint'] == .2
    assert state['odom_x'] == pytest.approx(best['final_grasp_base_pose']['x'] - 2)
    with pytest.raises(RuntimeError, match='missing_transform'):
        action(world_frame='unknown')._select(fk_context)


def test_top_down_lift_uses_fk_height_and_frame_offset():
    subject = action(arm_lift_min=.1, arm_lift_max=.55)
    seed_pose = np.eye(4)
    seed_pose[2, 3] = .3
    joints = subject._top_down_candidate_joint_sets(pregrasp_pose=hand(.5), wrist_bias=0,
        planner=object(), context=None, compute_fk=lambda j: PlannerResult(success=True, metadata={'pose': seed_pose.tolist()}), base_height=.05)
    assert all(j['arm_lift_joint'] == pytest.approx(.25) for j in joints)


def test_async_publication_and_cached_revalidation(fk_context):
    inline(fk_context)
    first = action()
    first.start(fk_context)
    assert first.check_result(fk_context).status == 'success'
    before = fk_context.planners['manipulation'].fk_calls
    second = action()
    second.execute_arm = True
    goals = []
    fk_context.arm = SimpleNamespace(send_joint_goal=lambda joints: goals.append(joints))
    second.start(fk_context)
    assert len(goals) == 1
    assert fk_context.planners['manipulation'].fk_calls == before+1
    assert fk_context.world_state.get('mug_fk_posture_metadata')['cache_reused'] is True
    fk_context.planners['manipulation'].invalid_lifts = (0.,)
    third = action()
    third.execute_arm = True
    third.start(fk_context)
    assert third.check_result(fk_context).status == 'failure'
    assert len(goals) == 1


def test_stale_async_callback_cannot_publish_or_dispatch(fk_context):
    queued = []
    fk_context.run_in_context_loop = lambda work, *args, **kwargs: queued.append((work, args, kwargs))
    subject = action()
    subject.start(fk_context)
    work, args, kwargs = queued[0]
    result = work(*args, cancelled=lambda: False)
    subject.cancel(fk_context)
    subject.start(fk_context)
    kwargs['callback'](result)
    assert fk_context.world_state.get('mug_fk_joint_goal') is None
    assert subject.check_result(fk_context).status == 'running'
    work, args, kwargs = queued[1]
    kwargs['callback'](work(*args, cancelled=kwargs['cancelled']))
    assert subject.check_result(fk_context).status == 'success'


def test_real_async_start_is_nonblocking_and_cancel_stops_next_call(fk_context, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    planner = fk_context.planners['manipulation']
    original = planner.compute_fk
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    planner.compute_fk = blocked
    original_run = fk_context.run_in_context_loop
    def tracked(work, *args, callback=None, error_callback=None, **kwargs):
        def complete(value):
            try:
                callback(value)
            finally:
                finished.set()
        def failed(exc):
            try:
                error_callback(exc)
            finally:
                finished.set()
        original_run(work, *args, callback=complete, error_callback=failed, **kwargs)
    fk_context.run_in_context_loop = tracked
    subject = action()
    subject.start(fk_context)
    assert entered.wait(3)
    assert subject.check_result(fk_context).status == 'running'
    subject.cancel(fk_context)
    release.set()
    assert finished.wait(3)
    fk_context._async_loop.call_soon_threadsafe(fk_context._async_loop.stop)
    fk_context._async_loop_thread.join(timeout=3)
    assert planner.fk_calls == 1
    assert planner.validity_calls == 0
    assert fk_context.world_state.get('mug_fk_joint_goal') is None


def test_service_client_reuse_and_module_owned_executor(monkeypatch):
    import rclpy
    monkeypatch.setattr(rclpy, 'ok', lambda: True)
    calls = []
    pending = []
    class Future:
        def __init__(self): self.ready = False
        def done(self): return self.ready
        def result(self): return {'answer': 1}
    class Client:
        def wait_for_service(self, **kwargs): return True
        def call_async(self, request):
            result = Future()
            pending.append(result)
            return result
    def create_client(*args):
        calls.append(args)
        return Client()
    module = ROS2ServiceModule('/fk', {'service_type': SimpleNamespace(Request=lambda: object())})
    module._node = SimpleNamespace(create_client=create_client)
    module._executor = SimpleNamespace(spin_once=lambda **k: setattr(pending[-1], 'ready', True))
    assert module.call_sync({'raw_response': True}) == {'answer': 1}
    assert asyncio.run(module.call_async({'raw_response': True})) == {'answer': 1}
    assert len(calls) == 1
    released = []
    module._executor.shutdown = lambda: released.append('executor')
    module._node.destroy_node = lambda: released.append('node')
    from src.core.module_proxy import ModuleProxy
    proxy = ModuleProxy()
    proxy._modules['fk'] = module
    proxy.clear()
    assert released == ['executor', 'node']
    assert module._client is None and module._node is None
    assert proxy._modules == {}


def test_skill_examples_follow_real_lifecycle_contract():
    root = Path(__file__).resolve().parents[2]
    text = (root/'skills/orchid-actions/references/examples.md').read_text()
    code = text.split('```python\n', 1)[1].split('```', 1)[0]
    namespace = {}
    exec(compile(code, 'orchid-action-examples', 'exec'), namespace)
    context = Context()
    context.world_state.set('x', 1)
    inline(context)
    leaf = namespace['ReadValue']('x').build_node(context, 'Read')
    assert isinstance(leaf, BtActionLeaf)
    workflow = namespace['ReadWorkflow']('x').build_node(context, 'Workflow')
    list(workflow.tick())
    assert workflow.status == py_trees.common.Status.SUCCESS
    workflow.stop(py_trees.common.Status.INVALID)
    context.world_state.set('x', 2)
    list(workflow.tick())
    assert workflow.status == py_trees.common.Status.SUCCESS
    context.world_state.set('x', None)
    workflow.stop(py_trees.common.Status.INVALID)
    list(workflow.tick())
    assert workflow.status == py_trees.common.Status.FAILURE
