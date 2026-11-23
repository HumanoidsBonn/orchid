"""Semantic stage metadata, existing transport and Fetch boundaries; no motion."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import py_trees
from src.core.context import Context
from src.tree.runtime import ExecutionRuntime
from src.tree.execution import ExecutionStatus
from src.plugin.actions.fetch import FetchObject
from tests.unit.test_workflow_catalog import make_context


def test_deepest_running_stage_and_terminal_reset():
    child=py_trees.behaviours.Running('work')
    inner=py_trees.composites.Sequence('inner',memory=True,children=[child])
    root=py_trees.composites.Sequence('outer',memory=True,children=[inner])
    runtime=ExecutionRuntime();runtime.stage_labels={root.id:'Performing pickup',inner.id:'Exploring'}
    root.tick_once();assert runtime.active_stage(root)=='Exploring'
    root.stop(py_trees.common.Status.INVALID);assert runtime.active_stage(root) is None


def test_fetch_labels_known_and_additional_visits_and_has_no_release():
    ctx=make_context();action=FetchObject('blue jar');root=action.build_node(ctx,'Fetch')
    action.on_enter(ctx)
    ctx.world_state.set(action._key+'.search',['table_0','table_1'])
    ctx.world_state.set(action._key+'.search.known_locations',['table_0'])
    attempts=sorted([n for n in root.iterate() if n.name.startswith('Awaiting search ranking: slot')],key=lambda n:int(n.name.rsplit(' ',1)[1]))
    attempts[0].initialise();attempts[1].initialise()
    assert ctx.execution.stage_labels[attempts[0].id]=='Verifying at known location'
    assert ctx.execution.stage_labels[attempts[1].id]=='Exploring'
    assert {'Preparing gripper','Planning for locations','Estimating Grasp','Performing pickup','Asking for drop','Going to drop','Asking where to look'}<=set(ctx.execution.stage_labels.values())
    delivery=next(n for n in root.iterate() if n.name=='Delivery')
    assert not any(type(getattr(n,'action',None)).__name__=='ReleaseObjectAction' for n in delivery.iterate())
    action.on_enter(ctx);assert ctx.world_state.get(action._key+'.search.known_locations')==[]


def test_gui_stage_authored_coarse_and_idle():
    from src.orchestrator.session import Ros2RobotSessionNode
    ctx=Context();snapshot=NS(status=ExecutionStatus.RUNNING,task_stage='Estimating Grasp')
    wrapper=Ros2RobotSessionNode.__new__(Ros2RobotSessionNode)
    wrapper.session=NS(context=ctx,_active_task='fetch',_pending_task=None,get_execution_snapshot=lambda:snapshot)
    wrapper._goal_started_monotonic=None;wrapper._session_instance_id='stage';wrapper.gui_state_publisher=Mock()
    def published():
        wrapper.publish_gui_state(robot_status='Running',active_node='test')
        return json.loads(wrapper.gui_state_publisher.publish.call_args.args[0].data)['task_stage']
    assert published()=='Estimating Grasp'
    snapshot.task_stage=None;ctx.world_state.set('current_operation',{'sequence_label':'Planning'})
    assert published()=='Planning'
    snapshot.status=ExecutionStatus.SUCCEEDED;wrapper.session._active_task=None
    assert published() is None


def test_scan_first_miss_announces_panning_before_next_head_view(monkeypatch):
    from src.core.action import ActionResult
    from src.core.failure import FailureInfo,FailureCode
    from src.plugin.actions.observe import ScanScene,_DetectTargetInView
    from src.plugin.actions.head import MoveHeadToPanTilt
    ctx=make_context();ctx.sensor_data_center=Mock();events=[]
    monkeypatch.setattr('src.plugin.actions.navigation._head_joint_snapshot',lambda *a,**k:{'head_tilt_joint':-.2})
    monkeypatch.setattr('src.core.logger.speak',lambda fmt,*args:events.append(fmt%args if args else fmt))
    class Head(py_trees.behaviour.Behaviour):
        def update(self):events.append(self.name);return py_trees.common.Status.SUCCESS
    class Detect(py_trees.behaviour.Behaviour):
        def update(self):
            missing=self.name=='Detect 0'
            self.result=ActionResult(status='failure' if missing else 'success',failure=FailureInfo(FailureCode.NOT_FOUND) if missing else None)
            return py_trees.common.Status.FAILURE if missing else py_trees.common.Status.SUCCESS
    monkeypatch.setattr(MoveHeadToPanTilt,'build_node',lambda self,ctx,name:Head(name))
    monkeypatch.setattr(_DetectTargetInView,'build_node',lambda self,ctx,name:Detect(name))
    root=ScanScene('blue_jar',exploration_mode=True,strict_failures=True).build_node(ctx,'Scan')
    root.tick_once()
    assert root.status==py_trees.common.Status.SUCCESS
    assert events==['Looking for blue jar','Look 0','blue jar not found, panning left-right','Look 1']
    root.tick_once()
    assert events.count('Looking for blue jar')==2
    assert events.count('blue jar not found, panning left-right')==2
