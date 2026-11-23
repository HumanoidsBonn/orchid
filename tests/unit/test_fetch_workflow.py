import json
from unittest.mock import Mock
import py_trees
import pytest
from src.core.action import ActionResult
from src.core.failure import FailureCode, FailureInfo
from src.core.llm_planner import LLMPlanner
from src.plugin.actions.fetch import FetchObject, ResolveFetchLocations
from tests.unit.test_workflow_catalog import make_context


@pytest.mark.parametrize('failure',[None,'close','scan_backend','pickup','delivery'])
def test_native_fetch_hints_pickup_delivery_order(monkeypatch,failure):
    from src.plugin.actions.navigation import MoveToTarget
    from src.plugin.actions.observe import ScanScene, ObserveScene
    from src.plugin.actions.grasp_alignment import EstimateSimplePcaGrasp
    from src.plugin.actions.hsr_pickup import HsrPickup
    from src.plugin.actions.manipulation import GraspObjectAction, ReleaseObjectAction
    ctx=make_context();trace=[];search_labels=[];scans=0;answers=iter(['table_0','here','table_1'])
    speech=[]
    monkeypatch.setattr('src.core.logger.speak',lambda text,*args:speech.append(text%args if args else text))
    ctx.request_user_input=Mock(side_effect=lambda prompt,**kwargs:next(answers))
    def immediately(fn,*args,callback=None,error_callback=None,**kwargs):
        try: result=fn(*args,**kwargs)
        except Exception as exc:
            if error_callback:error_callback(exc)
            else:raise
        else:
            if callback:callback(result)
    ctx.run_in_context_loop=immediately
    monkeypatch.setattr(LLMPlanner,'_resolve_llm_module_name',lambda self:'planner_llm')
    ctx.modules.call=Mock(return_value={'content':json.dumps({'known_object_ids':['jar_near'],'additional_furniture_ids':['table_1']})})
    class Operation(py_trees.behaviour.Behaviour):
        def __init__(self,action,name):super().__init__(name);self.action=action;self.result=None
        def update(self):
            nonlocal scans
            key=type(self.action).__name__;trace.append((key,getattr(self.action,'target',None)))
            if isinstance(self.action, MoveToTarget) and not any(t[0]=='HsrPickup' for t in trace):
                search_labels.append(self.parent.parent.name)
            code=None
            if isinstance(self.action,GraspObjectAction) and failure=='close':code=FailureCode.CONTROLLER_FAILED
            if isinstance(self.action,ScanScene):
                scans+=1
                if failure=='scan_backend':code=FailureCode.UNKNOWN
                elif scans<=2:code=FailureCode.NOT_FOUND
            if isinstance(self.action,HsrPickup) and failure=='pickup':code=FailureCode.CONTROLLER_FAILED
            if isinstance(self.action,MoveToTarget) and failure=='delivery' and any(t[0]=='HsrPickup' for t in trace):code=FailureCode.CONTROLLER_FAILED
            self.result=ActionResult(status='failure' if code else 'success',failure=FailureInfo(code) if code else None)
            return py_trees.common.Status.FAILURE if code else py_trees.common.Status.SUCCESS
    for cls in [GraspObjectAction,MoveToTarget,ScanScene,ObserveScene,EstimateSimplePcaGrasp,HsrPickup,ReleaseObjectAction]:
        monkeypatch.setattr(cls,'build_node',lambda self,context,name:Operation(self,name))
    action=FetchObject('blue jar');root=action.build_node(ctx,'Fetch')
    descendants=list(root.iterate())
    topology = {n.id for n in descendants}
    assert any(n.name=='Scan for target' for n in descendants)
    assert not any('Inspect' in n.name for n in descendants)
    for _ in range(100):
        root.tick_once()
        if root.status in (py_trees.common.Status.SUCCESS,py_trees.common.Status.FAILURE):break
    names=[t[0] for t in trace]
    assert names[0]=='GraspObjectAction'
    if failure=='close':
        assert ctx.modules.call.call_count==0 and 'MoveToTarget' not in names and 'ScanScene' not in names
    if failure:
        assert root.status==py_trees.common.Status.FAILURE
        assert 'ReleaseObjectAction' not in names
        if failure=='scan_backend':assert 'HsrPickup' not in names
        if failure=='scan_backend':assert 'blue jar not found' not in speech
        if failure=='scan_backend':
            assert scans > 1 and ctx.request_user_input.called  # Exhausted calls reach the search-hint question.
    else:
        assert root.status==py_trees.common.Status.SUCCESS
        assert names.count('HsrPickup')==1 and 'ReleaseObjectAction' not in names
        assert names.index('ObserveScene')<names.index('EstimateSimplePcaGrasp')<names.index('HsrPickup')<len(names)-1
        assert trace[-1]==('MoveToTarget','table_1')
        assert ctx.modules.call.call_count==1
        assert speech.count('Continuing to look for blue jar')==1
    assert ctx.execution.control is None
    assert {n.id for n in root.iterate()} == topology
    assert all(label.startswith('Search: table (table_') and 'kitchen' in label for label in search_labels)
    assert bool(search_labels)==(failure!='close')
    assert not any(n.name.startswith('Candidate ') for n in root.iterate())
    if failure!='close':
        assert any(n.name == 'Unused search slot' for n in root.iterate())


def test_resolver_cancelled_callback_cannot_write(monkeypatch):
    ctx=make_context();queued=[]
    ctx.run_in_context_loop=lambda fn,*args,callback=None,error_callback=None,**kwargs:queued.append(callback)
    leaf=ResolveFetchLocations('blue jar','test.output')
    leaf.start(ctx);leaf.cancel(ctx);queued[0]({'known_object_ids':[],'additional_furniture_ids':[]})
    assert ctx.world_state.get('test.output') is None


def test_ask_user_continue_and_stale_response():
    from src.plugin.actions.user_input import AskUser
    ctx=make_context();callbacks=[]
    ctx.request_user_input=lambda prompt,**kwargs:'table_0'
    ctx.run_in_context_loop=lambda fn,*args,callback=None,**kwargs:callbacks.append(callback)
    action=AskUser(replan_on_response=False,choices=[{'value':'table_0','label':'Table'}])
    action.start(ctx);action.cancel(ctx);callbacks[0]('stale')
    assert ctx.world_state.get('last_user_input') is None
    action.reset();action.start(ctx);callbacks[-1]('table_0')
    assert action.check_result(ctx).status=='success'
    legacy=AskUser();legacy.start(ctx);callbacks[-1]('yes')
    assert legacy.check_result(ctx).status=='needs_replan'


@pytest.mark.parametrize('failure_code',[FailureCode.NOT_FOUND,FailureCode.UNKNOWN])
def test_strict_scan_preserves_backend_failure_vs_miss(monkeypatch,failure_code):
    from src.plugin.actions.observe import ScanScene, _DetectTargetInView
    ctx=make_context();ctx.head=None;ctx.sensor_data_center=object();seen=[]
    class Detect(py_trees.behaviour.Behaviour):
        def __init__(self,name):super().__init__(name);self.result=None
        def update(self):
            seen.append(self.name)
            self.result=ActionResult(status='failure',failure=FailureInfo(failure_code),reason='classified failure')
            return py_trees.common.Status.FAILURE
    monkeypatch.setattr(_DetectTargetInView,'build_node',lambda self,context,name:Detect(name))
    root=ScanScene(target='jar',exploration_mode=True,strict_failures=True).build_node(ctx,'Scan')
    root.tick_once()
    assert root.status==py_trees.common.Status.FAILURE
    assert root.result.failure.code==failure_code
    assert len(seen)==(3 if failure_code==FailureCode.NOT_FOUND else 1)
