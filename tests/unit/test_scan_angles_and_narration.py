"""Focused scan input binding and action-owned speech checks."""
import json
from unittest.mock import Mock

import py_trees
import pytest

from src.core.action import AbstractAction, ActionResult
from src.core.context import Context
from src.core.llm_planner import LLMPlanner
from src.plugin.actions.fetch import ResolveFetchLocations
from src.plugin.actions.navigation import MoveToTarget
from src.plugin.actions.observe import ScanScene, _DetectTargetInView
from tests.unit.test_workflow_catalog import make_context


def test_scan_snapshots_tilt_once_clamps_offsets_and_rebinds_on_reentry(monkeypatch):
    context=make_context()
    context.sensor_data_center=Mock()
    context.modules=Mock()
    context.head.get_status.return_value="succeeded"
    context.hardware["head"]={"joint_limits":{"head_tilt_joint":[-.5,.5]}}
    snapshot=Mock(side_effect=[{"head_tilt_joint":.4},{"head_tilt_joint":-.4}])
    monkeypatch.setattr("src.plugin.actions.navigation._head_joint_snapshot",snapshot)
    monkeypatch.setattr(_DetectTargetInView,"build_node",lambda self,ctx,name:py_trees.behaviours.Failure(name=name))
    scan=ScanScene(target="jar",pan_angles=[0.],tilt_offsets=[0.,.25,-.25],exploration_mode=True)
    root=scan.build_node(context,"Scan")
    topology={n.id for n in root.iterate()}
    root.tick_once()
    assert [c.args[1] for c in context.head.send_pan_tilt.call_args_list]==pytest.approx([.4,.5,.15])
    assert snapshot.call_count==1
    assert [detector.tilt for _,detector,_,_ in scan._view_inputs]==pytest.approx([.4,.5,.15])
    context.head.send_pan_tilt.reset_mock()
    root.tick_once()
    assert [c.args[1] for c in context.head.send_pan_tilt.call_args_list]==pytest.approx([-.4,-.15,-.5])
    assert snapshot.call_count==2
    assert topology=={n.id for n in root.iterate()}


def test_scan_without_measured_tilt_does_not_dispatch(monkeypatch):
    context=make_context()
    context.sensor_data_center=context.modules=Mock()
    monkeypatch.setattr("src.plugin.actions.navigation._head_joint_snapshot",lambda *a,**k:None)
    root=ScanScene(target="jar").build_node(context,"Scan")
    root.tick_once()
    assert root.status==py_trees.common.Status.FAILURE
    context.head.send_pan_tilt.assert_not_called()


@pytest.mark.parametrize("status",["success","failure","needs_replan"])
def test_adapter_never_invents_action_speech(monkeypatch,status):
    speak=Mock()
    monkeypatch.setattr("src.core.logger.speak",speak)
    class Quiet(AbstractAction):
        def execute(self,context):return ActionResult(status=status)
    context=Context(supported_actions={"Quiet":Quiet})
    root=Quiet().build_node(context,"Quiet")
    root.binding.action_type="Quiet"
    root.binding.planner_visible=True
    root.tick_once()
    speak.assert_not_called()


def test_fetch_announces_planning_and_all_known_locations(monkeypatch):
    context=make_context()
    context.modules=Mock()
    context.modules.call.return_value={"content":json.dumps({"known_object_ids":["jar_far","jar_near"],"additional_furniture_ids":[]})}
    def immediately(work,callback=None,error_callback=None):
        callback(work())
    context.run_in_context_loop=immediately
    monkeypatch.setattr(LLMPlanner,"_resolve_llm_module_name",lambda self:"planner_llm")
    speech=[]
    monkeypatch.setattr("src.core.logger.speak",lambda text,*args:speech.append(text%args if args else text))
    action=ResolveFetchLocations("blue_jar","locations")
    action.start(context)
    assert action.check_result(context).status=="success"
    assert speech[0]=="Planning"
    assert speech[1]=="Found the last known locations of blue jar on table in room kitchen; and on table in room kitchen"
    assert context.world_state.get("locations")==["table_0","table_7"]
    assert {ref['id'] for ref in context.world_state.get('scene_known_objects')}=={'jar_far','jar_near'}
    speech.clear()
    context.world_state.set("answer","table_0")
    ResolveFetchLocations("blue_jar","destination",kind="delivery",response_key="answer").start(context)
    assert speech==[]  # Exact choices do not invoke or announce LLM planning.


def test_navigation_announces_furniture_and_room_without_dispatch(monkeypatch):
    context=make_context()
    speech=Mock()
    monkeypatch.setattr("src.core.logger.speak",speech)
    action=MoveToTarget("table_0")
    action.build_node(context,"Visit")
    assert action.on_enter(context) is None
    speech.assert_called_once_with("Going to %s%s","table"," in room kitchen")
    context.nav.send_goal.assert_not_called()


@pytest.mark.parametrize('name,expected', [('work_desk_41','work desk'),('work desk 41','work desk'),('work-desk-41','work-desk'),('desk2','desk'),('3d_printer','3d printer')])
def test_spoken_location_removes_only_trailing_numbers(name,expected):
    from src.plugin.actions._helpers import scene_location_description
    assert scene_location_description(Context(),name)==(expected,None)
