"""Focused action feedback contracts; no motion, perception or network calls."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import py_trees
import pytest
from src.core.action import ActionResult
from src.core.failure import FailureInfo,FailureCode,FailureOrigin
from src.plugin.actions.navigation import MoveToTarget
from src.plugin.actions.observe import ScanScene,_DetectTargetInView
from src.plugin.actions.fetch import FetchObject
from src.plugin.actions.trajectory import ExecuteJointTrajectory
from tests.unit.test_workflow_catalog import make_context
from tests.unit.test_hsr_pickup import trajectory_message


@pytest.mark.parametrize('finish',['success','failure','cancel'])
def test_navigation_highlight_uses_support_and_raw_id_and_clears(finish):
    context=make_context()
    context.scene_graph.get_node('table_0').attributes['boxer_id']=41
    action=MoveToTarget('jar_near')
    action.build_node(context,'Visit')
    action.on_enter(context)
    assert context.world_state.get('scene_navigation_target')=={'id':'table_0','boxer_id':41}
    if finish=='cancel':action.on_cancel(context)
    else:action.on_complete(context,py_trees.common.Status.SUCCESS if finish=='success' else py_trees.common.Status.FAILURE)
    assert context.world_state.get('scene_navigation_target') is None
    context.nav.send_goal.assert_not_called()


@pytest.mark.parametrize('code,origin,expected',[(FailureCode.NOT_FOUND,FailureOrigin.PERCEPTION,1),(FailureCode.UNKNOWN,FailureOrigin.PERCEPTION,0),(FailureCode.CONTROLLER_FAILED,FailureOrigin.CONTROLLER,0)])
def test_scan_default_three_views_and_exhaustion_speech(monkeypatch,code,origin,expected):
    context=make_context();context.head=None;context.sensor_data_center=Mock()
    speech=Mock();monkeypatch.setattr('src.core.logger.speak',speech)
    calls=[]
    class Detect(py_trees.behaviour.Behaviour):
        def update(self):
            calls.append(self.name)
            self.result=ActionResult(status='failure',failure=FailureInfo(code,origin=origin))
            return py_trees.common.Status.FAILURE
    monkeypatch.setattr(_DetectTargetInView,'build_node',lambda self,ctx,name:Detect(name))
    action=ScanScene('mug',exploration_mode=True,strict_failures=True)
    assert action.tilt_offsets==[0.]
    root=action.build_node(context,'Scan');root.tick_once()
    assert len(calls)==(1 if code==FailureCode.UNKNOWN else 3)
    speech.assert_any_call('Looking for %s','mug')
    assert speech.call_count==expected+1
    if expected:speech.assert_any_call('%s not found','mug')


def test_fetch_clears_only_its_object_highlights_at_end():
    context=make_context();action=FetchObject('mug')
    context.world_state.set('scene_known_objects',[{'id':'jar_near','boxer_id':1}])
    context.world_state.set('scene_navigation_target',{'id':'table_0','boxer_id':2})
    action.on_cancel(context)
    assert context.world_state.get('scene_known_objects')==[]
    assert context.world_state.get('scene_navigation_target')['id']=='table_0'


def test_gui_feedback_preserves_both_highlight_roles():
    from src.orchestrator.session import Ros2RobotSessionNode
    context=make_context()
    context.world_state.set('scene_navigation_target',{'id':'table_0','boxer_id':41})
    context.world_state.set('scene_known_objects',[{'id':'jar_near','boxer_id':99}])
    wrapper=Ros2RobotSessionNode.__new__(Ros2RobotSessionNode)
    wrapper.session=NS(context=context,get_execution_snapshot=lambda:NS(status='idle'));wrapper._goal_started_monotonic=None;wrapper._session_instance_id='test-session';wrapper.gui_state_publisher=Mock()
    wrapper.publish_gui_state(robot_status='Running',active_node='Navigate')
    payload=json.loads(wrapper.gui_state_publisher.publish.call_args.args[0].data)
    assert payload['scene_highlights']=={'navigation_target':{'id':'table_0','boxer_id':41},'known_objects':[{'id':'jar_near','boxer_id':99}]}
    assert payload['session_id']=='test-session'


def test_staging_speech_once_before_valid_dispatch(monkeypatch):
    context=make_context();context.arm.get_status.return_value='succeeded'
    context.run_in_context_loop=lambda work,**kwargs:work()
    trajectory=trajectory_message({'joint_names':['arm_lift_joint'],'points':[{'positions':[.1],'time_from_start':0.},{'positions':[.2],'time_from_start':1.}]})
    context.world_state.set('stage',trajectory)
    events=[];monkeypatch.setattr('src.core.logger.speak',lambda message:events.append(message))
    context.arm.send_synchronized_trajectory_goal.side_effect=lambda *a,**k:events.append('dispatch')
    action=ExecuteJointTrajectory('stage',start_message='Starting pickup execution')
    action.start(context);action.check_result(context);action.check_result(context)
    assert events==['Starting pickup execution','dispatch']
    ExecuteJointTrajectory('missing',start_message='wrong').start(context)
    assert 'wrong' not in events


@pytest.mark.parametrize("target_label,spoken", [(None,"coffee mug"),("Coffee mug","Coffee mug")])
def test_observation_speaks_only_after_successful_extraction(monkeypatch,target_label,spoken):
    import numpy as np
    from src.plugin.actions.observe import ObserveScene
    context=make_context();context.sensor_data_center=Mock()
    action=ObserveScene(target='coffee_mug',target_label=target_label)
    action._context=context;action.sdc=context.sensor_data_center;action.hware={'odom':'odom'}
    action._extract_content=Mock(return_value='response')
    action._extract_bb_and_mask=Mock(return_value=([0,0,4,4],[[0,0,4,0,4,4]]))
    points=np.ones((2,3))
    extraction=NS(reason=None,frame_id='camera',object_points=points,surround_points=points,object_colors=points,surround_colors=points,method='mock')
    action._extract_observation_points=Mock(return_value=extraction)
    action.sdc.transform_data.return_value=points
    action._viz_pcd=Mock()
    monkeypatch.setattr('src.plugin.actions.perception_preview.observation_preview',lambda *a:None)
    monkeypatch.setattr('time.sleep',lambda *a:None)
    speech=Mock();monkeypatch.setattr('src.core.logger.speak',speech)
    action.response_callback({})
    assert action.status=='success'
    speech.assert_called_once_with('%s detected',spoken)
    speech.reset_mock();extraction.reason='missing_depth'
    action.response_callback({})
    assert action.status=='failure';speech.assert_not_called()
