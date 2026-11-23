from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.core.context import Context
from src.core.controllers import Pose
from src.core.scene_graph import SceneGraph
from src.workflows.catalog import WorkflowCatalog, WorkflowSettings, DEFINITIONS
from src.plugin.actions.fetch import choose_search_locations


def make_context():
    graph = SceneGraph()
    graph.add_node('kitchen', 'room', {'room_type':'kitchen'})
    for i in range(8):
        graph.add_node(f'table_{i}', 'furniture', {'label':'table', 'access_poses':[{'x':float(i+1),'y':0.,'yaw':0.,'frame_id':'map','valid':True}]})
        graph.add_edge('kitchen', f'table_{i}', 'contains')
    graph.add_node('blocked', 'furniture', {'label':'shelf', 'access_poses':[]})
    graph.add_node('invalid', 'furniture', {'label':'shelf', 'access_poses':[{'x':0,'y':0,'yaw':0,'frame_id':'map','valid':False}]})
    for key, support in [('jar_far','table_7'),('jar_near','table_0')]:
        graph.add_node(key,'object',{'label':'blue jar'})
        graph.add_edge(support,key,'supports')
    graph.add_node('wall','null_class',{'label':'wall'})
    graph.add_node('fixed','object',{'label':'fixed fixture','movable':False})
    ctx = Context(scene_graph=graph, nav=Mock(), arm=Mock(), head=Mock(), gripper=Mock())
    ctx.nav.get_current_pose.return_value=Pose(0,0)
    ctx.supported_actions={key:object for d in DEFINITIONS.values() for key in d.required_actions}
    return ctx


def test_catalog_filter_revision_and_no_images():
    ctx=make_context(); catalog=WorkflowCatalog(ctx,'robot',WorkflowSettings(include_scene_items=True,items=[{'id':'banana','label':'Banana','query':'banana'}]))
    document=catalog.document()
    assert {row['id'] for row in document['locations']}=={f'table_{i}' for i in range(8)}
    assert all(row['room']=='kitchen' for row in document['locations'])
    assert {row['label'] for row in document['items']}=={'Banana','blue jar'}
    assert document['revision']==catalog.document()['revision']
    assert 'access_poses' not in str(document)
    ctx.scene_graph.add_node('apple','object',{'label':'apple'})
    assert catalog.document()['revision']!=document['revision']


def test_known_nearest_then_five_valid_unique_extras():
    ctx=make_context()
    response={'known_object_ids':['jar_far','jar_near'],'additional_furniture_ids':['table_0','table_2','table_2','table_1','table_3','table_4','table_5','table_6']}
    assert choose_search_locations(ctx,'blue jar',5,response)==['table_0','table_7','table_2','table_1','table_3','table_4','table_5']
    with pytest.raises(ValueError): choose_search_locations(ctx,'blue jar',5,{'known_object_ids':[], 'additional_furniture_ids':['blocked']})


def test_release_home_factories_validate_without_llm():
    ctx=make_context();ctx.modules.call=Mock(side_effect=AssertionError('LLM must not run'))
    catalog=WorkflowCatalog(ctx,'robot',WorkflowSettings())
    release=catalog.build_plan(catalog.validate('release',{}))
    assert release.root.type=='retry' and release.root.params['num_attempts']==2
    assert release.root.child.action_type=='ReleaseObjectAction'
    close=catalog.build_plan(catalog.validate('close_gripper',{}))
    assert close.root.type=='retry' and close.root.params['num_attempts']==2
    assert close.root.child.action_type=='GraspObjectAction' and close.root.child.params=={}
    assert next(row for row in catalog.document()['workflows'] if row['id']=='close_gripper')['available']
    home=catalog.build_plan(catalog.validate('home',{}))
    assert [(n.action_type,n.params['state']) for n in home.root.children]==[('MoveArmToState','go'),('MoveHeadToState','center')]
    ctx.modules.call.assert_not_called()
    with pytest.raises(ValueError): catalog.validate('release',{'distance':2})
    with pytest.raises(ValueError): catalog.validate('close_gripper',{'unknown':2})
    with pytest.raises(ValueError): catalog.validate('fetch',{'item_id':'unknown'})
    del ctx.supported_actions['HsrPickup']
    assert not next(row for row in catalog.document()['workflows'] if row['id']=='fetch')['available']


def test_scene_items_are_opt_in_and_custom_choices_remain_valid():
    ctx=make_context()
    settings=WorkflowSettings(items=[{'id':'mug','label':'Coffee mug','query':'coffee mug'}])
    catalog=WorkflowCatalog(ctx,'robot',settings)
    assert catalog.items()==[{'id':'mug','label':'Coffee mug','query':'coffee mug'}]
    revision=catalog.document()['revision']
    assert catalog.validate('fetch',{'item_id':'mug'}).inputs=={'item_id':'mug'}
    with pytest.raises(ValueError,match='no longer available'):
        catalog.validate('fetch',{'item_id':'scene:jar_near'})
    settings.include_scene_items=True
    assert {item['query'] for item in catalog.items()}=={'coffee mug','blue jar'}
    assert catalog.document()['revision']!=revision


def test_shipped_custom_fetch_list():
    import json
    from pathlib import Path
    settings=WorkflowSettings.model_validate(json.loads((Path(__file__).resolve().parents[2]/'config/workflows.json').read_text()))
    assert not settings.include_scene_items
    assert {'mustard_bottle','cardboard_food_box','coffee_mug'} <= {item.id for item in settings.items}


def test_shared_admission_and_atomic_consume():
    import threading
    from src.orchestrator.session import RobotSession
    session=RobotSession.__new__(RobotSession)
    session._task_lock=threading.Lock();session._task_event=threading.Event();session._pending_task=None;session._active_task=None;session._intake_reserved=False;session.planning=False;session.robot_id='robot'
    session.executor=SimpleNamespace(is_running=False)
    session.workflow_catalog=WorkflowCatalog(make_context(),'robot',WorkflowSettings())
    assert session.reserve_task()
    assert not session.submit_task('other')
    assert session.submit_workflow('home',{},reserved=True)
    assert not session.submit_task('concurrent')
    task=session._consume_pending_task()
    assert str(task)=='Home' and session._active_task=='Home'
    assert not session.submit_workflow('release',{})


@pytest.mark.parametrize('succeeds',[True,False])
def test_workflow_session_never_calls_general_planner_or_replays(succeeds):
    from src.orchestrator.session import RobotSession
    from src.tree.execution import ExecutionStatus
    session=RobotSession.__new__(RobotSession)
    session.context=make_context();session.robot_id='robot';session.workflow_catalog=WorkflowCatalog(session.context,'robot',WorkflowSettings())
    session.executor=Mock();session.executor.is_running=False;session.executor._thread=None
    session.executor.get_snapshot.return_value=SimpleNamespace(status=ExecutionStatus.SUCCEEDED if succeeds else ExecutionStatus.FAILED)
    session.planner=Mock();session.stop_planning=False;session.planning=False
    session._set_current_operation=Mock();session._publish_task_status=Mock()
    session._handle_new_task(session.workflow_catalog.validate('release',{}))
    session.planner.generate_plan.assert_not_called()
    session.executor.start_plan.assert_called_once()
    assert session._active_task is None and not session.planning
