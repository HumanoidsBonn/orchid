import json
from types import SimpleNamespace
from threading import Lock
from unittest.mock import Mock

from src.orchestrator.session import Ros2RobotSessionNode


def test_heartbeat_reports_instance_readiness_and_task_without_motion():
    wrapper = object.__new__(Ros2RobotSessionNode)
    wrapper._session_instance_id = 'instance-a'
    wrapper._action_server = object()
    wrapper._heartbeat_publisher = Mock()
    wrapper.session = SimpleNamespace(_task_lock=Lock(), _active_task=None,
        _pending_task='pick jar', executor=SimpleNamespace(is_running=False))
    wrapper._publish_heartbeat()
    assert json.loads(wrapper._heartbeat_publisher.publish.call_args.args[0].data) == {
        'session_id': 'instance-a', 'ready': True, 'busy': True, 'task_description': 'pick jar', 'task_kind': 'task'}
    wrapper.session._pending_task = None
    wrapper._publish_heartbeat()
    state = json.loads(wrapper._heartbeat_publisher.publish.call_args.args[0].data)
    assert not state['busy'] and state['session_id'] == 'instance-a'
    wrapper._action_server = None
    wrapper._publish_heartbeat()
    assert not json.loads(wrapper._heartbeat_publisher.publish.call_args.args[0].data)['ready']
