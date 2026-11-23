"""Focused ROS message contract checks; no controller or motion dispatch."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.core.logger import CustomLogger
from src.core.sensor_data_center import SensorDataCenter
from src.orchestrator.session import Ros2RobotSessionNode
from src.orchestrator.dispatch_server import _parse_dispatch_args
from src.plugin.controllers.ros2_control_trajectory import Ros2ControlController, _ControllerGoal


def test_speech_modes_preserve_logs_and_replace_sinks():
    logged, first, second = [], [], []
    logger = CustomLogger(name='speech-mode-test', console=False,
                          sinks=[lambda level, msg: logged.append(msg)])
    logger.set_speech_publisher(first.append)
    logger.speak('hello %s', 'robot')
    logger.set_speech_mode('off')
    logger.speak('silent')
    logger.set_speech_mode('tmc')
    logger.set_speech_publisher(second.append)
    logger.speak('robot voice')
    logger.set_speech_publisher(None)
    logger.speak('detached')
    assert first == ['hello robot']
    assert second == ['robot voice']
    assert logged == ['hello robot', 'silent', 'robot voice', 'detached']
    with pytest.raises(ValueError):
        logger.set_speech_mode('typo')


def test_speech_boot_selection(monkeypatch):
    monkeypatch.setenv('ORCHID_SPEECH_MODE', 'off')
    assert _parse_dispatch_args([])[0].speech_mode == 'off'
    assert _parse_dispatch_args(['--speech-mode', 'tmc'])[0].speech_mode == 'tmc'


def test_tmc_message_contract():
    wrapper = object.__new__(Ros2RobotSessionNode)
    wrapper._tmc_speech_publisher = Mock()
    wrapper._publish_tmc_speech('Ready')
    msg = wrapper._tmc_speech_publisher.publish.call_args.args[0]
    assert (msg.interrupting, msg.queueing, msg.language, msg.sentence) == (False, False, 1, 'Ready')


def test_pose_qos_nonblocking_transform_stamp_and_shutdown():
    import rclpy
    from rclpy.qos import ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import TransformStamped
    context = rclpy.Context()
    rclpy.init(context=context)
    node = rclpy.create_node('orchid_pose_contract_test', context=context)
    try:
        center = SensorDataCenter('pose-test', SimpleNamespace(hardware={'base_frame':'base_footprint'}))
        center.node = center._rclpy_node = node
        tf = TransformStamped()
        tf.header.frame_id = 'map'; tf.header.stamp.sec = 42
        tf.transform.translation.x = 1.2; tf.transform.translation.z = 0.1
        tf.transform.rotation.w = 1.0
        center._tf_buffer = Mock()
        center._tf_buffer.lookup_transform.return_value = tf
        center._start_pose_publisher()
        assert center._pose_timer.timer_period_ns == 100000000
        qos = center._pose_publisher.qos_profile
        assert qos.depth == 1 and qos.reliability == ReliabilityPolicy.BEST_EFFORT
        assert qos.durability == DurabilityPolicy.VOLATILE
        original = center._pose_publisher
        center._pose_publisher = Mock()
        center._publish_robot_pose()
        call = center._tf_buffer.lookup_transform.call_args
        assert call.args[:2] == ('map', 'base_footprint')
        assert call.kwargs['timeout'].nanoseconds == 0
        msg = center._pose_publisher.publish.call_args.args[0]
        assert msg.header.stamp.sec == 42 and msg.header.frame_id == 'map'
        assert msg.pose.position.x == 1.2 and msg.pose.position.z == 0.1
        center._tf_buffer.lookup_transform.side_effect = RuntimeError('missing TF')
        center._publish_robot_pose()
        assert center._pose_publisher.publish.call_count == 1
        center._pose_publisher = original
        center.shutdown()
        assert center._pose_publisher is None and center._pose_timer is None
    finally:
        node.destroy_node(); rclpy.shutdown(context=context)


@pytest.mark.parametrize('frequency', [0, -1, float('nan'), float('inf')])
def test_pose_frequency_validation(frequency):
    with pytest.raises(ValueError):
        SensorDataCenter('invalid', SimpleNamespace(hardware={'robot_pose_publisher':{'frequency_hz':frequency}}))


def test_sender_populates_named_base_tolerances_without_arm_tolerances():
    from trajectory_msgs.msg import JointTrajectory
    client = Ros2ControlController({'goal_response_retry_settle_s':0})
    action_client = Mock()
    client._client = Mock(return_value=action_client)
    client._spin_once = Mock()
    trajectory = JointTrajectory(joint_names=['odom_y','arm_lift_joint','odom_t','odom_x'])
    client._send_controller_trajectory_goal(_ControllerGoal('base','/mock',trajectory.joint_names), trajectory)
    goal = action_client.send_goal_async.call_args.args[0]
    values = {t.name:t.position for t in goal.path_tolerance}
    assert values == {'odom_x':.25,'odom_y':.25,'odom_t':math.pi/6}
    assert not goal.goal_tolerance
    custom = Ros2ControlController({'base_path_tolerance_by_joint':{'odom_x':.3}})
    assert custom.base_path_tolerance['odom_x'] == .3
    with pytest.raises(ValueError):
        Ros2ControlController({'base_path_tolerance_by_joint':{'odom_x':float('nan')}})
