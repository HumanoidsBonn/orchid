"""Catalog-only ROS check: unique topic, no task goals or robot commands."""
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import Mock
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from src.orchestrator.session import Ros2RobotSessionNode
from src.workflows.catalog import WorkflowCatalog, WorkflowSettings
from tests.unit.test_workflow_catalog import make_context


def test_retained_catalog_late_subscriber_and_revision_suppression():
    owned=not rclpy.ok()
    if owned:rclpy.init()
    node=rclpy.create_node('workflow_catalog_test_'+uuid.uuid4().hex[:8])
    try:
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        publisher=node.create_publisher(String,'/orchid_test/catalog_'+uuid.uuid4().hex,qos)
        ctx=make_context();catalog=WorkflowCatalog(ctx,'robot',WorkflowSettings())
        spy=Mock(wraps=publisher)
        wrapper=SimpleNamespace(session=SimpleNamespace(context=ctx,workflow_catalog=catalog),_catalog_publisher=spy,_catalog_source=None,_catalog_revision=None,_session_instance_id='test-session')
        Ros2RobotSessionNode._publish_workflow_catalog(wrapper)
        Ros2RobotSessionNode._publish_workflow_catalog(wrapper)
        assert spy.publish.call_count==1
        messages=[]
        node.create_subscription(String,publisher.topic_name,lambda msg:messages.append(json.loads(msg.data)),qos)
        deadline=time.monotonic()+5
        while not messages and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        assert messages and messages[0]['session_id']=='test-session'
        assert messages[0]['revision']==catalog.document()['revision']
        ctx.scene_graph.add_node('apple','object',{'label':'apple'})
        Ros2RobotSessionNode._publish_workflow_catalog(wrapper)
        assert spy.publish.call_count==2
    finally:
        node.destroy_node()
        if owned:rclpy.shutdown()
