"""Tests for ROS2 service module helpers."""

from __future__ import annotations

from src.core.modules.ros2_service_module import ROS2ServiceModule
from src.core.controllers import Pose3D


def test_import_type_resolves_colon_reference():
    resolved = ROS2ServiceModule._import_type("src.core.controllers:Pose3D")
    assert resolved is Pose3D


def test_serialize_response_uses_ros_fields():
    class FloatValue:
        def __init__(self, data):
            self.data = data

        def get_fields_and_field_types(self):
            return {"data": "float32"}

    class Grasp:
        def __init__(self):
            self.score = FloatValue(0.7)

        def get_fields_and_field_types(self):
            return {"score": "std_msgs/Float32"}

    class Response:
        def __init__(self):
            self.grasps = [Grasp()]

        def get_fields_and_field_types(self):
            return {"grasps": "sequence"}

    assert ROS2ServiceModule._serialize_response(Response()) == {
        "grasps": [{"score": {"data": 0.7}}]
    }
