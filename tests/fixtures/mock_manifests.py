"""Test fixtures for mock robot manifests."""

from src.core.action import AbstractAction, AsyncAction
from src.core.controllers import BaseArmController, BaseGripperController, BaseNavController, ControllerResult, Pose
from typing import Literal, Tuple
import pytest
from src.core.manifest import RobotManifest, ControllerMap, ModuleEndpoint
from src.plugin.registry import PluginRegistry
from tests.fixtures.mock_controllers import MockNavController, MockArmController, MockGripperController
from tests.fixtures.mock_actions import NavigationAction, ObjectPickAction, PerceptionAction

@pytest.fixture
def basic_registry() -> PluginRegistry:
    """
    Populate plugin registry via discovery mechanism.

    This fixture:
    1. Creates a MockRegistry
    2. Pre-registers key actions/controllers
    3. Triggers discovery to populate registry
    """
    registry = PluginRegistry()  # Will use MockRegistry during tests

    # Register core actions
    registry.register_action("navigation", NavigationAction)
    registry.register_action("object_pick", ObjectPickAction)
    registry.register_action("environment_perception", PerceptionAction)

    # Register controllers
    registry.register_controller("nav2", MockNavController)
    registry.register_controller("moveit", MockArmController)
    registry.register_controller("hsr_gripper", MockGripperController)

    # Run discovery (no‑op for the test suite)
    registry.discover()

    return registry

@pytest.fixture
def rich_manifest(basic_registry):
    """
    Create a rich mobile manipulation manifest using populated registry.

    Returns:
        RobotManifest instance
    """
    # Use populated registry to get controller/action classes
    return RobotManifest(
        robot_id="hero_bot",
        robot_type="mobile_manipulation",
        capabilities=["navigation", "object_pick", "environment_perception"],
        controllers=ControllerMap(
            nav="nav2",
            arm="moveit",
            gripper="hsr_gripper"
        ),
        hardware={
            "camera": "realsense",
            "lidar": "yag_lidar",
            "microphone": "array_mic",
        },
        modules=[
            ModuleEndpoint(name="slam", endpoint="http://slam:8080", type="http"),
            ModuleEndpoint(name="weather", endpoint="https://api.weather.com", type="http"),
            ModuleEndpoint(name="battery_monitor", endpoint="/dom/battery", type="ros2_topic"),
        ],
    )


@pytest.fixture
def minimal_manifest(basic_registry):
    """
    Create a minimal differential-drive manifest.

    Returns:
        RobotManifest instance
    """
    return RobotManifest(
        robot_id="pioneer",
        robot_type="differential_drive",
        capabilities=["navigation"],
        controllers=ControllerMap(
            nav="nav2"
        ),
        hardware={"bumper": "mechanical"},
        modules=[]
    )

@pytest.fixture
def unsupported_manifest(basic_registry):
    """
    Create a manifest with unsupported capabilities.

    Returns:
        RobotManifest instance
    """
    return RobotManifest(
        robot_id="pioneer_2",
        robot_type="differential_drive",
        capabilities=["navigation", "fly"],  # "fly" is not registered
        controllers=ControllerMap(
            nav="nav2",
            arm="mavros"  # "mavros" controller not registered
        ),
        hardware={"bumper": "mechanical"},
        modules=[]
    )