"""Tests for the RobotSession class."""

import pytest
from src.orchestrator.session import RobotSession
from src.core.manifest import RobotManifest
from tests.fixtures.mock_controllers import MockNavController, MockArmController, MockGripperController
from tests.fixtures.mock_manifests import basic_registry, rich_manifest, minimal_manifest, unsupported_manifest
from src.plugin.registry import PluginRegistry
from src.core.controllers import BaseArmController, BaseGripperController, BaseNavController

try:
    import rclpy  # pragma: no cover - optional in CI
    HAS_RCLPY = True
except Exception:
    HAS_RCLPY = False

class TestRobotSessionCreation:
    """Validate that ``RobotSession`` builds the execution context correctly."""

    def test_rich_manifest_builds_full_context(self, rich_manifest: RobotManifest, basic_registry: PluginRegistry):
        if HAS_RCLPY and not rclpy.ok():
            rclpy.init()
        session = RobotSession(rich_manifest, basic_registry)
        # Basic attributes
        assert session.robot_id == "hero_bot"
        # Controllers should be instantiated (the mock classes are simple objects)
        print(type(session.context.nav))
        assert isinstance(session.context.nav, MockNavController)
        assert isinstance(session.context.nav, BaseNavController)
        assert isinstance(session.context.arm, BaseArmController)
        assert isinstance(session.context.arm, MockArmController)
        assert isinstance(session.context.gripper, BaseGripperController)
        assert isinstance(session.context.gripper, MockGripperController)
        # Supported actions should contain all declared capabilities
        for cap in ["navigation", "object_pick", "environment_perception"]:
            assert cap in session.supported_actions
        # No unsupported actions for a fully‑populated manifest
        assert session.unsupported_actions == []

    def test_minimal_manifest_builds_partial_context(self, minimal_manifest: RobotManifest, basic_registry: PluginRegistry):
        if HAS_RCLPY and not rclpy.ok():
            rclpy.init()
        session = RobotSession(minimal_manifest, basic_registry)
        assert session.robot_id == "pioneer"
        # Only navigation controller should be present
        assert isinstance(session.context.nav, BaseNavController)
        assert isinstance(session.context.nav, MockNavController)
        assert session.context.arm is None
        assert session.context.gripper is None
        # Supported actions should contain only "navigation"
        assert session.supported_actions == {"navigation": session.supported_actions["navigation"]}
        # No unsupported actions
        assert session.unsupported_actions == []

    def test_unsupported_capabilities_are_reported(self, unsupported_manifest: RobotManifest, basic_registry: PluginRegistry):
        if HAS_RCLPY and not rclpy.ok():
            rclpy.init()
        session = RobotSession(unsupported_manifest, basic_registry)
        assert "mavros" in session.unsupported_controllers
        assert "fly" in session.unsupported_actions
        # Navigation still recognized as supported
        assert "navigation" in session.supported_actions
        assert isinstance(session.context.nav, BaseNavController)
        assert isinstance(session.context.nav, MockNavController)
        assert session.context.arm is None
        assert session.context.gripper is None
        assert "nav2" in session.supported_controllers.values()


