"""Mock session fixtures for testing RobotSession and TreeBuilder."""

import pytest
from src.orchestrator.session import RobotSession
from src.tree.builder import TreeBuilder
from src.plugin.registry import PluginRegistry

# Reuse existing mock manifests
from tests.fixtures.mock_manifests import rich_manifest, minimal_manifest


@pytest.fixture
def basic_registry() -> PluginRegistry:
    """
    Create a registry populated with the action and controller classes from mock_manifests.

    This is a lightweight fixture that mimics the real discovery process without
    requiring actual plugin entry points.
    """
    from tests.fixtures.mock_manifests import (
        NavigationAction, ObjectPickAction, PerceptionAction,
        Nav2Controller, MoveitController, HsrGripperController
    )

    registry = PluginRegistry()

    # Register actions
    registry.register_action("navigation", NavigationAction)
    registry.register_action("object_pick", ObjectPickAction)
    registry.register_action("environment_perception", PerceptionAction)

    # Register controllers
    registry.register_controller("nav2", Nav2Controller)
    registry.register_controller("moveit", MoveitController)
    registry.register_controller("hsr_gripper", HsrGripperController)

    registry.discover()
    return registry


# Session fixtures using rich_manifest and minimal_manifest
@pytest.fixture
def session_rich(rich_manifest, basic_registry) -> RobotSession:
    """Create RobotSession for the rich manifest fixture."""
    return RobotSession(rich_manifest, basic_registry)


@pytest.fixture
def session_minimal(minimal_manifest, basic_registry) -> RobotSession:
    """Create RobotSession for the minimal manifest fixture."""
    return RobotSession(minimal_manifest, basic_registry)


@pytest.fixture
def tree_builder_full(session_rich) -> TreeBuilder:
    """Create TreeBuilder with a session that has full controller set."""
    return TreeBuilder(context=session_rich.context)


@pytest.fixture
def tree_builder_minimal(session_minimal) -> TreeBuilder:
    """Create TreeBuilder with a session that has minimal controller set."""
    return TreeBuilder(context=session_minimal.context)