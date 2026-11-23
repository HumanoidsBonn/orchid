"""Mock Context fixtures for testing."""

import pytest
from src.core.context import Context
from src.core.world_state import WorldState
from src.core.module_proxy import ModuleProxy
from tests.fixtures.mock_actions import (
    AsyncFailingAction,
    AsyncSuccessAction,
    ContextCheckAction,
    FailingAction,
    ForceReplanAction,
    SuccessAction,
)
from tests.fixtures.mock_controllers import MockNavController, MockArmController, MockGripperController
from tests.fixtures.mock_manifests import basic_registry


@pytest.fixture
def context_full_controllers() -> Context:
    """
    Create a Context with all mock controllers attached.

    Returns:
        Context with nav, arm, gripper controllers and some supported actions.
    """
    return Context(
        robot_id="test_robot",
        hardware={"camera": "test"},
        modules=ModuleProxy([]),
        nav=MockNavController(),
        arm=MockArmController(),
        gripper=MockGripperController(),
        supported_actions={
            "SuccessAction": SuccessAction,
            "FailingAction": FailingAction,
            "AsyncFailingAction": AsyncFailingAction,
            "ForceReplanAction": ForceReplanAction,
        },
        world_state=WorldState(max_history=10)
    )


@pytest.fixture
def context_minimal_controllers() -> Context:
    """
    Create a Context with only navigation controller.

    Returns:
        Context with only nav controller, arm and gripper are None.
    """
    return Context(
        robot_id="minimal_robot",
        hardware={},
        modules=ModuleProxy([]),
        nav=MockNavController(),
        supported_actions={},
        world_state=WorldState(max_history=10)
    )


@pytest.fixture
def context_no_controllers() -> Context:
    """
    Create a Context with no controllers attached.

    Useful for testing error handling when controllers are missing.
    """
    return Context(
        robot_id="no_ctrl_robot",
        hardware={},
        modules=ModuleProxy([]),
        supported_actions={},
        world_state=WorldState(max_history=10)
    )


@pytest.fixture
def context_with_world_state() -> Context:
    """
    Create a Context with populated world state for testing.

    Returns:
        Context with pre-populated world state.
    """
    context = context_full_controllers()

    # Populate world state with test data
    context.world_state.set("robot_position", {"x": 1.0, "y": 2.0, "theta": 0.0})
    context.world_state.set("objects", [
        {"id": "cup_1", "type": "cup", "position": [1.5, 1.0]}
    ])
    context.world_state.set("current_plan", "navigate_to_pickup")

    return context


@pytest.fixture
def context_with_modules() -> Context:
    """
    Create a Context with module proxy for testing module interactions.

    Returns:
        Context with mock modules configured.
    """
    from tests.fixtures.mock_manifests import ModuleEndpoint

    modules = [
        ModuleEndpoint(name="slam", endpoint="http://slam:8080", type="http"),
        ModuleEndpoint(name="weather", endpoint="https://api.weather.com", type="http"),
    ]

    return Context(
        robot_id="test_robot",
        hardware={"camera": "test"},
        modules=ModuleProxy(modules),
        supported_actions={},
        world_state=WorldState(max_history=10)
    )
