"""Unit tests for Context module."""

import asyncio
import threading

import pytest
from src.core.context import Context
from src.core.world_state import WorldState
from src.core.module_proxy import ModuleProxy
from tests.fixtures.mock_controllers import MockNavController, MockArmController, MockGripperController


class TestContextCreation:
    """Tests for Context creation and initialization."""
    
    def test_context_creation_minimal(self):
        """Test Context creation with no arguments."""
        context = Context()
        assert context.nav is None
        assert context.arm is None
        assert context.gripper is None
    
    def test_context_with_controllers(self):
        """Test Context creation with controllers."""
        nav = MockNavController()
        arm = MockArmController()
        gripper = MockGripperController()
        
        context = Context(nav=nav, arm=arm, gripper=gripper)
        assert context.nav == nav
        assert context.arm == arm
        assert context.gripper == gripper
    
    def test_context_default_modules_created(self):
        """Test that Context creates default module proxy."""
        context = Context()
        assert context.modules is not None
        assert isinstance(context.modules, ModuleProxy)
    
    def test_context_default_world_state_created(self):
        """Test that Context creates default world state."""
        context = Context()
        assert context.world_state is not None
        assert isinstance(context.world_state, WorldState)


class TestContextRequireMethods:
    """Tests for Context require_* methods."""
    
    def test_require_nav_success(self):
        """Test require_nav when controller available."""
        nav = MockNavController()
        context = Context(nav=nav)
        
        assert context.require_nav() == nav
    
    def test_require_nav_failure(self):
        """Test require_nav raises when controller unavailable."""
        context = Context()
        
        with pytest.raises(RuntimeError, match="Navigation controller not available"):
            context.require_nav()
    
    def test_require_arm_success(self):
        """Test require_arm when controller available."""
        arm = MockArmController()
        context = Context(arm=arm)
        
        assert context.require_arm() == arm
    
    def test_require_arm_failure(self):
        """Test require_arm raises when controller unavailable."""
        context = Context()
        
        with pytest.raises(RuntimeError, match="Arm controller not available"):
            context.require_arm()
    
    def test_require_gripper_success(self):
        """Test require_gripper when controller available."""
        gripper = MockGripperController()
        context = Context(gripper=gripper)
        
        assert context.require_gripper() == gripper
    
    def test_require_gripper_failure(self):
        """Test require_gripper raises when controller unavailable."""
        context = Context()
        
        with pytest.raises(RuntimeError, match="Gripper controller not available"):
            context.require_gripper()

    def test_require_head_success(self):
        """Test require_head when controller available."""
        head = object()
        context = Context(head=head)

        assert context.require_head() is head

    def test_require_head_failure(self):
        """Test require_head raises when controller unavailable."""
        context = Context()

        with pytest.raises(RuntimeError, match="Head controller not available"):
            context.require_head()


class TestContextState:
    """Tests for Context state management."""
    
    def test_context_world_state_modification(self):
        """Test that world state can be modified through context."""
        context = Context()
        
        context.world_state.set("test_key", {"value": 123})
        assert context.world_state.get("test_key") == {"value": 123}
    
    def test_context_preserves_controller_state(self):
        """Test that controller state changes persist."""
        nav = MockNavController()
        context = Context(nav=nav)
        
        initial_count = context.nav.move_count
        context.nav.send_goal((1.0, 2.0, 0.0))
        
        assert context.nav.move_count > initial_count


class TestContextAsyncLoop:
    """Tests for context-owned async loop scheduling."""

    def test_run_in_context_loop_supports_reentrant_coroutines(self):
        """Nested loop use inside a context coroutine should remain supported."""
        context = Context(robot_id="nested-test")
        done = threading.Event()
        results = []

        async def outer():
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.sleep(0))
            context.run_in_context_loop(
                lambda: "inner",
                callback=lambda result: (results.append(result), done.set()),
            )
            return "outer"

        context.run_in_context_loop(outer)

        assert done.wait(timeout=1.0)
        assert results == ["inner"]

    def test_run_coroutine_reports_callback_exception(self):
        """Callback exceptions should be forwarded to the action owner."""
        context = Context(robot_id="callback-error-test")
        done = threading.Event()
        errors = []

        async def task():
            return "ok"

        def callback(_result):
            raise RuntimeError("callback exploded")

        def error_callback(exc):
            errors.append(str(exc))
            done.set()

        context.run_coroutine(task(), callback=callback, error_callback=error_callback)

        assert done.wait(timeout=1.0)
        assert errors == ["callback exploded"]

    def test_run_in_context_loop_reports_task_exception(self):
        """Async function exceptions should be forwarded to error_callback."""
        context = Context(robot_id="task-error-test")
        done = threading.Event()
        errors = []

        async def task():
            raise RuntimeError("task exploded")

        context.run_in_context_loop(
            task,
            error_callback=lambda exc: (errors.append(str(exc)), done.set()),
        )

        assert done.wait(timeout=1.0)
        assert errors == ["task exploded"]
