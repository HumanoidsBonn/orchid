"""Unit tests for ExecutionEngine module."""

import pytest
import py_trees
from src.tree.execution import ExecutionEngine
from src.tree.plan import ActionNode, CompositeNode, Plan
from src.core.context import Context
from src.core.world_state import WorldState
from tests.fixtures.mock_actions import SuccessAction, FailingAction
from tests.fixtures.mock_controllers import MockNavController
from tests.fixtures.mock_contexts import context_full_controllers, context_minimal_controllers, context_no_controllers

class TestExecutionEngine:
    """Tests for ExecutionEngine."""
    
    def test_engine_creation(self):
        """Test ExecutionEngine initialization."""
        context = Context()
        engine = ExecutionEngine(context=context)
        
        assert engine.context == context
    
    def test_engine_with_custom_context(self, context_full_controllers):
        """Test ExecutionEngine with custom context."""
        engine = ExecutionEngine(context=context_full_controllers)
        
        assert engine.context == context_full_controllers
        assert engine.context.nav is not None
    
    def test_execute_simple_success_action(self, context_full_controllers):
        """Test executing a simple single-action plan."""
        action_node = ActionNode(
            id="a1",
            action_type="SuccessAction",
            name="TestAction"
        )
        plan = Plan(root=action_node)        
        engine = ExecutionEngine(context=context_full_controllers)

        result = engine._execute_plan(plan)        
        assert result is True   
        assert engine.context is not None

    def test_execute_simple_failing_action(self, context_full_controllers):
        """Test executing a simple single-action plan."""
        action_node = ActionNode(
            id="a1",
            action_type="FailingAction",
            name="TestAction"
        )
        plan = Plan(root=action_node)        
        engine = ExecutionEngine(context=context_full_controllers)

        result = engine._execute_plan(plan)        
        assert result is False   
        assert engine.context is not None

    def test_terminal_async_failure_is_not_cancelled(self, context_full_controllers):
        """A completed async failure should not trigger a second cancel call."""
        action_node = ActionNode(
            id="a1",
            action_type="AsyncFailingAction",
            name="AsyncFailingAction",
        )
        plan = Plan(root=action_node)
        engine = ExecutionEngine(context=context_full_controllers)

        result = engine._execute_plan(plan)

        assert result is False
        assert engine.tree.root.action.cancel_called is False
    
    def test_engine_max_ticks_safeguard(self):
        """Test that engine has max_ticks safeguard."""
        context = Context()
        engine = ExecutionEngine(context=context)
        
        # Verify max_ticks is set (prevents infinite loops)
        assert hasattr(engine, 'max_ticks') or hasattr(engine, '_max_ticks')
    
    def test_engine_execution_state(self):
        """Test engine maintains execution state."""
        context = Context()
        engine = ExecutionEngine(context=context)
        
        # Engine should be able to track state
        # modify engine to expose state for testing if needed
        assert engine.context is not None
        # assert engine.get_execution_history() is not None
    
    def test_engine_multiple_plans(self, context_full_controllers):
        """Test engine can handle multiple plans sequentially."""
        engine = ExecutionEngine(context=context_full_controllers)
        
        # Create two simple plans
        ac1 = ActionNode(id="a1", action_type="SuccessAction", name="TestAction1")
        plan1 = Plan(root=ac1)
        ac2 = ActionNode(id="a2", action_type="FailingAction", name="TestAction2")
        plan2 = Plan(root=ac2)
        
        res1 = engine._execute_plan(plan1)
        res2 = engine._execute_plan(plan2)
        # Engine should support multiple plans
        assert engine is not None
        assert res1 is True
        assert res2 is False

    def test_force_replan_stops_fallback_before_next_branch(self, context_full_controllers):
        """A force_replan result ends the run even though the selector has untried children."""
        root = CompositeNode(
            id="root",
            type="fallback",
            name="Search",
            children=[
                ActionNode(id="a1", action_type="ForceReplanAction", name="Inspect"),
                ActionNode(id="a2", action_type="SuccessAction", name="NextFurniture"),
            ],
        )
        engine = ExecutionEngine(context=context_full_controllers)

        result = engine._execute_plan(Plan(root=root))

        assert result is False
        last = context_full_controllers.world_state.get("last_action_result")
        assert last["action_name"] == "Inspect"
        assert last["status"] == "needs_replan"
        assert last["force_replan"] is True
        history = [r.action_name for r in context_full_controllers.world_state.get_action_history()]
        assert "NextFurniture" not in history

    def test_stale_force_replan_does_not_stop_next_plan(self, context_full_controllers):
        """last_action_result from a previous plan must not short-circuit the next one."""
        engine = ExecutionEngine(context=context_full_controllers)
        engine._execute_plan(Plan(root=ActionNode(id="a1", action_type="ForceReplanAction", name="Inspect")))
        assert context_full_controllers.world_state.get("last_action_result")["force_replan"] is True

        result = engine._execute_plan(Plan(root=ActionNode(id="a2", action_type="SuccessAction", name="Grasp")))

        assert result is True
        assert context_full_controllers.world_state.get("last_action_result")["action_name"] == "Grasp"

    def test_serialize_tree_state_includes_action_timing_metadata(self, context_full_controllers):
        """Behavior-tree UI payload includes optional action timing fields."""
        action_node = ActionNode(
            id="a1",
            action_type="SuccessAction",
            name="NamedSuccess",
        )
        plan = Plan(root=action_node)
        engine = ExecutionEngine(context=context_full_controllers)

        assert engine._execute_plan(plan) is True
        payload = engine._serialize_tree_state(engine.tree.root)

        assert payload["id"] == str(engine.tree.root.id)
        assert engine._serialize_tree_state(engine.tree.root)["id"] == payload["id"]
        assert payload["name"] == "NamedSuccess"
        assert payload["node_type"] == "BtActionLeaf"
        assert payload["action_type"] == "SuccessAction"
        assert payload["display_name"] == "NamedSuccess"
        assert payload["duration_s"] >= 0.0
        history = context_full_controllers.world_state.get_action_history()
        assert history[-1].bt_node_name == "NamedSuccess"
        assert history[-1].action_class == "SuccessAction"
        assert history[-1].action_type == "SuccessAction"

    def test_serialize_tree_state_attaches_matching_current_operation(self, context_full_controllers):
        """Active composite progress is exposed on the matching BT action leaf."""
        action_node = ActionNode(
            id="a1",
            action_type="SuccessAction",
            name="NamedSuccess",
        )
        plan = Plan(root=action_node)
        engine = ExecutionEngine(context=context_full_controllers)
        tree = engine.build_tree(plan)
        tree.root.status = py_trees.common.Status.RUNNING
        tree.root._monotonic_start_time = 1.0
        context_full_controllers.world_state.set(
            "current_operation",
            {
                "sequence_label": "SuccessAction",
                "current_action_name": "ChildStep",
                "current_action_index": 1,
                "total_actions": 3,
                "elapsed_s": 2.5,
                "current_action_elapsed_s": 0.4,
                "status": "running",
            },
        )

        payload = engine._serialize_tree_state(tree.root)

        assert payload["operation"]["current_action_name"] == "ChildStep"
        assert payload["operation"]["total_actions"] == 3
        assert payload["elapsed_s"] == pytest.approx(2.5)

class TestExecutionEngineIntegration:
    """Integration-style tests for ExecutionEngine."""
    
    def test_engine_with_world_state(self):
        """Test engine interaction with WorldState."""
        ws = WorldState(max_history=10)
        context = Context(world_state=ws)
        engine = ExecutionEngine(context=context)
        
        assert engine.context.world_state == ws
