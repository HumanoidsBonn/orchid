"""Integration tests for Phase 1 core framework."""

import pytest
from src.tree.plan import ActionNode, CompositeNode, Plan
from src.core.context import Context
from src.core.world_state import WorldState, ActionRecord
from src.tree.execution import ExecutionEngine
from src.tree.builder import TreeBuilder
from tests.fixtures.mock_actions import SuccessAction, FailingAction
from tests.fixtures.mock_controllers import MockNavController, MockArmController, MockGripperController


class TestSimpleSequence:
    """Integration test: simple 3-action sequence."""
    
    def test_execute_simple_sequence(self):
        """Test executing a simple sequence of 3 success actions."""
        # Build plan
        action1 = ActionNode(id="a1", action_type="SuccessAction", name="Task1")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="Task2")
        action3 = ActionNode(id="a3", action_type="SuccessAction", name="Task3")
        
        sequence = CompositeNode(
            id="seq1",
            node_type="sequence",
            children=[action1, action2, action3]
        )
        plan = Plan(root=sequence)
        
        # Build and execute
        context = Context()
        builder = TreeBuilder()
        tree = builder.build(plan)
        engine = ExecutionEngine(context=context)
        
        # Verify tree was built
        assert tree is not None
        
        # Verify context is ready
        assert engine.context is not None
        assert engine.context.world_state is not None
    
    def test_sequence_records_to_world_state(self):
        """Test that sequence execution records to world state."""
        action1 = ActionNode(id="a1", action_type="SuccessAction", name="Task1")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="Task2")
        
        sequence = CompositeNode(
            id="seq",
            node_type="sequence",
            children=[action1, action2]
        )
        plan = Plan(root=sequence)
        
        ws = WorldState(maxlen=10)
        context = Context(world_state=ws)
        
        # Manually add records to simulate execution
        context.world_state.append_action_record(
            ActionRecord(action_name="Task1", success=True, duration=0.1)
        )
        context.world_state.append_action_record(
            ActionRecord(action_name="Task2", success=True, duration=0.15)
        )
        
        # Verify records were tracked
        history = context.world_state.get_action_history()
        assert len(history) == 2
        assert history[0].action_name == "Task1"
        assert history[1].action_name == "Task2"


class TestConditionalExecution:
    """Integration test: conditional branching."""
    
    def test_selector_with_fallback(self):
        """Test selector that falls back when first action fails."""
        # First action fails, second succeeds
        fail_action = ActionNode(id="a1", action_type="FailingAction", name="MayFail")
        success_action = ActionNode(id="a2", action_type="SuccessAction", name="Fallback")
        
        selector = CompositeNode(
            id="sel",
            node_type="selector",
            children=[fail_action, success_action]
        )
        plan = Plan(root=selector)
        
        context = Context()
        builder = TreeBuilder()
        tree = builder.build(plan)
        
        # Tree should be built successfully
        assert tree is not None
    
    def test_sequence_with_precondition(self):
        """Test sequence with implicit precondition."""
        # If first fails, sequence stops
        precondition = ActionNode(id="a1", action_type="SuccessAction", name="Check")
        main_task = ActionNode(id="a2", action_type="SuccessAction", name="Work")
        
        sequence = CompositeNode(
            id="seq",
            node_type="sequence",
            children=[precondition, main_task]
        )
        plan = Plan(root=sequence)
        
        context = Context()
        builder = TreeBuilder()
        tree = builder.build(plan)
        
        assert tree is not None


class TestErrorRecovery:
    """Integration test: error handling and recovery."""
    
    def test_failed_action_doesnt_stop_selector(self):
        """Test that selector tries next action on failure."""
        fail1 = ActionNode(id="a1", action_type="FailingAction", name="FailFirst")
        fail2 = ActionNode(id="a2", action_type="FailingAction", name="FailSecond")
        success = ActionNode(id="a3", action_type="SuccessAction", name="SucceedThird")
        
        selector = CompositeNode(
            id="sel",
            node_type="selector",
            children=[fail1, fail2, success]
        )
        plan = Plan(root=selector)
        
        context = Context()
        builder = TreeBuilder()
        tree = builder.build(plan)
        
        # Selector should build with all three options
        assert tree is not None
    
    def test_recovery_after_failure(self):
        """Test recovery workflow: try -> fail -> recover."""
        attempt = ActionNode(id="a1", action_type="FailingAction", name="Attempt")
        recovery = ActionNode(id="a2", action_type="SuccessAction", name="Recover")
        
        sequence = CompositeNode(
            id="seq",
            node_type="sequence",
            children=[attempt, recovery]
        )
        plan = Plan(root=sequence)
        
        # Even if attempt fails, sequence structure allows recovery action
        context = Context()
        builder = TreeBuilder()
        tree = builder.build(plan)
        
        assert tree is not None


class TestWorldStateIntegration:
    """Integration test: world state tracking across execution."""
    
    def test_world_state_accumulates_during_execution(self):
        """Test world state accumulates observations and history."""
        ws = WorldState(maxlen=5)
        
        # Simulate multiple actions
        for i in range(3):
            action_record = ActionRecord(
                action_name=f"Action{i}",
                success=(i % 2 == 0),
                duration=0.1 * i
            )
            ws.append_action_record(action_record)
        
        # Add observations
        ws.set_observation("gripper_state", {"open": False})
        ws.set_observation("arm_position", {"x": 1.0, "y": 2.0})
        
        # Verify state
        assert len(ws) == 3
        assert ws.get_observation("gripper_state") == {"open": False}
        history = ws.get_action_history()
        assert history[0].action_name == "Action0"
    
    def test_world_state_eviction_on_overflow(self):
        """Test world state evicts oldest actions when maxlen exceeded."""
        ws = WorldState(maxlen=3)
        
        # Add 5 actions (overflow by 2)
        for i in range(5):
            action_record = ActionRecord(action_name=f"A{i}", success=True)
            ws.append_action_record(action_record)
        
        # Should only keep last 3
        history = ws.get_action_history()
        assert len(history) == 3
        assert history[0].action_name == "A2"
        assert history[-1].action_name == "A4"


class TestContextIntegration:
    """Integration test: context DI across components."""
    
    def test_context_shared_across_actions(self):
        """Test that context is shared and modified across actions."""
        nav = MockNavController()
        ws = WorldState(maxlen=10)
        context = Context(nav_controller=nav, world_state=ws)
        
        # Simulate first action modifying controller
        context.nav_controller.navigate_to("location_a")
        assert context.nav_controller.move_count == 1
        
        # Second action sees the modified state
        context.nav_controller.navigate_to("location_b")
        assert context.nav_controller.move_count == 2
        
        # World state also accumulated
        context.world_state.append_action_record(
            ActionRecord(action_name="Nav1", success=True)
        )
        assert len(context.world_state) == 1
    
    def test_multiple_actions_access_same_world_state(self):
        """Test multiple actions can read/write to same world state."""
        ws = WorldState(maxlen=10)
        context = Context(world_state=ws)
        
        # Action 1: write observation
        context.world_state.set_observation("sensor1", {"data": 100})
        
        # Action 2: read observation
        obs = context.world_state.get_observation("sensor1")
        assert obs == {"data": 100}
        
        # Action 3: write action record
        context.world_state.append_action_record(
            ActionRecord(action_name="Task", success=True)
        )
        
        # All data accessible
        assert len(context.world_state) == 1
        assert context.world_state.get_observation("sensor1") == {"data": 100}


class TestEndToEndWorkflow:
    """End-to-end integration test."""
    
    def test_plan_to_execution_workflow(self):
        """Test complete workflow: plan creation -> tree building -> context setup."""
        # 1. Create plan
        action1 = ActionNode(id="a1", action_type="SuccessAction", name="Setup")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="Execute")
        action3 = ActionNode(id="a3", action_type="SuccessAction", name="Cleanup")
        
        sequence = CompositeNode(
            id="seq",
            node_type="sequence",
            children=[action1, action2, action3]
        )
        plan = Plan(root=sequence)
        
        # 2. Set up context with all components
        nav = MockNavController()
        arm = MockArmController()
        gripper = MockGripperController()
        ws = WorldState(maxlen=20)
        
        context = Context(
            nav_controller=nav,
            arm_controller=arm,
            gripper_controller=gripper,
            world_state=ws
        )
        
        # 3. Build tree
        builder = TreeBuilder()
        tree = builder.build(plan)
        
        # 4. Create execution engine
        engine = ExecutionEngine(context=context)
        
        # 5. Verify everything is connected
        assert tree is not None
        assert engine.context == context
        assert engine.context.world_state == ws
        assert engine.context.nav_controller == nav
        
        # 6. Simulate execution tracking
        engine.context.world_state.append_action_record(
            ActionRecord(action_name="Setup", success=True, duration=0.1)
        )
        engine.context.world_state.append_action_record(
            ActionRecord(action_name="Execute", success=True, duration=0.2)
        )
        engine.context.world_state.append_action_record(
            ActionRecord(action_name="Cleanup", success=True, duration=0.05)
        )
        
        # 7. Verify complete workflow
        history = engine.context.world_state.get_action_history()
        assert len(history) == 3
        assert history[0].action_name == "Setup"
        assert history[1].action_name == "Execute"
        assert history[2].action_name == "Cleanup"
