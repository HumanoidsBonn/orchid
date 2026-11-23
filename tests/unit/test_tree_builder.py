"""Unit tests for TreeBuilder module."""

from pydantic import ValidationError
import pytest
from src.tree.plan import (
    ActionNode, ConditionNode, CompositeNode, DecoratorNode, Plan
)
from src.tree.builder import TreeBuilder
from tests.fixtures.mock_actions import SuccessAction, FailingAction
from tests.fixtures.mock_contexts import context_full_controllers, context_minimal_controllers, context_no_controllers
import py_trees


class TestTreeBuilder:
    """Tests for TreeBuilder."""

    def test_treebuilder_creation(self, context_full_controllers):
        """Test TreeBuilder initialization with context."""
        builder = TreeBuilder(context=context_full_controllers)
        assert builder is not None
        assert builder.context == context_full_controllers

    def test_build_single_action(self, context_full_controllers):
        """Test building tree from single action node."""
        action_node = ActionNode(
            id="a1",
            action_type="SuccessAction",
            name="SuccessAction"
        )
        plan = Plan(root=action_node)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)

        assert tree is not None
        assert isinstance(tree, py_trees.trees.BehaviourTree)

    def test_build_sequence(self, context_full_controllers):
        """Test building sequence of actions."""
        action1 = ActionNode(id="a1", action_type="SuccessAction", name="SuccessAction")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="SuccessAction")

        composite = CompositeNode(
            id="seq1",
            type="sequence",
            children=[action1, action2]
        )
        plan = Plan(root=composite)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))

        assert isinstance(tree.root, py_trees.composites.Sequence)
        assert len(tree.root.children) == 2

    def test_build_selector(self, context_full_controllers):
        """Test building selector (fallback) node."""
        action1 = ActionNode(id="a1", action_type="FailingAction", name="Action1")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="Action2")

        composite = CompositeNode(
            id="sel1",
            type="fallback",
            children=[action1, action2]
        )
        plan = Plan(root=composite)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))

        assert isinstance(tree.root, py_trees.composites.Selector)
        assert len(tree.root.children) == 2

    def test_build_parallel(self, context_full_controllers):
        """Test building parallel node."""
        action1 = ActionNode(id="a1", action_type="SuccessAction", name="Action1")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="Action2")

        composite = CompositeNode(
            id="par1",
            type="parallel",
            children=[action1, action2]
        )
        plan = Plan(root=composite)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))
    
        assert isinstance(tree.root, py_trees.composites.Parallel)

    def test_build_with_decorator(self, context_full_controllers):
        """Test building node with retry decorator."""
        action = ActionNode(id="a1", action_type="SuccessAction", name="Action1")

        decorator = DecoratorNode(
            id="d1",
            type="retry",
            child=action,
            parameters={"num_attempts": 3}
        )
        plan = Plan(root=decorator)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))
        
        assert tree is not None
        assert isinstance(tree.root, py_trees.decorators.EternalGuard)
        assert isinstance(tree.root.decorated, py_trees.decorators.Retry)

    def test_build_nested_sequence(self, context_full_controllers):
        """Test building nested sequence."""
        action1 = ActionNode(id="a1", action_type="SuccessAction", name="A1")
        action2 = ActionNode(id="a2", action_type="SuccessAction", name="A2")

        inner_seq = CompositeNode(
            id="inner",
            type="sequence",
            children=[action1, action2]
        )

        action3 = ActionNode(id="a3", action_type="SuccessAction", name="A3")
        outer_seq = CompositeNode(
            id="outer",
            type="sequence",
            children=[inner_seq, action3]
        )

        plan = Plan(root=outer_seq)
        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))

        assert isinstance(tree.root, py_trees.composites.Sequence)
        assert len(tree.root.children) == 2

    def test_build_complex_tree(self, context_full_controllers):
        """Test building a more complex tree structure."""
        success_a1 = ActionNode(id="a1", action_type="SuccessAction", name="Success1")
        fail_a2 = ActionNode(id="a2", action_type="FailingAction", name="Fail1")
        success_a3 = ActionNode(id="a3", action_type="SuccessAction", name="Success2")

        selector = CompositeNode(
            id="sel",
            type="fallback",
            children=[fail_a2, success_a3]
        )

        sequence = CompositeNode(
            id="seq",
            type="sequence",
            children=[success_a1, selector]
        )

        plan = Plan(root=sequence)
        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))

        assert isinstance(tree.root, py_trees.composites.Sequence)
        assert len(tree.root.children) >= 1

    def test_build_inverter_decorator(self, context_full_controllers):
        """Test building node with inverter decorator."""
        action = ActionNode(id="a1", action_type="FailingAction", name="Action1")

        decorator = DecoratorNode(
            id="d1",
            type="inverter",
            child=action
        )
        plan = Plan(root=decorator)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))

        assert tree is not None
        assert isinstance(tree.root, py_trees.decorators.Inverter)

    def test_build_timeout_decorator(self, context_full_controllers):
        """Test building node with timeout decorator."""
        action = ActionNode(id="a1", action_type="SuccessAction", name="Action1")

        decorator = DecoratorNode(
            id="d1",
            type="timeout",
            child=action,
            parameters={"duration": 5.0}
        )
        plan = Plan(root=decorator)

        builder = TreeBuilder(context=context_full_controllers)
        tree = builder.build(plan)
        print("\n" + py_trees.display.unicode_tree(tree.root))

        assert tree is not None
        assert isinstance(tree.root, py_trees.decorators.Timeout)

    def test_missing_action_type_raises(self, context_no_controllers):
        """Test that missing action type raises ValueError."""
        action = ActionNode(id="a1", action_type="NonExistentAction", name="Action1")
        plan = Plan(root=action)

        builder = TreeBuilder(context=context_no_controllers)

        with pytest.raises(RuntimeError, match="not supported"):
            builder.build(plan)

    def test_unknown_node_type_raises(self, context_full_controllers):
        """Test that unknown decorator type raises ValueError."""
        action = ActionNode(id="a1", action_type="SuccessAction", name="Action1")
        
        with pytest.raises(ValidationError, match="validation error"): 
            decorator = DecoratorNode(
                id="bad",
                type="unknown_decorator",
                child=action
            )
        
        with pytest.raises(ValidationError, match="validation error"): 
            composite = CompositeNode(
                id="bad",
                type="unknown_type",
                children=[action]
            )