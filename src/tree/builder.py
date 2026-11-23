"""TreeBuilder: converts Plan (JSON-like structure) to executable py_trees behavior tree.

Handles:
- Recursive conversion of PlanNode types (Action, Condition, Composite, Decorator)
- Attaching the native node exposed by each validated action
- Building composite nodes (Sequence, Fallback)
- Applying decorators (Retry, Timeout, Inverter)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import py_trees

if TYPE_CHECKING:
    from src.plugin.registry import PluginRegistry
    from src.core.context import Context
    from src.tree.plan import ActionNode, ConditionNode, CompositeNode, DecoratorNode, AbstractPlanNode

from src.tree.bt_node_wrapper import BtActionLeaf, BtConditionLeaf, retry_eligible

import src.core.logger as log


class TreeBuilder:
    """Converts a Plan to a py_trees BehaviorTree."""

    def __init__(self, context: Context):
        """Initialize TreeBuilder.

        Args:
            context: Context to inject into actions
        """
        self.context = context

    def build(self, plan) -> py_trees.trees.BehaviourTree:
        """Build a complete behavior tree from a Plan.

        Args:
            plan: Plan object with goal and root PlanNode

        Returns:
            py_trees.trees.BehaviourTree rooted at the converted tree
        """
        root = self._build_node(plan.root)
        log.debug("TreeBuilder.build complete: root=%s", root.name)
        from src.tree.binding_visitor import ActionBindingsVisitor
        tree = py_trees.trees.BehaviourTree(root=root)
        tree.visitors.append(ActionBindingsVisitor(self.context))
        return tree

    def _build_node(self, node: PlanNode) -> py_trees.behaviour.Behaviour:
        """Recursively convert a PlanNode to a py_trees Behaviour.

        Args:
            node: PlanNode to convert (ActionNode, ConditionNode, CompositeNode, DecoratorNode)

        Returns:
            Corresponding py_trees Behaviour

        Raises:
            ValueError: If action/condition type not found in registry
        """
        # Import here to avoid circular dependencies
        from src.tree.plan import (
            ActionNode,
            ConditionNode,
            CompositeNode,
            DecoratorNode,
        )

        if isinstance(node, ActionNode):
            return self._build_action_node(node)
        elif isinstance(node, ConditionNode):
            return self._build_condition_node(node)
        elif isinstance(node, CompositeNode):
            return self._build_composite_node(node)
        elif isinstance(node, DecoratorNode):
            return self._build_decorator_node(node)
        else:
            raise ValueError(f"Unknown PlanNode type: {type(node)}")

    def _build_action_node(self, node: ActionNode) -> py_trees.behaviour.Behaviour:
        """Convert ActionNode to BtActionLeaf.

        Args:
            node: ActionNode

        Returns:
            BtActionLeaf wrapping the action instance

        Raises:
            ValueError: If action class not found in registry
        """
        action_cls = self.context.require_action_type(node.action_type)
        if not action_cls:
            raise ValueError(f"Action '{node.action_type}' for action {node.name} not found in registry")

        # Instantiate action with params from plan
        action = action_cls.from_params(**node.params)
        leaf = action.build_node(self.context, node.name)
        self.context.execution.bind(leaf, action, action_type=node.action_type, planner_visible=True)
        if hasattr(leaf, "binding"):
            leaf.binding = self.context.execution.bindings[leaf.id]
        log.debug("Built ActionNode %s: %s -> %s", node.name, node.action_type, leaf.name)
        return leaf

    def _build_condition_node(self, node: ConditionNode) -> py_trees.behaviour.Behaviour:
        """Convert ConditionNode to BtConditionLeaf.

        Args:
            node: ConditionNode

        Returns:
            BtConditionLeaf

        Raises:
            ValueError: If condition class not found in registry
        """
        condition_cls = self.context.require_condition_type(node.name)
        if not condition_cls:
            raise ValueError(f"Condition '{node.name}' not found in registry")

        from src.core.inputs import instantiate_inputs
        condition = instantiate_inputs(condition_cls, node.params)
        leaf = BtConditionLeaf(condition, self.context, name=node.name)
        log.debug("Built ConditionNode: %s -> %s", node.name, leaf.name)
        return leaf

    def _build_composite_node(self, node: CompositeNode) -> py_trees.behaviour.Behaviour:
        """Convert CompositeNode (Sequence or Fallback) to py_trees composite.

        Args:
            node: CompositeNode with type 'sequence' or 'fallback'

        Returns:
            py_trees.composites.Sequence or Selector

        Raises:
            ValueError: If node.type not 'sequence' or 'fallback'
        """
        # Build all children recursively
        children = [self._build_node(child) for child in node.children]

        if node.type == "sequence":
            composite = py_trees.composites.Sequence(name=node.name, memory=True)
        elif node.type == "fallback" or node.type == "selector":  # allow "selector" as alias for "fallback"
            composite = py_trees.composites.Selector(name=node.name, memory=True)
        elif node.type == "parallel":
            composite = py_trees.composites.Parallel(name=node.name, policy=py_trees.common.ParallelPolicy.SuccessOnAll())
        else:
            raise ValueError(f"Unknown CompositeNode type: {node.type}")

        for child in children:
            composite.add_child(child)

        log.debug("Built CompositeNode: %s (%s) with %s children", node.name, node.type, len(children))
        return composite

    def _build_decorator_node(self, node: DecoratorNode) -> py_trees.behaviour.Behaviour:
        """Convert DecoratorNode (Retry, Timeout, Inverter) to py_trees decorator.

        Args:
            node: DecoratorNode with type 'retry', 'timeout', or 'inverter'

        Returns:
            Decorated py_trees behaviour

        Raises:
            ValueError: If node.type not recognized
        """
        child = self._build_node(node.child)

        if node.type == "retry":
            num_attempts = node.params.get("num_attempts", 3)
            retries = py_trees.decorators.Retry(name=node.name, child=child, num_failures=num_attempts)
            decorated = py_trees.decorators.EternalGuard(name=f"{node.name} retry eligibility", child=retries,
                condition=lambda: retry_eligible(self.context, child, retries))
        elif node.type == "timeout":
            duration = node.params.get("duration", 10.0)
            decorated = py_trees.decorators.Timeout(
                name=node.name, duration=duration, child=child
            )
        elif node.type == "inverter":
            decorated = py_trees.decorators.Inverter(name=node.name, child=child)
        else:
            raise ValueError(f"Unknown DecoratorNode type: {node.type}")

        log.debug("Built DecoratorNode: %s (%s)", node.name, node.type)
        return decorated
