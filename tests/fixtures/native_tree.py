"""Test-owned ticker for action node contracts (production scheduling stays in engine)."""
import py_trees
from src.core.action import ActionResult
from src.tree.bt_node_wrapper import ActionScope


def action_node(action, context):
    key = id(action)
    cache = getattr(context, "_test_nodes", None)
    if cache is None:
        cache = context._test_nodes = {}
    if key not in cache:
        cache[key] = action.build_node(context, type(action).__name__)
    return cache[key]


def tick_action(action, context):
    root = action_node(action, context)
    list(root.tick())
    if context.execution.control:
        return context.execution.control.result
    if root.status == py_trees.common.Status.RUNNING:
        return ActionResult(status="running")
    for node in [root] + list(root.children):
        result = getattr(node, "result", None)
        if result is not None and getattr(node, "action", None) is action:
            return result
    return ActionResult(status={py_trees.common.Status.SUCCESS: "success", py_trees.common.Status.RUNNING: "running"}.get(root.status, "failure"))


def prepare_action(action, context):
    root = action_node(action, context)
    scope = root.decorated if isinstance(root, py_trees.decorators.Timeout) else root
    scope.initialise()
    return root


def native_actions(action, context):
    root = action_node(action, context)
    if isinstance(root, py_trees.decorators.Timeout):
        root = root.decorated
    root = root.decorated if isinstance(root, ActionScope) else root
    def flatten(node):
        if hasattr(node, "action"):
            return [node.action]
        return [item for child in node.children for item in flatten(child)]
    return [item for child in root.children for item in flatten(child)]
