"""Execution bindings and first-wins control signals for a single mission tree."""
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class ActionBinding:
    action: Any
    name: str
    action_type: str
    planner_visible: bool = False
    result: Any = None


@dataclass(frozen=True)
class ExecutionControl:
    node_id: UUID
    result: Any


@dataclass
class ExecutionRuntime:
    bindings: dict[UUID, ActionBinding] = field(default_factory=dict)
    stage_labels: dict[UUID, str] = field(default_factory=dict)
    control: ExecutionControl | None = None
    cancel_requested: bool = False

    def active_stage(self, root):
        """Deepest running authored stage; equal-depth parallel ties use tree order."""
        selected, selected_depth = None, -1
        def visit(node, depth):
            nonlocal selected, selected_depth
            if getattr(node.status, 'value', None) != 'RUNNING':
                return
            if node.id in self.stage_labels and depth > selected_depth:
                selected, selected_depth = self.stage_labels[node.id], depth
            for child in node.children:
                visit(child, depth + 1)
        visit(root, 0)
        return selected

    def bind(self, node, action, *, action_type=None, planner_visible=False):
        self.bindings[node.id] = ActionBinding(action, node.name, action_type or type(action).__name__, planner_visible)

    def request_control(self, node, result):
        if self.control is None and (result.force_replan or result.status == "needs_user_input"):
            self.control = ExecutionControl(node.id, result)

    def reset_control(self):
        self.control = None
        self.cancel_requested = False
