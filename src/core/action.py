"""Core action definitions for the modular manipulation framework.

This module defines the base classes used by all concrete actions:

* ``ActionResult`` – a lightweight data container describing the outcome of an
  action. It uses a ``Literal`` status field so callers can safely match on the
  possible values.
* ``AbstractAction`` – an action definition exposing a native node through
  ``build_node``. Executable leaves implement ``execute``; workflows define
  native topology and optional invocation lifecycle hooks.
* ``AsyncAction`` – a helper base for actions that need to start a long‑running
  operation (e.g., sending a navigation goal) and then poll for completion on
  subsequent ticks. It implements ``execute`` by delegating to ``start`` on the
  first tick and ``check_result`` on later ticks.

The design mirrors the architecture diagram you shared and will be used by the
``BtActionLeaf`` wrapper to integrate actions into a ``py_trees`` behaviour tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.context import Context
from src.core.failure import FailureInfo
import src.core.logger as log

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Result of an ``AbstractAction`` execution.

    Attributes
    ----------
    status:
        One of ``"success"``, ``"failure"``, ``"running"``,
        ``"needs_replan"`` or ``"needs_user_input"``.
    reason:
        Optional human‑readable explanation for failure or replanning.
    suggested_replan_context:
        Optional dict that can be passed back to the planner when a replanning
        request is issued.
    payload:
        Arbitrary data produced by the action (e.g., detected object info).
    force_replan:
        When true, the executor stops ticking the tree as soon as this result
        is recorded and hands it to the session for replanning, even if the
        tree could continue (e.g. a fallback with untried branches).
    """

    status: Literal[
        "success",
        "failure",
        "running",
        "needs_replan",
        "needs_user_input",
    ]
    reason: str | None = None
    suggested_replan_context: dict | None = None
    payload: dict = field(default_factory=dict)
    force_replan: bool = False
    failure: FailureInfo | None = None

    def __post_init__(self):
        if self.status == "failure" and self.failure is None:
            self.failure = FailureInfo()


# ---------------------------------------------------------------------------
# Base action classes
# ---------------------------------------------------------------------------

class AbstractAction(ABC):
    """Action definition. Executable leaves override execute; workflows build nodes."""

    required_controllers: list[str] = []
    required_modules: list[str] = []
    input_attributes: dict[str, str] = {}

    def __init__(self) -> None:
        pass

    @classmethod
    def from_params(cls, **params: Any):
        if "composition_backend" in params:
            raise ValueError("composition_backend was removed; actions expose native BT nodes")
        from src.core.inputs import instantiate_inputs
        return instantiate_inputs(cls, params)

    def param_dict(self) -> dict[str, Any]:
        import inspect
        return {name: getattr(self, self.input_attributes.get(name, name)) for name, parameter in inspect.signature(type(self)).parameters.items()
                if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL) and hasattr(self, self.input_attributes.get(name, name))}

    def build_node(self, context: Context, name: str):
        from src.tree.bt_node_wrapper import BtActionLeaf
        if type(self).execute is AbstractAction.execute:
            raise TypeError(f"{type(self).__name__} must implement execute or build_node")
        return BtActionLeaf(self, context, name)

    def execute(self, context: Context) -> ActionResult:
        raise NotImplementedError("Workflow actions expose a node; they cannot be executed as leaves")

    def pre_condition(self, context: Context) -> bool:
        return True

    def post_condition(self, context: Context) -> bool:
        return True

    def on_enter(self, context: Context) -> ActionResult | None:
        """Prepare invocation data before a workflow's children are visited."""

    def on_complete(self, context: Context, status) -> ActionResult:
        import py_trees
        return ActionResult(status="success" if status == py_trees.common.Status.SUCCESS else "failure")

    def on_cancel(self, context: Context) -> None:
        """Release local resources without dispatching recovery motions."""


class AsyncAction(AbstractAction):
    """Leaf lifecycle: dispatch once, poll on subsequent visits, cancel on preemption."""

    def __init__(self) -> None:
        super().__init__()
        self._started = False
        self._cancel_requested = False

    @abstractmethod
    def start(self, context: Context) -> None:
        pass

    @abstractmethod
    def check_result(self, context: Context) -> ActionResult:
        pass

    def execute(self, context: Context) -> ActionResult:
        if not self._started:
            self.start(context)
            self._started = True
        return self.check_result(context)

    def reset(self) -> None:
        self._started = False
        self._cancel_requested = False

    @abstractmethod
    def cancel(self, context: Context) -> None:
        pass
