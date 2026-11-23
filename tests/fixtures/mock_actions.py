"""Mock action implementations for testing."""

from dataclasses import dataclass

from src.core.action import AbstractAction, AsyncAction, ActionResult
from src.core.context import Context


@dataclass
class SuccessAction(AbstractAction):
    """Action that always succeeds."""

    name: str = "SuccessAction"

    def __post_init__(self) -> None:
        AbstractAction.__init__(self)

    def execute(self, context: Context) -> ActionResult:
        return ActionResult(status="success", reason="Action completed successfully")


@dataclass
class FailingAction(AbstractAction):
    """Action that always fails."""

    name: str = "FailingAction"

    def __post_init__(self) -> None:
        AbstractAction.__init__(self)

    def execute(self, context: Context) -> ActionResult:
        return ActionResult(status="failure", reason="Intentional failure for testing")


@dataclass
class ForceReplanAction(AbstractAction):
    """Action that asks the executor to stop the tree and replan immediately."""

    name: str = "ForceReplanAction"

    def __post_init__(self) -> None:
        AbstractAction.__init__(self)

    def execute(self, context: Context) -> ActionResult:
        return ActionResult(
            status="needs_replan",
            reason="Found something, replan now",
            suggested_replan_context={"found_label": "thing"},
            force_replan=True,
        )


class AsyncSuccessAction(AsyncAction):
    """Async action that always succeeds."""

    def __init__(self, name: str = "AsyncSuccessAction", delay: float = 0.01) -> None:
        super().__init__()
        self.name = name
        self.delay = delay

    def start(self, context: Context) -> None:
        return None

    def check_result(self, context: Context) -> ActionResult:
        return ActionResult(status="success", reason="Async action completed")

    def cancel(self, context):
        self._cancel_requested = True


class AsyncFailingAction(AsyncAction):
    """Async action that returns a terminal failure without needing cancellation."""

    def __init__(self, name: str = "AsyncFailingAction") -> None:
        super().__init__()
        self.name = name
        self.cancel_called = False

    def start(self, context: Context) -> None:
        return None

    def check_result(self, context: Context) -> ActionResult:
        return ActionResult(status="failure", reason="Async action failed")

    def cancel(self, context: Context) -> None:
        self.cancel_called = True


class ContextCheckAction(AbstractAction):
    """Action that verifies context is properly injected."""

    def __init__(self, name: str = "ContextCheckAction") -> None:
        super().__init__()
        self.name = name
        self.name = name

    def execute(self, context: Context) -> ActionResult:
        if context is None:
            return ActionResult(status="failure", reason="No context provided")
        try:
            ws = context.world_state
            return ActionResult(
                status="success",
                reason="Context valid",
                payload={"has_world_state": ws is not None},
            )
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failure", reason=str(exc))


class NavigationAction(AsyncAction):
    def cancel(self, context): pass
    """Legacy test stub."""

    def start(self, context: Context) -> None:
        return None

    def check_result(self, context: Context) -> ActionResult:
        return ActionResult(status="success")


class ObjectPickAction(AsyncAction):
    def cancel(self, context): pass
    """Legacy test stub."""

    def start(self, context: Context) -> None:
        return None

    def check_result(self, context: Context) -> ActionResult:
        return ActionResult(status="success")


class PerceptionAction(AbstractAction):
    """Legacy test stub."""

    def execute(self, context: Context) -> ActionResult:
        return ActionResult(status="success")
