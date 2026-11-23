"""Context: Dependency injection container for action execution.

Provides all resources an action needs:
- Controllers (nav, arm, gripper)
- Module proxy (for external service calls)
- World state (observations and execution history)
- Sensor data center (for sensor subscriptions and TF2 transforms)
- Robot metadata
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from inspect import isawaitable, iscoroutinefunction
from typing import TYPE_CHECKING, Any, Callable, Optional

import nest_asyncio
import src.core.logger as log
from src.tree.runtime import ExecutionRuntime

if TYPE_CHECKING:
    from src.core.controllers import (
        BaseNavController,
        BaseArmController,
        BaseHeadController,
        BaseGripperController,
    )
    from src.core.module_proxy import ModuleProxy
    from src.core.world_state import WorldState
    from src.core.sensor_data_center import SensorDataCenter
    from src.core.scene_graph import SceneGraph


@dataclass
class Context:
    """Dependency injection container for actions and execution engine.

    Holds references to all resources needed by actions:
    - Controllers for hardware abstraction (nav, arm, gripper)
    - ModuleProxy for external module calls (HTTP, ROS2 services, etc.)
    - WorldState for observations and execution history
    - SensorDataCenter for sensor subscriptions and TF2 transforms
    - Robot metadata (ID, hardware constraints)
    """

    nav: BaseNavController | None = None
    arm: BaseArmController | None = None
    head: BaseHeadController | None = None
    gripper: BaseGripperController | None = None
    planners: dict[str, Any] = field(default_factory=dict)
    modules: ModuleProxy | None = None
    world_state: WorldState | None = None
    sensor_data_center: Optional[SensorDataCenter] = None
    scene_graph: SceneGraph | None = None
    robot_id: str = "unknown"
    hardware: dict = field(default_factory=dict)
    supported_actions: dict[str, type] = field(default_factory=dict)
    supported_conditions: dict[str, type] = field(default_factory=dict)
    request_user_input: Optional[Callable[..., str]] = None
    ros2_node: None = None
    execution: ExecutionRuntime = field(default_factory=ExecutionRuntime)
    perception_preview_publishers: dict[str, Any] = field(default_factory=dict, repr=False)
    _async_loop: Optional[asyncio.AbstractEventLoop] = field(default=None, init=False, repr=False)
    _async_loop_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize module proxy and world state if not provided."""
        if self.modules is None:
            from src.core.module_proxy import ModuleProxy

            self.modules = ModuleProxy()

        if self.world_state is None:
            from src.core.world_state import WorldState

            self.world_state = WorldState(max_history=10)

    def _ensure_async_loop(self) -> asyncio.AbstractEventLoop:
        if self._async_loop is not None and self._async_loop.is_running():
            return self._async_loop

        self._async_loop = asyncio.new_event_loop()
        nest_asyncio.apply(self._async_loop)

        self._async_loop_thread = threading.Thread(
            target=self._run_async_loop,
            name=f"context-async-loop-{self.robot_id}",
            daemon=True,
        )
        self._async_loop_thread.start()
        deadline = time.time() + 1.0
        while not self._async_loop.is_running() and time.time() < deadline:
            time.sleep(0.001)
        return self._async_loop

    def _run_async_loop(self) -> None:
        asyncio.set_event_loop(self._async_loop)
        assert self._async_loop is not None
        self._async_loop.run_forever()

    def run_coroutine(
        self,
        coro: Any,
        *,
        callback: Optional[Callable[[Any], Any]] = None,
        error_callback: Optional[Callable[[Exception], Any]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        loop = self._ensure_async_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)

        def _on_done(task: asyncio.Future[Any]) -> None:
            try:
                result = task.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                log.exception("Context async task failed: %s", exc)
                if error_callback is not None:
                    try:
                        error_callback(exc)
                    except Exception:  # noqa: BLE001
                        log.exception("Context async error callback failed")
                return
            if callback is not None:
                try:
                    callback(result)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Context async callback failed: %s", exc)
                    if error_callback is not None:
                        try:
                            error_callback(exc)
                        except Exception:  # noqa: BLE001
                            log.exception("Context async error callback failed")

        future.add_done_callback(_on_done)

    def run_in_context_loop(
        self,
        func: Callable[..., Any],
        *args: Any,
        callback: Optional[Callable[[Any], Any]] = None,
        error_callback: Optional[Callable[[Exception], Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        async def _runner() -> Any:
            if iscoroutinefunction(func):
                return await func(*args, **kwargs)
            result = await asyncio.to_thread(func, *args, **kwargs)
            if isawaitable(result):
                return await result
            return result

        self.run_coroutine(
            _runner(),
            callback=callback,
            error_callback=error_callback,
            timeout=timeout,
        )

    def require_action_type(self, action_name: str) -> type:
        """Get action class by name, raising if unsupported.

        Args:
            action_name: Name of the action to retrieve

        Returns:
            Action class type

        Raises:
            RuntimeError: If action not supported by this robot
        """
        if action_name not in self.supported_actions:
            raise RuntimeError(f"Action '{action_name}' not supported by this robot")
        return self.supported_actions[action_name]
    
    def require_condition_type(self, condition_name: str) -> type:
        """Resolve a condition declared by this robot's manifest."""
        if condition_name not in self.supported_conditions:
            raise RuntimeError(f"Condition '{condition_name}' not supported by this robot")
        return self.supported_conditions[condition_name]

    def require_nav(self) -> BaseNavController:
        """Get nav controller, raising if unavailable.

        Returns:
            BaseNavController instance

        Raises:
            RuntimeError: If nav controller not available
        """
        if not self.nav:
            raise RuntimeError("Navigation controller not available in context")
        return self.nav

    def require_arm(self) -> BaseArmController:
        """Get arm controller, raising if unavailable.

        Returns:
            BaseArmController instance

        Raises:
            RuntimeError: If arm controller not available
        """
        if not self.arm:
            raise RuntimeError("Arm controller not available in context")
        return self.arm

    def require_head(self) -> BaseHeadController:
        """Get head controller, raising if unavailable."""
        if not self.head:
            raise RuntimeError("Head controller not available in context")
        return self.head

    def get_planner(self, name: str | None = None) -> Any | None:
        """Return a configured manipulation planner by name.

        If *name* is omitted, the ``default`` planner is returned when present.
        """
        if not self.planners:
            return None
        if name:
            return self.planners.get(name)
        return self.planners.get("default")

    def require_planner(self, name: str | None = None) -> Any:
        """Return a configured planner, raising when unavailable."""
        planner = self.get_planner(name)
        if planner is None:
            label = name or "default"
            raise RuntimeError(f"Planner '{label}' not available in context")
        return planner

    def require_gripper(self) -> BaseGripperController:
        """Get gripper controller, raising if unavailable.

        Returns:
            BaseGripperController instance

        Raises:
            RuntimeError: If gripper controller not available
        """
        if not self.gripper:
            raise RuntimeError("Gripper controller not available in context")
        return self.gripper
