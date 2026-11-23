"""ModuleProxy: routes calls to external service endpoints (HTTP, ROS2, etc.).

Provides:
- Unified interface for sync and async module calls
- Lazy module instantiation
- Endpoint registry and routing
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.manifest import ModuleEndpoint

import src.core.logger as log


class ModuleProxy:
    """Routes calls to external service modules.

    Supports both synchronous and asynchronous calls to:
    - HTTP endpoints (model APIs, etc.)
    - ROS2 services
    - ROS2 topics (one-shot subscribe)
    """

    def __init__(self, endpoints: list[ModuleEndpoint] | None = None):
        """Initialize ModuleProxy.

        Args:
            endpoints: List of ModuleEndpoint configurations
        """
        self.endpoints = {ep.name: ep for ep in (endpoints or [])}
        self._modules: dict[str, Any] = {}  # Cached module instances

    def call(
        self, module_name: str, payload: dict, timeout: float | None = None
    ) -> dict:
        """Synchronous call to a module.

        Args:
            module_name: Name of the module to call
            payload: Input payload (dict)
            timeout: Optional timeout in seconds

        Returns:
            Response dict from module

        Raises:
            ValueError: If module not configured
        """
        module = self._get_or_create_module(module_name)
        if not module:
            raise ValueError(f"Module '{module_name}' not configured")

        log.debug(f"ModuleProxy.call('{module_name}')")
        return module.call_sync(payload, timeout=timeout)

    async def call_async(
        self,
        module_name: str,
        payload: dict,
        files: dict | None = None,
        callback: Callable[[dict], Any] | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Asynchronous call to a module.

        Args:
            module_name: Name of the module to call
            payload: Input payload (dict)
            callback: Optional callback(result) when complete
            timeout: Optional timeout in seconds

        Returns:
            Response dict from module
        """
        module = self._get_or_create_module(module_name)
        if not module:
            raise ValueError(f"Module '{module_name}' not configured")

        log.debug(f"ModuleProxy.call_async('{module_name}')")
        return await module.call_async(payload, files=files, callback=callback, timeout=timeout)

    def has(self, module_name: str) -> bool:
        """Check if a module is available.

        Args:
            module_name: Name of module

        Returns:
            True if module is configured, False otherwise
        """
        return module_name in self.endpoints

    def close(self) -> None:
        """Release resources held by cached service modules."""
        for module in self._modules.values():
            close = getattr(module, "close", None)
            if close is not None:
                close()

    def _get_or_create_module(self, name: str) -> Any:
        """Lazily instantiate and cache a module.

        Args:
            name: Module name

        Returns:
            Module instance, or None if not found
        """
        if name in self._modules:
            return self._modules[name]

        ep = self.endpoints.get(name)
        if not ep:
            log.warning(f"Endpoint not found for module '{name}'")
            return None

        # Route to correct module implementation based on type

        if ep.type == "http":
            if ep.config.get("protocol") == "openai_chat":
                from src.core.modules.llm_chat_module import LLMChatModule
                module = LLMChatModule(ep.endpoint, ep.config)
            else:
                from src.core.modules.http_module import HTTPModule
                module = HTTPModule(ep.endpoint, ep.config)
        elif ep.type == "llm":
            from src.core.modules.llm_chat_module import LLMChatModule
            module = LLMChatModule(ep.endpoint, ep.config)
        elif ep.type == "ros2_service":
            from src.core.modules.ros2_service_module import ROS2ServiceModule
            module = ROS2ServiceModule(ep.endpoint, ep.config)
        elif ep.type == "ros2_subscriber":
            from src.core.modules.ros2_subscriber_module import ROS2SubscriberModule
            module = ROS2SubscriberModule(ep.endpoint)
        else:
            log.error(f"Unknown module type: {ep.type}")
            return None

        self._modules[name] = module
        log.debug(f"Created module: {name} (type={ep.type})")
        return module

    def clear(self) -> None:
        """Clear all cached module instances."""
        self.close()
        self._modules.clear()
