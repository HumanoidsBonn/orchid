"""ROS2ServiceModule: call ROS2 service endpoints."""

from __future__ import annotations

import asyncio
import importlib
import time
import threading
from typing import Any

import src.core.logger as log



class ROS2ServiceModule:
    """ROS2 service client for calling remote services."""

    def __init__(self, service_name: str, config: dict[str, Any] | None = None):
        self.service_name = service_name
        self.config = dict(config or {})
        self._service_type = self.config.get("service_type")
        self._request_type = self.config.get("request_type")
        self._timeout = float(self.config.get("timeout", 5.0))
        self._node = None
        self._client = None
        self._executor = None
        self._call_lock = threading.RLock()

    def call_sync(self, payload: dict, timeout: float | None = None) -> dict:
        # One request owns this module's executor until its response is consumed.
        with self._call_lock:
            return self._call_sync_serial(payload, timeout)

    def _call_sync_serial(self, payload: dict, timeout: float | None = None) -> dict:
        try:
            import rclpy

            if not rclpy.ok():
                log.warning("ROS2 not initialized; returning mock response")
                return {"status": "mock", "service": self.service_name, "request": payload}

            log.debug("ROS2ServiceModule.call_sync(%s)", self.service_name)
            request = self._build_request(payload)
            client = self._get_client()
            if client is None:
                return {"error": "ROS2 service client unavailable"}

            if not client.wait_for_service(timeout_sec=timeout or self._timeout):
                return {"error": f"ROS2 service unavailable: {self.service_name}"}

            future = client.call_async(request)
            deadline = time.monotonic() + (timeout or self._timeout)
            while not future.done():
                self._executor.spin_once(timeout_sec=0.01)
                if time.monotonic() > deadline:
                    break
            if future.done():
                response = future.result()
                if isinstance(payload, dict) and payload.get("raw_response"):
                    return response
                return self._serialize_response(response)
            future.cancel()
            return {"error": f"ROS2 service call timed out: {self.service_name}"}

        except ImportError:
            log.warning("rclpy not installed; returning mock response")
            return {"status": "mock", "request": payload}
        except Exception as exc:  # noqa: BLE001
            log.error("ROS2 service call failed: %s", exc)
            return {"error": str(exc)}

    async def call_async(
        self,
        payload: dict,
        files: dict | None = None,
        callback=None,
        timeout: float | None = None,
    ) -> dict:
        result = await asyncio.to_thread(self.call_sync, payload, timeout)
        if callback is not None:
            callback(result)
        return result

    def _build_request(self, payload: dict) -> Any:
        service_type = self._resolve_service_type()
        request_type = self._resolve_request_type(service_type)
        if request_type is None:
            raise ValueError("ROS2 service type not configured")

        if isinstance(payload, dict) and "request" in payload and payload.get("request") is not None:
            return payload["request"]

        request = request_type()
        if isinstance(payload, dict):
            for key, value in payload.items():
                if hasattr(request, key):
                    setattr(request, key, value)
        return request

    def _resolve_service_type(self) -> Any:
        if self._service_type is not None:
            if isinstance(self._service_type, str):
                self._service_type = self._import_type(self._service_type)
            return self._service_type
        raise ValueError("ROS2 service type not configured")

    def _resolve_request_type(self, service_type: Any) -> Any:
        if self._request_type is not None:
            if isinstance(self._request_type, str):
                self._request_type = self._import_type(self._request_type)
            return self._request_type
        return getattr(service_type, "Request", None)

    @staticmethod
    def _import_type(type_ref: str) -> Any:
        module_name, sep, attr_name = type_ref.partition(":")
        if not sep:
            module_name, sep, attr_name = type_ref.rpartition(".")
        if not module_name or not attr_name:
            raise ValueError(f"Invalid ROS2 type reference '{type_ref}'")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def _get_node(self) -> Any:
        if self._node is None:
            import rclpy

            from rclpy.executors import SingleThreadedExecutor
            self._node = rclpy.create_node(f"ros2_service_client_{self.service_name.replace('/', '_')}")
            self._executor = SingleThreadedExecutor(context=self._node.context)
            self._executor.add_node(self._node)
        return self._node

    def _get_client(self) -> Any:
        with self._call_lock:
            if self._client is None:
                service_type = self._resolve_service_type()
                self._client = self._get_node().create_client(service_type, self.service_name)
            return self._client

    def close(self) -> None:
        """Release the cached client and module-owned node/executor."""
        with self._call_lock:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
            self._client = self._executor = self._node = None

    @staticmethod
    def _serialize_response(response: Any) -> dict[str, Any]:
        if response is None:
            return {"status": "ok"}
        if hasattr(response, "get_fields_and_field_types"):
            return {
                field: ROS2ServiceModule._serialize_value(getattr(response, field))
                for field in response.get_fields_and_field_types().keys()
            }
        if hasattr(response, "__dict__"):
            return {
                k: ROS2ServiceModule._serialize_value(v)
                for k, v in vars(response).items()
                if not k.startswith("_")
            }
        return {"response": response}

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if hasattr(value, "get_fields_and_field_types"):
            return {
                field: ROS2ServiceModule._serialize_value(getattr(value, field))
                for field in value.get_fields_and_field_types().keys()
            }
        if isinstance(value, (list, tuple)):
            return [ROS2ServiceModule._serialize_value(item) for item in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
