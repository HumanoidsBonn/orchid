"""Manifest listener utilities.

Two ways of obtaining :class:`~src.core.manifest.RobotManifest` instances:

1. **FileManifestListener** – reads manifests from JSON files on disk.
   Useful for test suites and local development.
2. **Ros2ManifestListener** – minimal ROS2 node exposing a ``get_manifest``
   service.  The implementation is guarded by ImportError handling so the module
   can be imported even when ROS2 (``rclpy``) is not installed.

Both implement the :class:`ManifestListener` interface with a ``load_manifests()``
method returning a list of :class:`src.core.manifest.RobotManifest` instances.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.manifest import RobotManifest, ControllerMap, ModuleEndpoint, PlannerMap

import src.core.logger as log


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ManifestListener(ABC):
    """Abstract base class for manifest loading strategies."""

    @abstractmethod
    def load_manifests(self) -> list[RobotManifest]:
        """Load and return a list of manifests.

        Returns:
            List of loaded RobotManifest instances.
        """
        pass


# ---------------------------------------------------------------------------
# JSON‑file implementation
# ---------------------------------------------------------------------------

class FileManifestListener(ManifestListener):
    """Load manifests from JSON files on disk."""

    def __init__(self, manifest_paths: list[str | Path] | None = None):
        """Initialize FileManifestListener.

        Args:
            manifest_paths: List of JSON file paths to load.
                           If None, no manifests are pre-loaded.
        """
        self.manifest_paths = [Path(p) for p in (manifest_paths or [])]

    def load_manifests(self) -> list[RobotManifest]:
        """Load and return manifests from all configured paths.

        Returns:
            List of loaded RobotManifest instances.

        Raises:
            FileNotFoundError: If any manifest file does not exist.
        """
        manifests: list[RobotManifest] = []
        for path in self.manifest_paths:
            manifest = load_manifest_from_file(path)
            manifests.append(manifest)
        return manifests

    def add_manifest_path(self, path: str | Path) -> None:
        """Add a manifest path for loading later."""
        self.manifest_paths.append(Path(path))


def load_manifest_from_file(file_path: str | Path) -> RobotManifest:
    """Load a :class:`RobotManifest` from a JSON file.

    The JSON schema mirrors the fields of :class:`RobotManifest` and its nested
    dataclasses.  Example file content is provided in
    ``src/orchestrator/example_manifest.json``.

    Args:
        file_path: Path to the JSON manifest file.

    Returns:
        An instantiated :class:`RobotManifest`.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with path.open() as f:
        data: dict[str, Any] = json.load(f)

    controller_map = ControllerMap(
        nav=data.get("controllers", {}).get("nav"),
        arm=data.get("controllers", {}).get("arm"),
        head=data.get("controllers", {}).get("head"),
        gripper=data.get("controllers", {}).get("gripper"),
    )
    planner_map = PlannerMap(
        manipulation=data.get("planners", {}).get("manipulation"),
        default=data.get("planners", {}).get("default"),
    )

    modules = [ModuleEndpoint(**m) for m in data.get("modules", [])]

    manifest = RobotManifest(
        robot_id=data["robot_id"],
        robot_type=data["robot_type"],
        capabilities=data.get("capabilities", []),
        conditions=data.get("conditions", []),
        controllers=controller_map,
        planners=planner_map,
        hardware=data.get("hardware", {}),
        modules=modules,
    )
    log.debug("Loaded manifest %s from %s", manifest.robot_id, path)
    return manifest


# ---------------------------------------------------------------------------
# ROS2 service implementation (optional)
# ---------------------------------------------------------------------------

class Ros2ManifestListener(ManifestListener):
    """ROS2 node that exposes a ``get_manifest`` service for loading manifests."""

    def __init__(self):
        """Initialize Ros2ManifestListener.

        Raises:
            RuntimeError: If ROS2 libraries are not available.
        """
        self._manifests: list[RobotManifest] = []

        rclpy, Node, Trigger = _try_import_rclpy()
        if rclpy is None:
            raise RuntimeError("ROS2 libraries not available in this environment")
        if not rclpy.ok():
            raise RuntimeError("ROS2 client library not initialized; call rclpy.init() before creating Ros2ManifestListener")
        self.rclpy = rclpy
        self.Node = Node
        self.Trigger = Trigger

        self.node = self.Node("manifest_listener")
        self.srv = self.node.create_service(
            self.Trigger, "get_manifest", self._handle_request
        )
        log.info("ROS2 Manifest listener service started on 'get_manifest'")

    def load_manifests(self) -> list[RobotManifest]:
        """Return the list of manifests received via the ROS2 service.

        Returns:
            List of manifests received via the service.
        """
        return self._manifests

    def _handle_request(self, request, response):
        """Handle service request where ``request.data`` holds a JSON string."""
        try:
            data = json.loads(request.data)  # type: ignore[attr-defined]
            manifest = self._parse_manifest_data(data)
            self._manifests.append(manifest)
            response.success = True  # type: ignore[attr-defined]
            response.message = f"Loaded {manifest.robot_id}"  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            response.success = False  # type: ignore[attr-defined]
            response.message = f"Failed to parse manifest: {e}"  # type: ignore[attr-defined]
        return response

    def _parse_manifest_data(self, data: dict) -> RobotManifest:
        """Convert a dict into a :class:`RobotManifest` instance."""
        controller_map = ControllerMap(
            nav=data.get("controllers", {}).get("nav"),
            arm=data.get("controllers", {}).get("arm"),
            head=data.get("controllers", {}).get("head"),
            gripper=data.get("controllers", {}).get("gripper"),
        )
        planner_map = PlannerMap(
            manipulation=data.get("planners", {}).get("manipulation"),
            default=data.get("planners", {}).get("default"),
        )
        modules = [ModuleEndpoint(**m) for m in data.get("modules", [])]
        return RobotManifest(
            robot_id=data["robot_id"],
            robot_type=data["robot_type"],
            capabilities=data.get("capabilities", []),
            conditions=data.get("conditions", []),
            controllers=controller_map,
            planners=planner_map,
            hardware=data.get("hardware", {}),
            modules=modules,
        )

    def shutdown(self):
        """Cleanly shut down the ROS2 node and context.

        Tests call this method to ensure resources are released.
        """
        try:
            if hasattr(self, "node") and self.node is not None:
                self.node.destroy_node()
            if hasattr(self, "rclpy") and self.rclpy is not None:
                self.rclpy.shutdown()
            log.debug("Ros2ManifestListener shutdown complete")
        except Exception:  # pragma: no cover – defensive
            log.exception("Error during Ros2ManifestListener.shutdown")


def _try_import_rclpy():
    try:
        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import Trigger
        return rclpy, Node, Trigger
    except Exception as exc:  # pragma: no cover
        log.info("ROS2 not available – ROS2 manifest listener disabled: %s", exc)
        return None, None, None
