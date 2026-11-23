"""Dispatch server – manages robot manifests and sessions.

The server is deliberately lightweight: it does **not** start any network listeners
itself; it merely provides a programmatic API that other components (e.g. a ROS2
node or a test harness) can call.  It keeps two internal dictionaries:

* ``self.manifests`` – mapping ``robot_id`` → :class:`RobotManifest`
* ``self.sessions``  – mapping ``robot_id`` → :class:`RobotSession`

When a manifest is loaded – either via a JSON file or via the optional ROS2
service – the server stores the manifest and lazily creates a ``RobotSession``
iff one does not already exist for that robot.  Subsequent requests for the same
``robot_id`` reuse the existing session, avoiding duplicate initialisation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, TYPE_CHECKING
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Passing \(type, 1\) or '1type' as a synonym of type is deprecated.*",
    category=FutureWarning,
)

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64
    # np.int = int
    # np.bool = bool

if TYPE_CHECKING:
    from src.orchestrator.manifest_listener import ManifestListener

from src.core.manifest import RobotManifest
from src.orchestrator.session import RobotSession
from src.plugin.registry import PluginRegistry
import src.core.logger as log
from src.orchestrator.manifest_listener import FileManifestListener

DEFAULT_MANIFEST_PATH = Path(
    "/home/orchid/orchid_server/manifests/example_manifest_hsrb.json"
)

class DispatchServer:
    """Central registry for robot manifests and their sessions.

    Typical usage pattern::

        listener = FileManifestListener(["example_manifest.json"])
        server = DispatchServer(manifest_listener=listener)
        session = server.get_session("example_bot")
        # session.node is now ready to be added to a ROS2 executor
    """

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        manifest_listener: ManifestListener | None = None,
    ):
        """Initialize DispatchServer.

        Args:
            registry: Optional pre-populated PluginRegistry. If None, a fresh
                     registry is created and discovered.
            manifest_listener: Optional ManifestListener. If provided, manifests
                              are loaded immediately at init.
        """
        # Allow callers to provide a pre‑populated registry; otherwise create a fresh one.
        self.registry: PluginRegistry = registry or PluginRegistry()
        # Populate the registry (no entry‑points in the test environment).
        self.registry.discover()

        # from tests.fixtures.mock_actions import SuccessAction, FailingAction, AsyncSuccessAction, ContextCheckAction
        # self.registry.register_action("navigate", AsyncSuccessAction)
        # self.registry.register_action("pick_object", AsyncSuccessAction)
        # self.registry.register_action("place_object", FailingAction)
        # self.registry.register_action("grasp_object", AsyncSuccessAction)
        # self.registry.register_action("release_object", FailingAction)
        # self.registry.register_action("look_at_object", ContextCheckAction)
        # self.registry.register_action("look_at_location", ContextCheckAction)
        # self.registry.register_action("look_at_person", ContextCheckAction)

        self.manifests: Dict[str, RobotManifest] = {}
        self.sessions: Dict[str, RobotSession] = {}
        self.manifest_listener = manifest_listener

        log.debug("DispatchServer initialised with empty manifest/session caches")

        # Load manifests if a listener was provided
        if self.manifest_listener:
            self._load_manifests_from_listener()

    # -----------------------------------------------------------------------
    # Manifest loading
    # -----------------------------------------------------------------------

    def _load_manifests_from_listener(self) -> None:
        """Load manifests using the configured listener."""
        if not self.manifest_listener:
            log.debug("No manifest listener configured; skipping auto-load")
            return

        try:
            loaded_manifests = self.manifest_listener.load_manifests()
            for manifest in loaded_manifests:
                self.manifests[manifest.robot_id] = manifest
                log.info("Loaded manifest for robot %s via listener", manifest.robot_id)
        except Exception as e:  # noqa: BLE001
            log.error("Failed to load manifests from listener: %s", e)

    # -----------------------------------------------------------------------
    # Session handling
    # -----------------------------------------------------------------------
    def get_session(self, robot_id: str) -> RobotSession:
        """Return a :class:`RobotSession` for *robot_id*.

        The session is constructed on first request and then cached for future
        calls.  The underlying ``RobotSession`` holds its own ``Context`` that is
        built from the manifest and the shared plugin registry.
        """
        if robot_id not in self.manifests:
            raise KeyError(f"Manifest for robot_id '{robot_id}' not loaded yet")

        if robot_id not in self.sessions:
            manifest = self.manifests[robot_id]
            self.sessions[robot_id] = RobotSession(manifest, self.registry)
            log.info("Created new RobotSession for robot %s", robot_id)
        else:
            log.debug("Reusing existing RobotSession for robot %s", robot_id)
        return self.sessions[robot_id]

    def list_robots(self) -> list[str]:
        """Return a list of robot IDs known to the server."""
        return list(self.manifests.keys())


# ============================================================================
# Main entry point for ROS2-based dispatch
# ============================================================================

def _split_manifest_paths(value: str | None) -> list[str]:
    """Split comma-separated manifest paths, ignoring empty entries."""
    if not value:
        return []
    return [path.strip() for path in value.split(",") if path.strip()]


def _parse_dispatch_args(args: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse dispatch-specific args while preserving unknown ROS args."""
    parser = argparse.ArgumentParser(description="Launch the Orchid dispatch server")
    parser.add_argument('--speech-mode', choices=('current', 'off', 'tmc'),
                        default=os.environ.get('ORCHID_SPEECH_MODE', 'current'))
    parser.add_argument(
        "--manifest-type",
        choices=("file", "ros2"),
        default=None,
        help="Manifest source type. Defaults to CONFIG_DISPATCH_MANIFEST_TYPE or file.",
    )
    parser.add_argument(
        "--manifest-file",
        action="append",
        default=[],
        help="Manifest JSON path. Can be repeated.",
    )
    parser.add_argument(
        "--manifest-files",
        default=None,
        help="Comma-separated manifest JSON paths.",
    )
    return parser.parse_known_args(args)


def _resolve_manifest_config(
    parsed_args: argparse.Namespace,
    environ: dict[str, str] | None = None,
) -> tuple[str, list[Path]]:
    """Resolve manifest type and paths from CLI args, env vars, and defaults."""
    env = environ if environ is not None else os.environ
    manifest_type = (
        parsed_args.manifest_type
        or env.get("CONFIG_DISPATCH_MANIFEST_TYPE")
        or "file"
    ).lower()

    cli_paths = list(parsed_args.manifest_file or [])
    cli_paths.extend(_split_manifest_paths(parsed_args.manifest_files))
    env_paths = _split_manifest_paths(env.get("CONFIG_DISPATCH_MANIFEST_FILES"))

    if cli_paths:
        paths = cli_paths
    elif env_paths:
        paths = env_paths
    else:
        paths = [str(DEFAULT_MANIFEST_PATH)]

    return manifest_type, [Path(path).expanduser() for path in paths]


def main(args=None):
    """Main entry point for the dispatch server.

    This starts a ROS2 MultiThreadedExecutor that:
    1. Loads manifests via the configured listener
    2. Creates RobotSession ROS2 nodes for each robot
    3. Adds them to the executor
    4. Spins the executor to handle incoming tasks

    Uses environment variable CONFIG_DISPATCH_MANIFEST_TYPE to determine
    manifest listener type: "file" or "ros2" (default: "file").
    """
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy._rclpy_pybind11 import InvalidHandle
    except ImportError:
        raise RuntimeError("ROS2 is required to run the dispatch server")

    log.info("Starting dispatch server with ROS2 MultiThreadedExecutor")

    parsed_args, ros_args = _parse_dispatch_args(sys.argv[1:] if args is None else args)
    log.set_speech_mode(parsed_args.speech_mode)
    log.info('Speech output mode=%s', parsed_args.speech_mode)
    manifest_type, manifest_paths = _resolve_manifest_config(parsed_args)

    if manifest_type == "file":
        listener = FileManifestListener(manifest_paths)
        log.info(
            "Using FileManifestListener with paths: %s",
            [str(path) for path in manifest_paths],
        )
    elif manifest_type == "ros2":
        from src.orchestrator.manifest_listener import Ros2ManifestListener
        listener = Ros2ManifestListener()
        manifest_node = listener.node  # Keep a reference to the ROS2 node for later use
        log.info("Using Ros2ManifestListener")
    else:
        raise ValueError(f"Unknown manifest listener type: {manifest_type}")

    # Initialize ROS2
    rclpy.init(args=ros_args if ros_args else None)

    # Create server with listener
    server = DispatchServer(manifest_listener=listener)

    # Add RobotSession nodes to executor
    executor = MultiThreadedExecutor(num_threads=4)
    nodes_added = []  # Keep track of added nodes to destroy them on shutdown
    if manifest_type == "ros2":
        executor.add_node(manifest_node)  # Add the manifest listener node if using ROS2
        nodes_added.append(manifest_node)
    for robot_id in server.list_robots():
        session = server.get_session(robot_id)
        session.start()
        executor.add_node(session.ros2_wrapper.node)
        nodes_added.append(session.ros2_wrapper.node)
        log.info("Added RobotSession node %s for robot %s to executor", session.ros2_wrapper.node.get_name(), robot_id)

    # log.speak("Dispatch server running with robots: %s", server.list_robots())
    log.speak("Dispatch server running!")

    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     log.info("Dispatch server interrupted")
    # finally:
    #     executor.shutdown()
    #     for node in nodes_added:
    #         node.destroy_node()
    #     if rclpy.ok():
    #         rclpy.shutdown()
    #     log.info("Dispatch server shut down")

    
    try:
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except InvalidHandle:
                log.warn("ROS executor skipped an entity whose destruction was requested")
    except KeyboardInterrupt:
        log.info("Dispatch server interrupted")
    finally:
        executor.shutdown()
        for node in nodes_added:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        log.info("Dispatch server shut down")

if __name__ == "__main__":
    main()
