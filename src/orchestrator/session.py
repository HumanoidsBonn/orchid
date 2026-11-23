"""RobotSession: per-robot execution context as a ROS2 Node.

RobotSession manages:
- Manifest-to-context initialization
- Execution engine setup
- ROS2 node lifecycle (inherited from rclpy.node.Node)
- Task action server for receiving and executing tasks
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Callable, Optional
from src.workflows.catalog import WorkflowCatalog, WorkflowRequest

if TYPE_CHECKING:
    from src.core.manifest import RobotManifest
    from src.plugin.registry import PluginRegistry

from src.core.context import Context
from src.core.llm_planner import LLMPlanner, PlanGenerationError, UserInputRequired
from src.core.world_state import WorldState
from src.tree.execution import ExecutionEngine, ExecutionStatus
from src.tree.plan import Plan

import src.core.logger as log

try:
    import rclpy
    from rclpy.action import CancelResponse, GoalResponse
except ImportError:
    log.warn("ROS2 client library not available; RobotSession will run without ROS2 support")


def flush_input_buffer():
    try:
        # For Linux/macOS
        import termios
        # import tcflow
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (ImportError, OSError, ValueError, AttributeError):
        log.warn("terminal input buffer cannot be flushed")

def _try_import_rclpy():
    """Attempt to import rclpy components for ROS2 integration."""
    try:
        from rclpy.node import Node
        return Node
    except ImportError:
        return None


def _latest_reliable_qos(depth: int = 1):
    """Return a reliable keep-last QoS profile for latest-only UI state."""
    try:
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )
    except Exception:  # noqa: BLE001
        return depth


class RobotSession:
    """Session state for a connected robot.

    Owns the execution context, LLM planner, and execution engine. When ROS2 is
    available a ``Ros2RobotSessionNode`` wrapper receives tasks and forwards them
    into the session run loop.
    """

    def __init__(self, manifest: RobotManifest, registry: PluginRegistry):
        """Initialize session from manifest.

        Args:
            manifest: RobotManifest declaring capabilities
            registry: PluginRegistry with available actions/controllers
        """
        self.robot_id = manifest.robot_id
        self.manifest = manifest
        self.supported_actions: dict[str, type] = {}
        self.supported_controllers: dict[str, str] = {}
        self.supported_planners: dict[str, str] = {}
        self.unsupported_actions: list[str] = []
        self.unsupported_controllers: list[str] = []
        self.unsupported_planners: list[str] = []
        self.action_history_length: int = int(os.getenv("ORCHID_ACTION_HISTORY_LENGTH", "50"))

        self._task_lock = threading.Lock()
        self._task_event = threading.Event()
        self._intake_reserved = False
        self._active_request = None
        self._pending_task: str | None = None
        self._active_task: str | None = None
        self._run_thread: threading.Thread | None = None#
        self.planning = False
        self.stop_planning = False
        self._operation_started_at: float | None = None

        self.session_running = False
        self.ros2_wrapper: Ros2RobotSessionNode | None = None

        Node = _try_import_rclpy()
        if Node is not None:
            import rclpy

            if not rclpy.ok():
                log.error("ROS2 client library not initialized; cannot create node")
                raise RuntimeError(
                    "ROS2 client library not initialized; call rclpy.init() before creating RobotSession"
                )
            self._init_ros2_components(Node)
            log.debug("RobotSession %s initialized with ROS2 support", self.robot_id)
        else:
            log.debug("RobotSession %s initialized without ROS2", self.robot_id)

        self.context = self._build_context(manifest, registry)
        self.context.request_user_input = self.request_user_input
        self.executor = ExecutionEngine(self.context)
        ros2_node = getattr(self.ros2_wrapper, "node", None) if self.ros2_wrapper else None
        if ros2_node is not None:
            self.executor.set_ros2_node(ros2_node)
        self.planner = LLMPlanner(self.context)
        self.workflow_catalog = WorkflowCatalog(self.context, self.robot_id)
        if self.ros2_wrapper is not None:
            self.ros2_wrapper.setup_workflows()

    def _init_ros2_components(self, Node):
        """Initialize ROS2 node for task intake and status publishing."""
        self.ros2_wrapper = Ros2RobotSessionNode(self.robot_id, self)
        log.info("ROS2 components initialized for robot %s", self.robot_id)

    def _build_context(self, manifest: RobotManifest, registry: PluginRegistry) -> Context:
        """Build execution Context for this robot."""
        supported_conditions = {}
        for name in manifest.conditions:
            condition_cls = registry.get_condition(name)
            if condition_cls is None:
                raise ValueError(f"Unknown manifest condition: {name}")
            supported_conditions[name] = condition_cls
        for cap in manifest.capabilities:
            action_cls = registry.get_action(cap)
            if action_cls:
                self.supported_actions[cap] = action_cls
            else:
                self.unsupported_actions.append(cap)

        if self.unsupported_actions:
            log.warn(
                "Robot %s has unsupported capabilities: %s",
                manifest.robot_id,
                self.unsupported_actions,
            )

        controllers = {}
        planners = {}

        from src.core.module_proxy import ModuleProxy
        from src.core.sensor_data_center import SensorDataCenter
        from src.core.scene_graph import SceneGraph

        module_proxy = ModuleProxy(manifest.modules)

        # Create or get singleton SensorDataCenter for this robot before
        # controllers so MoveIt controllers can seed requests from joint state.
        sensor_data_center = SensorDataCenter.get_or_create(
            manifest.robot_id, manifest
        )
        if sensor_data_center is not None:
            ros2_node = getattr(self.ros2_wrapper, "node", None) if self.ros2_wrapper else None
            if ros2_node is not None:
                sensor_data_center.set_node(ros2_node)
                log.debug(
                    "ROS2 node linked to SensorDataCenter for robot %s",
                    self.robot_id,
                )

        if manifest.controllers.nav:
            ctrl_cls = registry.get_controller(manifest.controllers.nav)
            if ctrl_cls:
                controllers["nav"] = ctrl_cls(self._controller_hardware(manifest, "nav", module_proxy, sensor_data_center))
                self.supported_controllers["nav"] = manifest.controllers.nav
            else:
                self.unsupported_controllers.append(manifest.controllers.nav)

        if manifest.controllers.arm:
            ctrl_cls = registry.get_controller(manifest.controllers.arm)
            if ctrl_cls:
                controllers["arm"] = ctrl_cls(self._controller_hardware(manifest, "arm", module_proxy, sensor_data_center))
                self.supported_controllers["arm"] = manifest.controllers.arm
            else:
                self.unsupported_controllers.append(manifest.controllers.arm)

        if manifest.controllers.head:
            ctrl_cls = registry.get_controller(manifest.controllers.head)
            if ctrl_cls:
                controllers["head"] = ctrl_cls(self._controller_hardware(manifest, "head", module_proxy, sensor_data_center))
                self.supported_controllers["head"] = manifest.controllers.head
            else:
                self.unsupported_controllers.append(manifest.controllers.head)

        if manifest.controllers.gripper:
            ctrl_cls = registry.get_controller(manifest.controllers.gripper)
            if ctrl_cls:
                controllers["gripper"] = ctrl_cls(
                    self._controller_hardware(manifest, "gripper", module_proxy, sensor_data_center)
                )
                self.supported_controllers["gripper"] = manifest.controllers.gripper
            else:
                self.unsupported_controllers.append(manifest.controllers.gripper)

        planner_names = {
            "manipulation": getattr(manifest.planners, "manipulation", None),
            "default": getattr(manifest.planners, "default", None),
        }
        if planner_names["default"] is None and planner_names["manipulation"] is not None:
            planner_names["default"] = planner_names["manipulation"]
        for role, planner_name in planner_names.items():
            if not planner_name:
                continue
            planner_cls = registry.get_planner(planner_name)
            if planner_cls:
                planner = planner_cls(self._planner_config(manifest, role, planner_name))
                planners[role] = planner
                self.supported_planners[role] = planner_name
            else:
                self.unsupported_planners.append(planner_name)

        # Load the Boxer scene graph if available
        scene_graph = self._load_scene_graph()

        return Context(
            robot_id=manifest.robot_id,
            hardware=manifest.hardware,
            modules=module_proxy,
            sensor_data_center=sensor_data_center,
            scene_graph=scene_graph,
            planners=planners,
            **controllers,
            supported_actions=self.supported_actions,
            supported_conditions=supported_conditions,
            world_state=WorldState(max_history=self.action_history_length),
            ros2_node=getattr(self.ros2_wrapper, "node", None) if self.ros2_wrapper else None,
        )

    # @staticmethod
    def _controller_hardware(self, manifest: RobotManifest, role: str, module_proxy=None, sensor_data_center=None) -> dict:
        """Merge top-level hardware config with role-specific overrides."""
        hardware = dict(manifest.hardware)
        role_cfg = manifest.hardware.get(role)
        if isinstance(role_cfg, dict):
            hardware.update(role_cfg)
        if self.ros2_wrapper is not None:
            hardware["ros2_node"] = self.ros2_wrapper.node
        if module_proxy is not None:
            hardware["module_proxy"] = module_proxy
        if sensor_data_center is not None:
            hardware["sensor_data_center"] = sensor_data_center
            hardware["joint_state_provider"] = self._joint_state_provider(sensor_data_center)
        return hardware

    @staticmethod
    def _joint_state_provider(sensor_data_center):
        def provider():
            if hasattr(sensor_data_center, "peek_latest"):
                try:
                    msg = sensor_data_center.peek_latest("joint_states", timeout_s=0.2)
                except TypeError:
                    msg = sensor_data_center.peek_latest("joint_states")
            else:
                try:
                    msg = sensor_data_center.get_latest("joint_states", timeout_s=0.2)
                except TypeError:
                    msg = sensor_data_center.get_latest("joint_states")
            return getattr(msg, "data", msg)

        return provider

    @staticmethod
    def _planner_config(manifest: RobotManifest, role: str, planner_name: str) -> dict:
        """Return planner config from hardware.planners."""
        planners_cfg = manifest.hardware.get("planners", {})
        config: dict[str, Any] = {}
        if isinstance(planners_cfg, dict):
            by_role = planners_cfg.get(role)
            if isinstance(by_role, dict):
                config.update(by_role)
            by_name = planners_cfg.get(planner_name)
            if isinstance(by_name, dict):
                config.update(by_name)
        config.setdefault("name", planner_name)
        config.setdefault("role", role)
        return config

    @staticmethod
    def _load_scene_graph() -> SceneGraph | None:
        """Load the configured scene dataset independently of the UI process.

        Returns:
            Loaded SceneGraph, or None if not found or on error
        """
        import os
        from pathlib import Path
        from src.core.scene_graph import SceneGraph

        default_path = Path(__file__).resolve().parents[2] / 'config' / 'scene' / 'scene_graph.json'
        scene_graph_path = Path(os.getenv('ORCHID_SCENE_GRAPH_PATH', str(default_path))).expanduser().resolve()
        try:
            scene_graph = SceneGraph.from_json(str(scene_graph_path))
            furniture = scene_graph.get_nodes('furniture')
            visitable = sum(scene_graph.get_best_access_pose(node.id) is not None for node in furniture)
            log.info('Loaded scene graph from %s: objects=%d furniture=%d visitable_furniture=%d',
                     scene_graph_path, len(scene_graph.get_nodes('object')), len(furniture), visitable)
            return scene_graph
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to load scene graph from %s: %s", scene_graph_path, exc)

        return None

    def reserve_task(self) -> bool:
        """Reserve the single task intake slot before accepting a ROS action goal."""
        with self._task_lock:
            if self._intake_reserved or self._pending_task is not None or self._active_task is not None or self.planning or self.executor.is_running:
                return False
            self._intake_reserved = True
            return True

    def _submit_request(self, request: str | WorkflowRequest, *, reserved: bool = False) -> bool:
        with self._task_lock:
            if reserved:
                self._intake_reserved = False
            elif self._intake_reserved:
                return False
            if self._pending_task is not None or self._active_task is not None or self.planning or self.executor.is_running:
                return False
            self._pending_task = request
        self._task_event.set()
        log.info("Queued task for %s: %s", self.robot_id, request)
        return True

    def submit_task(self, command: str, *, reserved: bool = False) -> bool:
        command = command.strip()
        return bool(command) and self._submit_request(command, reserved=reserved)

    def submit_workflow(self, workflow_id: str, inputs: dict, *, reserved: bool = False) -> bool:
        try:
            request = self.workflow_catalog.validate(workflow_id, inputs)
        except Exception:
            if reserved:
                with self._task_lock:
                    self._intake_reserved = False
            raise
        return self._submit_request(request, reserved=reserved)

    def _consume_pending_task(self) -> str | WorkflowRequest | None:
        """Take and clear the next pending task, if any."""
        with self._task_lock:
            task = self._pending_task
            self._pending_task = None
            if task is not None:
                self._active_request = task
                self._active_task = str(task)
            return task

    def start(self) -> None:
        """Start the session run loop on a background thread."""
        if self._run_thread and self._run_thread.is_alive():
            return

        self.session_running = True
        self._run_thread = threading.Thread(
            target=self.run,
            name=f"session-{self.robot_id}",
            daemon=True,
        )
        self._run_thread.start()

    def start_session(self) -> None:
        """Start the session run loop on the calling thread."""
        self.session_running = True
        self.run()

    def run(self) -> None:
        """Main loop: consume queued tasks, plan, and execute."""
        log.info("Starting session loop for robot %s", self.robot_id)
        while self.session_running:
            self._task_event.wait(timeout=0.2)
            self._task_event.clear()

            command = self._consume_pending_task()
            if not command:
                continue

            self._handle_new_task(command)

        if self.executor.is_running:
            log.error('Executor stop function called by run!')
            self.executor.stop(wait=True)
        log.info("Session loop exited for robot %s", self.robot_id)

    def _handle_new_task(self, command: str | WorkflowRequest) -> None:
        """Cancel any active plan, generate a new one, and execute while handling replans."""
        self._active_request = command
        workflow_request = command if isinstance(command, WorkflowRequest) else None
        command = str(command)
        if self.executor.is_running:
            log.info(
                "Cancelling active plan on robot %s for new task: %s",
                self.robot_id,
                command,
            )
            self._set_current_operation(
                "Cancellation",
                "Cancelling previous task",
                status="running",
                percent_completed=0.0,
            )
            log.error('Executor stop function called by handle_new_task top!')
            self.executor.stop(wait=True)
            self._publish_task_status(
                f"cancelled previous task; status={self.executor.status.value}"
            )

        self._active_task = command
        self._set_current_operation(
            "Task",
            f"Accepted task: {command}",
            status="queued",
            percent_completed=0.0,
        )
        original_command = command
        user_feedback: str | None = None
        last_action_result: dict[str, Any] | None = None
        build_error: str | None = None
        attempt = 0
        max_attempts = 1 if workflow_request else 100
        task_started_at = time.time()

        while attempt < max_attempts and not self.stop_planning:
            attempt += 1
            planning_message = "Preparing workflow" if workflow_request else "Planning with LLM"
            if user_feedback:
                planning_message = "Planning with user feedback"
            elif last_action_result:
                planning_message = "Replanning from action result"
            elif build_error:
                planning_message = "Replanning after tree build failed"
            self._set_current_operation(
                "Planning",
                planning_message,
                status="running",
                percent_completed=5.0,
            )
            self._publish_task_status("planning")
            action_history = (
                self._current_task_action_history(task_started_at)
                if user_feedback or last_action_result or build_error
                else None
            )

            self.planning = True
            try:
                if workflow_request:
                    plan = self.workflow_catalog.build_plan(workflow_request)
                else:
                    log.speak("Planning")
                    plan = self.planner.generate_plan(
                        original_command,
                        user_feedback=user_feedback,
                        last_action_result=last_action_result,
                        build_error=build_error,
                        action_history=action_history,
                    )
            except UserInputRequired as exc:
                self._set_current_operation(
                    "Waiting For Input",
                    str(exc),
                    status="needs_user_input",
                    percent_completed=5.0,
                )
                clarification = self.request_user_input(str(exc))
                if not clarification or not clarification.strip():
                    self._set_current_operation(
                        "Task",
                        "Cancelled: no clarification provided by user",
                        status="cancelled",
                        percent_completed=100.0,
                    )
                    self._publish_task_status(
                        "cancelled: no clarification provided by user"
                    )
                    self._active_task = None
                    self.planning = False
                    return
                user_feedback = clarification
                continue
            except (PlanGenerationError, ValueError) as exc:
                log.error(
                    "Plan generation failed for robot %s task %r: %s",
                    self.robot_id,
                    original_command,
                    exc,
                )
                self._active_task = None
                self._set_current_operation(
                    "Planning",
                    f"Planning failed: {exc}",
                    status="failure",
                    percent_completed=100.0,
                )
                self._publish_task_status(f"planning_failed: {exc}")
                self.planning = False
                return

            if self._plan_contains_error_action(plan):
                prompt = self._plan_error_prompt(plan)
                self._set_current_operation(
                    "Waiting For Input",
                    prompt,
                    status="needs_user_input",
                    percent_completed=5.0,
                )
                clarification = self.request_user_input(prompt)
                if not clarification or not clarification.strip():
                    self._set_current_operation(
                        "Task",
                        "Cancelled: planner reported task impossible",
                        status="cancelled",
                        percent_completed=100.0,
                    )
                    self._publish_task_status(
                        "cancelled: planner reported task impossible"
                    )
                    self._active_task = None
                    self.planning = False
                    return
                user_feedback = clarification
                last_action_result = None
                build_error = None
                continue
            
            if self.stop_planning: 
                self.planning = False
                self.stop_planning = False
                self._set_current_operation(
                    "Planning",
                    "Cancelled",
                    status="cancelled",
                    percent_completed=100.0,
                )
                return
            
            try:
                self._set_current_operation(
                    "Build Tree",
                    "Building behavior tree",
                    status="running",
                    percent_completed=10.0,
                )
                self._publish_task_status("building_tree")
                tree = self.executor.build_tree(plan)
            except Exception as exc:  # noqa: BLE001
                log.warn(
                    "Tree build failed for robot %s: %s",
                    self.robot_id,
                    exc,
                )
                self._set_current_operation(
                    "Build Tree",
                    f"Tree build failed: {exc}",
                    status="failure",
                    percent_completed=10.0,
                )
                build_error = str(exc)
                continue

            log.info(
                "Starting execution for robot %s goal=%r",
                self.robot_id,
                plan.goal,
            )
            if self.stop_planning: 
                self.planning = False
                self.stop_planning = False
                self._set_current_operation(
                    "Execution",
                    "Cancelled before execution",
                    status="cancelled",
                    percent_completed=100.0,
                )
                return
            self._set_current_operation(
                "Execution",
                f"Executing: {plan.goal}",
                status="running",
                percent_completed=15.0,
            )
            self._publish_task_status(f"executing: {plan.goal}")
            self.planning = False
            self.executor.start_plan(plan, tree=tree)
            # self.executor.join(timeout=60.0)
            self.executor.join()
            if getattr(self.executor, "_thread", None) is not None and self.executor._thread.is_alive():
                log.error('Executor stop function called by handle_new_task 2nd!')
                self.executor.stop(wait=True)

            snapshot = self.executor.get_snapshot()
            if snapshot.status == ExecutionStatus.SUCCEEDED:
                self._set_current_operation(
                    "Task",
                    f"Succeeded: {plan.goal}",
                    status="success",
                    percent_completed=100.0,
                )
                self._publish_task_status(f"succeeded: {plan.goal}")
                self._active_task = None
                return

            if snapshot.status == ExecutionStatus.CANCELLED:
                self._set_current_operation(
                    "Task",
                    f"Cancelled: {plan.goal}",
                    status="cancelled",
                    percent_completed=100.0,
                )
                self._publish_task_status(f"cancelled: {plan.goal}")
                self._active_task = None
                return

            if workflow_request:
                last = self.context.world_state.get("last_action_result") or {}
                reason = last.get("reason") or "Workflow execution failed"
                log.warn("Workflow %s failed: %s", workflow_request.workflow_id, reason)
                self._set_current_operation("Task", str(reason), status="failure", percent_completed=100.0)
                self._publish_task_status(f"failed: {reason}")
                self._active_task = None
                return

            last_action_result = self.context.world_state.get("last_action_result")
            if last_action_result and last_action_result.get("status") == "needs_replan":
                payload = last_action_result.get("payload") or {}
                suggested = last_action_result.get("suggested_replan_context") or {}
                clarification = payload.get("user_input") or suggested.get("user_input")
                if clarification:
                    user_feedback = str(clarification)
                self._set_current_operation(
                    "Replanning",
                    "Replanning after action requested it",
                    status="running",
                    percent_completed=5.0,
                )
                self._publish_task_status("replanning")
                build_error = None
                continue

            if last_action_result and last_action_result.get("status") == "needs_user_input":
                prompt = last_action_result.get(
                    "reason",
                    "The task needs more information. Please clarify the next step.",
                )
                self._set_current_operation(
                    "Waiting For Input",
                    str(prompt),
                    status="needs_user_input",
                    percent_completed=5.0,
                )
                clarification = self.request_user_input(prompt)
                if not clarification or not clarification.strip():
                    self._set_current_operation(
                        "Task",
                        "Cancelled: no clarification provided by user",
                        status="cancelled",
                        percent_completed=100.0,
                    )
                    self._publish_task_status(
                        "cancelled: no clarification provided by user"
                    )
                    self._active_task = None
                    return
                user_feedback = clarification
                last_action_result = None
                build_error = None
                continue

            log.error(
                "Execution failed for robot %s goal=%r after attempt %s",
                self.robot_id,
                plan.goal,
                attempt,
            )
            self._set_current_operation(
                "Execution",
                f"Execution failed: {plan.goal}",
                status="failure",
                percent_completed=100.0,
            )
            self._publish_task_status(f"execution_failed: {plan.goal}")
            self._active_task = None
            return

        if self.stop_planning:
            log.error(f"Planning Stopped externally for robot {self.robot_id} after {attempt} attempt(s)")
            self._set_current_operation(
                "Planning",
                "Cancelled",
                status="cancelled",
                percent_completed=100.0,
            )
            self._publish_task_status("Cancelled!")
            self.stop_planning = False
        else: 
            log.error(f"Aborted replanning loop for robot {self.robot_id} after {max_attempts} attempts")
            self._set_current_operation(
                "Replanning",
                "Failed: too many replanning attempts",
                status="failure",
                percent_completed=100.0,
            )
            self._publish_task_status("failed: too many replanning attempts")
        self.planning = False
        self._active_task = None

    def _current_task_action_history(self, task_started_at: float) -> list[dict[str, Any]]:
        """Return compact action history for the active task only."""
        world_state = self.context.world_state
        if world_state is None:
            return []
        return [
            record.to_dict()
            for record in world_state.get_action_history()
            if record.timestamp >= task_started_at
        ]

    def get_execution_status(self) -> ExecutionStatus:
        """Return the current execution engine status."""
        return self.executor.status

    @staticmethod
    def _plan_contains_error_action(plan: Plan) -> bool:
        """Return True if the plan contains a terminal error request action."""
        from src.tree.plan import ActionNode, CompositeNode, DecoratorNode

        def traverse(node):
            if isinstance(node, ActionNode):
                return node.action_type == "error"
            if isinstance(node, CompositeNode):
                return any(traverse(child) for child in node.children)
            if isinstance(node, DecoratorNode):
                return traverse(node.child)
            return False

        return traverse(plan.root)

    @staticmethod
    def _plan_error_prompt(plan: Plan) -> str:
        """Generate a user prompt when the planner reports an error action."""
        from src.tree.plan import ActionNode, CompositeNode, DecoratorNode

        error_reasons: list[str] = []

        def traverse(node):
            if isinstance(node, ActionNode):
                if node.action_type == "error":
                    reason = node.params.get("message") if isinstance(node.params, dict) else None
                    error_reasons.append(
                        reason or node.name or "Planner cannot perform the requested action."
                    )
            elif isinstance(node, CompositeNode):
                for child in node.children:
                    traverse(child)
            elif isinstance(node, DecoratorNode):
                traverse(node.child)

        traverse(plan.root)
        reason_text = "; ".join(error_reasons) if error_reasons else "Planner reported the task cannot be executed."
        return (
            f"Planner cannot execute the task as requested: {reason_text} "
            "Please clarify the task or provide an alternative instruction."
        )

    def get_execution_snapshot(self):
        """Return the latest plan execution snapshot."""
        return self.executor.get_snapshot()

    def cancel_current_execution(self) -> None:
        """Cancel the active plan execution and clear the active task marker."""
        if self.planning:
            self.stop_planning = True
        if self.executor.is_running:
            log.info("Cancelling active execution for robot %s", self.robot_id)
            self.executor.stop(wait=False)
        self._active_task = None

    def stop_session(self) -> None:
        """Stop the session loop and any active execution."""
        self.session_running = False
        self._task_event.set()
        self._active_task = None
        self._pending_task = None
        if self.executor.is_running:
            log.error('Executor stop function called by stop session!')
            self.executor.stop(wait=True)
        if self.context.modules is not None:
            close = getattr(self.context.modules, "close", None)
            if close is not None:
                close()
        log.info("Stopping session for robot %s", self.robot_id)

    def _publish_task_status(self, message: str) -> None:
        """Publish task status to ROS2 subscribers when available."""
        log.info("Task status [%s]: %s", self.robot_id, message)
        if self.ros2_wrapper is not None:
            self.ros2_wrapper.publish_task_status(message)

    def _set_current_operation(
        self,
        label: str,
        message: str,
        *,
        status: str = "running",
        current_action_name: str | None = None,
        current_action_index: int = 1,
        total_actions: int = 1,
        percent_completed: float | None = None,
    ) -> None:
        """Publish a session-level operation for UI feedback outside BT ticks."""
        started_at = time.monotonic()
        self._operation_started_at = started_at
        action_name = current_action_name or message
        operation = {
            "sequence_label": label,
            "current_action_name": action_name,
            "current_action_index": current_action_index,
            "total_actions": total_actions,
            "elapsed_s": 0.0,
            "current_action_elapsed_s": 0.0,
            "started_monotonic": started_at,
            "status": status,
            "message": message,
        }
        if percent_completed is not None:
            operation["percent_completed"] = float(percent_completed)

        if self.context.world_state is not None:
            self.context.world_state.set("current_operation", operation)

        log.info(f"Operation state [{self.robot_id}]: {label} - {message}")
        if self.ros2_wrapper is not None:
            self.ros2_wrapper.publish_gui_state(
                robot_status=f"{label} {status.replace('_', ' ').title()}",
                active_node=f"{label}: {action_name}",
                eta="Elapsed 0.0s",
            )

    def request_user_input(self, prompt: str, *, choices: list[dict[str, str]] | None = None) -> str:
        """Request user input during task execution."""
        if self.ros2_wrapper is not None:
            result = self.ros2_wrapper.request_user_input(prompt, choices=choices)
            if result:
                return result

        flush_input_buffer()
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                return input(f"{prompt} ")
            except EOFError:
                log.warn("Terminal user input not available for prompt: %s", prompt)
                return ""

        log.warn("User input requested but not available: %s", prompt)
        return ""

    def set_user_input_callback(
        self, callback: Optional[Callable[[str], str]]
    ) -> None:
        """Set a callback for user input requests."""
        if self.ros2_wrapper is not None:
            self.ros2_wrapper.set_user_input_callback(callback)


class Ros2RobotSessionNode:
    """ROS2 Node wrapper for RobotSession with task subscription."""

    def __init__(self, robot_id: str, session: RobotSession):
        """Initialize ROS2 node and task topics."""
        self.robot_id = robot_id
        self.session = session

        import rclpy

        self.node = rclpy.create_node(f"robot_{robot_id}")
        self._user_input_callback: Optional[Callable[[str], str]] = None
        self._active_goal_handle = None
        self._cancel_requested = False
        self._goal_started_monotonic: float | None = None
        self._followup_url = os.getenv(
            "ORCHID_UI_FOLLOWUP_URL",
            os.getenv("ORCHID_UI_URL", "http://localhost:8000/api/followup"),
        )
        self._followup_timeout = float(os.getenv("ORCHID_UI_FOLLOWUP_TIMEOUT", "300"))
        self._setup_task_subscriber()
        self._setup_task_action_server()
        self._session_instance_id = uuid.uuid4().hex
        from std_msgs.msg import String
        from rclpy.clock import Clock, ClockType
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        self._heartbeat_group = MutuallyExclusiveCallbackGroup()
        self._heartbeat_publisher = self.node.create_publisher(
            String, f"{self.robot_id}/session_heartbeat", _latest_reliable_qos())
        self._heartbeat_timer = self.node.create_timer(
            1.0, self._publish_heartbeat, clock=Clock(clock_type=ClockType.STEADY_TIME),
            callback_group=self._heartbeat_group)

        log.set_ui_publisher(self._publish_ui_log)
        speech_mode = log.get_speech_mode()
        if speech_mode == 'tmc':
            try:
                from tmc_voice_msgs.msg import Voice
            except ImportError as exc:
                raise RuntimeError('tmc speech mode requires tmc_voice_msgs') from exc
            topic = (self.session.manifest.hardware or {}).get('speech_topic', '/talk_request')
            self._tmc_speech_publisher = self.node.create_publisher(Voice, topic, 1)
            log.set_speech_publisher(self._publish_tmc_speech)
        else:
            log.set_speech_publisher(self._publish_ui_speech if speech_mode == 'current' else None)
        log.info(
            "ROS2 node created for robot %s (%s)",
            self.robot_id,
            self.node.get_name(),
        )

    def setup_workflows(self) -> None:
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rclpy.clock import Clock, ClockType
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._catalog_publisher = self.node.create_publisher(String, f"{self.robot_id}/workflow_catalog", qos)
        self._catalog_group = MutuallyExclusiveCallbackGroup()
        self._catalog_source = None
        self._catalog_revision = None
        self._catalog_message = None
        self._catalog_subscriptions = 0
        self._catalog_timer = self.node.create_timer(1.0, self._publish_workflow_catalog,
            clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=self._catalog_group)
        self._publish_workflow_catalog()
        try:
            from orchid_ros_interfaces.action import ExecuteWorkflow
            from rclpy.action import ActionServer
            self.ros2_workflow_type = ExecuteWorkflow
            self._workflow_action_server = ActionServer(self.node, ExecuteWorkflow, f"{self.robot_id}/execute_workflow",
                execute_callback=self._execute_task_callback, goal_callback=self._goal_callback, cancel_callback=self._cancel_callback)
        except ImportError:
            log.warn("ExecuteWorkflow interface missing; rebuild/source orchid_ros_interfaces")

    def _publish_workflow_catalog(self) -> None:
        from std_msgs.msg import String
        graph = self.session.context.scene_graph
        source = (id(graph), getattr(graph, "revision", 0))
        count = self._catalog_publisher.get_subscription_count()
        new_subscriber = count > getattr(self, "_catalog_subscriptions", 0)
        self._catalog_subscriptions = count
        if source == self._catalog_source:
            # Replay cached metadata for a bridge that subscribed before publisher discovery.
            if new_subscriber and getattr(self, "_catalog_message", None) is not None:
                self._catalog_publisher.publish(self._catalog_message)
            return
        try:
            document = self.session.workflow_catalog.document()
            document["session_id"] = self._session_instance_id
            if document["revision"] != self._catalog_revision:
                self._catalog_message = String(data=json.dumps(document))
                self._catalog_publisher.publish(self._catalog_message)
                self._catalog_revision = document["revision"]
            self._catalog_source = source
        except Exception as exc:
            log.warn("Workflow catalog update failed: %s", exc)

    def _publish_heartbeat(self) -> None:
        """Wall-clock availability, independent of simulated time and BT progress."""
        from std_msgs.msg import String
        with self.session._task_lock:
            task = self.session._active_task or self.session._pending_task
            request = getattr(self.session, "_active_request", None) if self.session._active_task else self.session._pending_task
        message = String()
        message.data = json.dumps({
            "session_id": self._session_instance_id,
            "ready": getattr(self, "_action_server", None) is not None,
            "busy": bool(task or self.session.executor.is_running),
            "task_description": str(task) if task else "",
            "task_kind": "workflow" if isinstance(request, WorkflowRequest) else "task",
        })
        self._heartbeat_publisher.publish(message)

    def _setup_task_subscriber(self) -> None:
        """Create task request subscription and status publisher."""
        try:
            from std_msgs.msg import String
        except ImportError:
            log.error("Required ROS2 message types not available")
            return

        self.node.create_subscription(
            String,
            f"{self.robot_id}/task_request",
            self._handle_task_request,
            _latest_reliable_qos(),
        )
        self.task_publisher = self.node.create_publisher(
            String,
            f"{self.robot_id}/task_status",
            _latest_reliable_qos(),
        )
        self.gui_state_publisher = self.node.create_publisher(
            String,
            f"{self.robot_id}/gui_state",
            _latest_reliable_qos(),
        )
        log.info("Task topics created for robot %s", self.robot_id)

    def _setup_task_action_server(self) -> None:
        """Create a ROS2 action server for task intake when the interface is available."""
        try:
            from orchid_ros_interfaces.action import ExecuteTask
            from rclpy.action import ActionServer, CancelResponse, GoalResponse
        except ImportError:
            log.warn("ROS2 action interface not available; skipping execute_task action server")
            return

        self._action_server = ActionServer(
            self.node,
            ExecuteTask,
            f"{self.robot_id}/execute_task",
            execute_callback=self._execute_task_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )
        self.ros2_action_type = ExecuteTask
        log.info("Action server created for robot %s", self.robot_id)

    def _goal_callback(self, goal_request) -> object:
        try:
            if hasattr(goal_request, "workflow_id"):
                self.session.workflow_catalog.validate(goal_request.workflow_id, json.loads(goal_request.inputs_json or "{}"))
            elif not goal_request.task_description.strip():
                return GoalResponse.REJECT
        except Exception as exc:
            log.warn("Task goal rejected: %s", exc)
            return GoalResponse.REJECT
        if not self.session.reserve_task():
            log.warn("Task goal rejected: robot is busy")
            return GoalResponse.REJECT
        self._cancel_requested = False
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> object:
        self._cancel_requested = True
        log.error("Action server cancellation requested", self.robot_id, goal_handle.goal_id.uuid)
        self.publish_task_status(f"Triggered Goal Cancellation: {goal_handle.goal_id.uuid}")
        return CancelResponse.ACCEPT

    def _execute_task_callback(self, goal_handle) -> object:
        workflow = hasattr(goal_handle.request, "workflow_id")
        action_type = self.ros2_workflow_type if workflow else self.ros2_action_type
        task_description = goal_handle.request.workflow_id if workflow else goal_handle.request.task_description.strip()

        self._active_goal_handle = goal_handle
        self._goal_started_monotonic = time.monotonic()
        try:
            accepted = self.session.submit_workflow(goal_handle.request.workflow_id, json.loads(goal_handle.request.inputs_json or "{}"), reserved=True) if workflow else self.session.submit_task(task_description, reserved=True)
        except Exception as exc:
            log.warn("Workflow intake failed: %s", exc)
            accepted = False
        self.publish_task_status(f"accepted_action: {task_description}" if accepted else f"ignored_action: {task_description}")

        feedback = action_type.Feedback()
        feedback.current_action = "queued"
        feedback.percent_completed = 0.0
        goal_handle.publish_feedback(feedback)

        if not accepted:
            result = action_type.Result()
            result.success = False
            result.result_message = f"Task already queued/executing: {task_description}"
            goal_handle.abort()
            self._active_goal_handle = None
            self._goal_started_monotonic = None
            return result

        while True:
            if self._cancel_requested:
                self._cancel_requested = False
                self.session.cancel_current_execution()
                self.publish_task_status(f"Cancelling task: {task_description}")
                result = action_type.Result()
                result.success = False
                result.result_message = f"Cancelled task: {task_description}"
                goal_handle.canceled()
                self._active_goal_handle = None
                self._goal_started_monotonic = None
                return result

            with self.session._task_lock:
                active_task = self.session._active_task
                pending_task = self.session._pending_task

            snapshot = self.session.get_execution_snapshot()
            operation = self._current_operation()
            feedback = action_type.Feedback()
            feedback.current_action = self._feedback_message(operation, snapshot, active_task)
            feedback.percent_completed = self._feedback_percent(operation, snapshot)
            goal_handle.publish_feedback(feedback)
            self.publish_gui_state(
                robot_status=self._robot_status(operation, snapshot, active_task, pending_task),
                active_node=self._active_node(operation, snapshot),
                eta=self._elapsed_label(operation),
            )

            if active_task is None and pending_task is None and not self.session.executor.is_running:
                break

            time.sleep(0.2)

        result = action_type.Result()
        if self.session.get_execution_snapshot().status == ExecutionStatus.SUCCEEDED and self._current_operation().get("status") == "success":
            result.success = True
            result.result_message = f"Completed task: {task_description}"
            goal_handle.succeed()
        else:
            result.success = False
            result.result_message = f"Task failed: {task_description}"
            goal_handle.abort()
        self._active_goal_handle = None
        self._goal_started_monotonic = None
        return result

    def _handle_task_request(self, msg) -> None:
        """Forward a received task string into the session queue."""
        task_description = msg.data.strip()
        log.info("Received task for %s: %s", self.robot_id, task_description)

        if not task_description:
            self.publish_task_status("rejected: empty task")
            return

        accepted = self.session.submit_task(task_description)
        if accepted:
            self.publish_task_status(f"accepted: {task_description}")
        else:
            self.publish_task_status(f"ignored: {task_description}")

    def publish_task_status(self, message: str) -> None:
        """Publish a task status update."""
        try:
            from std_msgs.msg import String
        except ImportError:
            log.error("Required ROS2 message types not available")
            return

        status_msg = String()
        status_msg.data = message
        self.task_publisher.publish(status_msg)

    def publish_gui_state(
        self,
        *,
        robot_status: str,
        active_node: str,
        eta: str = "N/A",
        linear_speed: float = 0.0,
        angular_speed: float = 0.0,
    ) -> None:
        """Publish UI diagnostics using the existing Orchid UI gui_state schema."""
        try:
            from std_msgs.msg import String
        except ImportError:
            log.error("Required ROS2 message types not available")
            return

        payload = {
            "robot_status": robot_status,
            "session_id": self._session_instance_id,
            "eta": eta,
            "active_node": active_node,
            "joint_states": {
                "linear_speed": float(linear_speed),
                "angular_speed": float(angular_speed),
            },
        }
        operation = self._current_operation()
        snapshot = self.session.get_execution_snapshot()
        ongoing = bool(getattr(self.session,'_active_task',None) or getattr(self.session,'_pending_task',None)) or snapshot.status == ExecutionStatus.RUNNING
        stage = getattr(snapshot,'task_stage',None) if snapshot.status == ExecutionStatus.RUNNING else None
        if ongoing and not stage:
            label=(operation or {}).get('sequence_label','Execution')
            stage={'Planning':'Planning','Replanning':'Planning','Execution':'Executing','Task':'Starting','Cancellation':'Cancelling'}.get(label,label)
        payload['task_stage'] = stage if ongoing else None
        if operation is not None:
            operation_payload = dict(operation)
            started_monotonic = operation_payload.pop("started_monotonic", None)
            if started_monotonic is not None:
                operation_payload["elapsed_s"] = max(
                    0.0,
                    time.monotonic() - float(started_monotonic),
                )
            payload["operation"] = operation_payload
        world_state=self.session.context.world_state
        payload["scene_highlights"]={
            "navigation_target":world_state.get("scene_navigation_target"),
            "known_objects":world_state.get("scene_known_objects") or [],
        }
        if self._goal_started_monotonic is not None:
            payload["task_elapsed_s"] = max(
                0.0,
                time.monotonic() - self._goal_started_monotonic,
            )
        msg = String()
        msg.data = json.dumps(payload)
        if not hasattr(self, "gui_state_publisher"):
            self.gui_state_publisher = self.node.create_publisher(
                String,
                f"{self.robot_id}/gui_state",
                _latest_reliable_qos(),
            )
        self.gui_state_publisher.publish(msg)

    def publish_gui_state_from_status(self, message: str) -> None:
        """Publish a coarse gui_state update for status-only session transitions."""
        status = message.split(":", 1)[0].strip().replace("_", " ").title()
        self.publish_gui_state(
            robot_status=status or "Unknown",
            active_node=message,
            eta="N/A",
        )

    def _current_operation(self) -> dict[str, Any] | None:
        world_state = self.session.context.world_state
        if world_state is None:
            return None
        operation = world_state.get("current_operation")
        return operation if isinstance(operation, dict) else None

    @staticmethod
    def _feedback_message(
        operation: dict[str, Any] | None,
        snapshot,
        active_task: str | None,
    ) -> str:
        if operation and operation.get("message"):
            message = str(operation["message"])
            if operation.get("started_monotonic") is not None:
                return f"{message} ({Ros2RobotSessionNode._operation_elapsed(operation):.1f}s)"
            return message
        return snapshot.goal or ("executing" if active_task else "queued")

    @staticmethod
    def _feedback_percent(operation: dict[str, Any] | None, snapshot) -> float:
        if operation:
            if operation.get("percent_completed") is not None:
                return min(100.0, max(0.0, float(operation["percent_completed"])))
            total = int(operation.get("total_actions") or 0)
            index = int(operation.get("current_action_index") or 0)
            if total > 0 and index > 0:
                return min(95.0, max(0.0, 10.0 + ((index - 1) / total) * 85.0))
        return min(95.0, 10.0 + (snapshot.tick_count * 2.0))

    @staticmethod
    def _robot_status(
        operation: dict[str, Any] | None,
        snapshot,
        active_task: str | None,
        pending_task: str | None,
    ) -> str:
        if operation and operation.get("sequence_label"):
            status = str(operation.get("status") or "running").replace("_", " ").title()
            return f"{operation['sequence_label']} {status}"
        if active_task:
            return "Executing"
        if pending_task:
            return "Queued"
        return snapshot.status.value.replace("_", " ").title()

    @staticmethod
    def _active_node(operation: dict[str, Any] | None, snapshot) -> str:
        if operation and operation.get("current_action_name"):
            return f"{operation.get('sequence_label')}: {operation.get('current_action_name')}"
        if operation and operation.get("sequence_label"):
            return str(operation["sequence_label"])
        return snapshot.goal or "None"

    @staticmethod
    def _elapsed_label(operation: dict[str, Any] | None) -> str:
        if operation and (
            operation.get("elapsed_s") is not None
            or operation.get("started_monotonic") is not None
        ):
            return f"Elapsed {Ros2RobotSessionNode._operation_elapsed(operation):.1f}s"
        return "N/A"

    @staticmethod
    def _operation_elapsed(operation: dict[str, Any]) -> float:
        started_monotonic = operation.get("started_monotonic")
        if started_monotonic is not None:
            return max(0.0, time.monotonic() - float(started_monotonic))
        return max(0.0, float(operation.get("elapsed_s") or 0.0))

    def _publish_ui_log(self, payload: dict[str, Any]) -> None:
        try:
            from std_msgs.msg import String
        except ImportError:
            return

        try:
            msg = String()
            msg.data = json.dumps(payload)
            self._ui_log_publisher.publish(msg)
        except AttributeError:
            self._ui_log_publisher = self.node.create_publisher(
                String,
                f"{self.robot_id}/gui_logs",
                _latest_reliable_qos(),
            )
            msg = String()
            msg.data = json.dumps(payload)
            self._ui_log_publisher.publish(msg)

    def _publish_ui_speech(self, text: str) -> None:
        try:
            from std_msgs.msg import String
        except ImportError:
            return

        try:
            msg = String()
            msg.data = str(text)
            self._ui_speech_publisher.publish(msg)
        except AttributeError:
            self._ui_speech_publisher = self.node.create_publisher(
                String,
                f"{self.robot_id}/ui_speech",
                _latest_reliable_qos(),
            )
            msg = String()
            msg.data = str(text)
            self._ui_speech_publisher.publish(msg)

    def _publish_tmc_speech(self, text: str) -> None:
        from tmc_voice_msgs.msg import Voice
        msg = Voice()
        msg.interrupting = False
        msg.queueing = False
        msg.language = 1
        msg.sentence = str(text)
        self._tmc_speech_publisher.publish(msg)

    def set_user_input_callback(
        self, callback: Optional[Callable[[str], str]]
    ) -> None:
        """Set callback for user input during task execution."""
        self._user_input_callback = callback

    def _request_user_input_via_rest(self, prompt: str, choices: list[dict[str, str]] | None = None) -> str:
        if not self._followup_url:
            return ""

        payload = json.dumps({"prompt": prompt, "choices": choices or [], "timeout_s": self._followup_timeout}).encode("utf-8")
        request = urllib.request.Request(
            self._followup_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._followup_timeout + 10.) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body) if body else {}
                if data.get("error") == "timeout" or data.get("response") == "timeout":
                    raise TimeoutError(f"User input timed out after {self._followup_timeout:g} seconds")
                return str(data.get("response", ""))
        except urllib.error.HTTPError as exc:
            if exc.code == 408:
                raise TimeoutError(f"User input timed out after {self._followup_timeout:g} seconds") from exc
            raise ConnectionError(f"User-input broker HTTP error: {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            log.warn("REST follow-up request failed: %s", exc)
            if isinstance(exc, TimeoutError):
                raise
            raise ConnectionError(f"User-input broker failed: {exc}") from exc

    def request_user_input(self, prompt: str, *, choices: list[dict[str, str]] | None = None) -> str:
        """Request user input during execution.

        Args:
            prompt: Prompt to display

        Returns:
            User input string
        """
        if self._user_input_callback:
            text = prompt + ("\n" + "\n".join(f"{c['value']}: {c['label']}" for c in choices) if choices else "")
            return self._user_input_callback(text)

        rest_response = self._request_user_input_via_rest(prompt, choices)
        if rest_response:
            return rest_response

        log.warn(f"No user input callback set: {prompt}")
        return ""
