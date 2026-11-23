"""ExecutionEngine: orchestrates plan execution, tree ticking, and replanning.

Manages:
- Tree initialization and ticking
- Context setup and injection
- Execution history tracking
- Background plan execution with observable state
"""

from __future__ import annotations

import threading
from copy import copy
from dataclasses import dataclass
from enum import Enum
import time
from typing import TYPE_CHECKING, Any

import py_trees
from pydantic import ValidationError
import src.core.logger as log

# Optional ROS integration: use py_trees_ros BehaviourTree if available
try:
    from py_trees_ros.trees import BehaviourTree as ROSBehaviourTree
except ImportError:
    log.debug("py_trees_ros.trees.BehaviourTree not available")
    ROSBehaviourTree = None

if TYPE_CHECKING:
    from src.tree.plan import Plan

from src.core.context import Context


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


class ExecutionStatus(str, Enum):
    """Lifecycle state of the most recent plan execution."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlanExecutionSnapshot:
    """Point-in-time view of plan execution progress."""

    status: ExecutionStatus = ExecutionStatus.IDLE
    goal: str = ""
    tick_count: int = 0
    success: bool | None = None
    error: str | None = None
    task_stage: str | None = None


class ExecutionEngine:
    """Orchestrates behavior tree execution for a robot session."""

    def __init__(self, context: Context, ros2_node: Any | None = None):
        """Initialize ExecutionEngine.

        Args:
            context: Pre-built execution context
            ros2_node: Optional existing ROS2 node for tree visualization
        """
        if not context:
            raise ValueError("ExecutionEngine requires a valid Context")

        self.context = context
        self.ros2_node = ros2_node
        self.tree: py_trees.trees.BehaviourTree | None | None = None  # May be replaced with ROSBehaviourTree at runtime
        self.max_ticks = 1000

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._snapshot = PlanExecutionSnapshot()

    @property
    def status(self) -> ExecutionStatus:
        """Current execution status."""
        with self._lock:
            return self._snapshot.status

    @property
    def is_running(self) -> bool:
        """True while a plan is actively ticking."""
        return self.status == ExecutionStatus.RUNNING

    def get_snapshot(self) -> PlanExecutionSnapshot:
        """Return a copy of the current execution snapshot."""
        with self._lock:
            return copy(self._snapshot)

    def set_ros2_node(self, ros2_node: Any | None) -> None:
        """Attach the session ROS2 node used for tree visualization."""
        self.ros2_node = ros2_node

    def build_tree(self, plan: Plan) -> py_trees.trees.BehaviourTree:
        """Build a behavior tree from a plan."""
        from src.tree.builder import TreeBuilder

        try:
            builder = TreeBuilder(self.context)
            tree = builder.build(plan)
            log.debug(
                "ExecutionEngine.build_tree: built tree with root %s",
                tree.root.name,
            )
            return tree
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "ExecutionEngine.build_tree: failed to build tree for plan %r. "
                "Check for unsupported actions, decorators, or node types.",
                plan.goal,
            )
            raise

    def start_plan(
        self,
        plan_or_tree: Plan | py_trees.trees.BehaviourTree,
        tree: py_trees.trees.BehaviourTree | None = None,
    ) -> None:
        """Stop any running plan and execute a plan or prebuilt tree in a background thread."""
        log.error('Executor stop function called by start!')
        self.stop(wait=True)

        if isinstance(plan_or_tree, py_trees.trees.BehaviourTree):
            if tree is not None:
                raise ValueError("Cannot pass both a BehaviourTree and an additional tree")
            plan = None
            tree = plan_or_tree
            goal = tree.root.name
        else:
            plan = plan_or_tree
            goal = plan.goal
            if tree is not None and not isinstance(tree, py_trees.trees.BehaviourTree):
                raise TypeError("tree must be a py_trees BehaviourTree")

        with self._lock:
            self._stop_requested = False
            self._snapshot = PlanExecutionSnapshot(
                status=ExecutionStatus.RUNNING,
                goal=goal,
            )

        thread = threading.Thread(
            target=self._run_plan_in_thread,
            args=(plan, tree),
            name="plan-executor",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def _run_plan_in_thread(
        self,
        plan: Plan | None,
        tree: py_trees.trees.BehaviourTree | None,
    ) -> None:
        """Background worker that executes a plan and updates snapshot state."""
        try:
            success = self._execute_plan(plan=plan, tree=tree)
            with self._lock:
                if self._stop_requested:
                    self._snapshot.status = ExecutionStatus.CANCELLED
                    self._snapshot.success = False
                elif success:
                    self._snapshot.status = ExecutionStatus.SUCCEEDED
                    self._snapshot.success = True
                else:
                    self._snapshot.status = ExecutionStatus.FAILED
                    self._snapshot.success = False
        except Exception as exc:  # noqa: BLE001
            log.exception("ExecutionEngine: unhandled error during plan execution")
            with self._lock:
                self._snapshot.status = ExecutionStatus.FAILED
                self._snapshot.success = False
                self._snapshot.error = str(exc)
        finally:
            self.context.world_state.set("scene_navigation_target",None)
            self.context.world_state.set("scene_known_objects",[])
            with self._lock:
                self._snapshot.task_stage = None
                self._thread = None
                self._stop_requested = False

    def _serialize_tree_state(self, node: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": str(getattr(node, "id", "")),
            "name": getattr(node, "name", "unknown"),
            "status": getattr(getattr(node, "status", None), "value", str(getattr(node, "status", ""))),
            "node_type": type(node).__name__,
        }
        self._add_action_leaf_metadata(node, payload)
        children = getattr(node, "children", None)
        if children:
            payload["children"] = [self._serialize_tree_state(child) for child in children]
        return payload

    def _add_action_leaf_metadata(self, node: Any, payload: dict[str, Any]) -> None:
        binding = self.context.execution.bindings.get(node.id)
        if binding is None:
            return
        action = binding.action
        action_type = binding.action_type
        payload["action_type"] = action_type
        payload["display_name"] = getattr(node, "name", action_type)

        status = payload.get("status")
        monotonic_start = getattr(node, "_monotonic_start_time", 0.0) or 0.0
        if status == "RUNNING" and monotonic_start > 0.0:
            payload["elapsed_s"] = max(0.0, time.monotonic() - monotonic_start)

        duration_s = getattr(node, "_last_duration_s", None)
        if duration_s is not None and status in {"SUCCESS", "FAILURE"}:
            payload["duration_s"] = max(0.0, float(duration_s))

        operation = self._matching_current_operation(
            node_name=payload["name"],
            action_type=action_type,
        )
        if operation is not None and status == "RUNNING":
            payload["operation"] = operation
            if operation.get("elapsed_s") is not None:
                payload["elapsed_s"] = max(0.0, float(operation["elapsed_s"]))

    def _force_replan_requested(self) -> dict[str, Any] | None:
        """Read the engine's latched control request independently of BT status."""
        control = self.context.execution.control
        if control is None:
            return None
        binding = self.context.execution.bindings[control.node_id]
        result = control.result
        return {"action_name": binding.name, "action_type": binding.action_type, "status": result.status,
                "reason": result.reason, "force_replan": result.force_replan,
                "suggested_replan_context": result.suggested_replan_context}

    def _matching_current_operation(
        self,
        *,
        node_name: str,
        action_type: str,
    ) -> dict[str, Any] | None:
        world_state = getattr(self.context, "world_state", None)
        if world_state is None:
            return None
        operation = world_state.get("current_operation")
        if not isinstance(operation, dict):
            return None
        labels = {
            str(operation.get("sequence_label") or ""),
            str(operation.get("current_action_name") or ""),
        }
        if action_type not in labels and node_name not in labels:
            return None

        operation_payload = dict(operation)
        started_monotonic = operation_payload.pop("started_monotonic", None)
        if started_monotonic is not None:
            operation_payload["elapsed_s"] = max(
                0.0,
                time.monotonic() - float(started_monotonic),
            )
        return operation_payload

    def _publish_behavior_tree(self, tree_payload: dict[str, Any]) -> None:
        try:
            from std_msgs.msg import String
            import json
            msg = String()
            msg.data = json.dumps(tree_payload)
            self.tree_publisher.publish(msg)
        except Exception as e:
            log.exception('Tree publish failed', e)

    def _create_tree_pub(self):
        try:
            from std_msgs.msg import String
        except ImportError:
            log.error("Required ROS2 message types not available")
            return
        self.tree_publisher = self.ros2_node.create_publisher(
            String,
            f"{self.context.robot_id}/behavior_tree",
            _latest_reliable_qos(),
        )

    def _publish_tree_state(self, tree: py_trees.trees.BehaviourTree | None) -> None:
        if tree is None or self.ros2_node is None: return
        if not hasattr(self, "tree_publisher"): self._create_tree_pub()
        self._publish_behavior_tree(self._serialize_tree_state(tree.root))

    def _execute_plan(
        self,
        plan: Plan | None = None,
        tree: py_trees.trees.BehaviourTree | None = None,
    ) -> bool:
        """Build and tick the behavior tree until completion or cancellation."""
        if not self.context:
            raise RuntimeError("Context not passed")

        if tree is None:
            if plan is None:
                raise RuntimeError("No plan or tree provided for execution")
            tree = self.build_tree(plan)

        # If ROS behaviour tree is available and a ROS node is provided, wrap the tree
        if ROSBehaviourTree is not None and self.ros2_node is not None:
            try:
                # Create a ROS-enabled behaviour tree using the same root
                if self.tree is None:
                    ros_tree = ROSBehaviourTree(root=tree.root)
                    ros_tree.setup(node=self.ros2_node)
                    log.info("ROSBehaviourTree created for execution visualization")
                    self.tree = ros_tree
                else:
                    self.tree.root = tree.root
            except Exception as exc:  # noqa: BLE001
                log.warn("Failed to create ROS BehaviourTree: %s", exc)
                self.tree = tree
        else:
            self.tree = tree

        # Transfer observers along with the root when ROS owns visualization.
        if self.tree is not tree:
            from src.tree.binding_visitor import ActionBindingsVisitor
            self.tree.visitors[:] = [visitor for visitor in self.tree.visitors if not isinstance(visitor, ActionBindingsVisitor)]
            self.tree.visitors.extend(tree.visitors)

        # A previous plan may have ended on a force_replan result; the session
        # has consumed it by now, so clear it before this tree records anything.
        world_state = getattr(self.context, "world_state", None)
        if world_state is not None:
            world_state.set("last_action_result", None)
            world_state.set("scene_navigation_target",None)
            world_state.set("scene_known_objects",[])
        self.context.execution.reset_control()
        active_ids = {node.id for node in self.tree.root.iterate()}
        self.context.execution.bindings = {node_id: binding for node_id, binding in self.context.execution.bindings.items() if node_id in active_ids}
        self.context.execution.stage_labels = {node_id: label for node_id, label in self.context.execution.stage_labels.items() if node_id in active_ids}

        log.info("\n%s", py_trees.display.unicode_tree(self.tree.root))
        self._publish_tree_state(self.tree)

        while (
            not self._stop_requested
            and self.tree.count < self.max_ticks
            and self.tree.root.status
            not in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE)
        ):
            self.tree.tick()
            with self._lock:
                self._snapshot.task_stage = self.context.execution.active_stage(self.tree.root)
            self._publish_tree_state(self.tree)
            if self._force_replan_requested():
                break
            time.sleep(1)  # Adjust tick rate as needed
            if self.tree.count%20 == 0: log.info(f"Tick {self.tree.count}: root status={self.tree.root.status.value}, max_ticks={self.max_ticks}")

        with self._lock:
            self._snapshot.tick_count = self.tree.count

        if self._stop_requested:
            log.error('Stop requested by someone!')
            self.tree.root.stop(py_trees.common.Status.INVALID)
            return False

        last = self._force_replan_requested()
        if last:
            # An action asked for an immediate replan (e.g. InspectScene found
            # something). Tear the tree down so the session sees a failed run
            # and picks up last_action_result; untried siblings never run.
            log.info(
                "ExecutionEngine: %s requested replan, stopping tree",
                last.get("action_name"),
            )
            self.tree.root.stop(py_trees.common.Status.INVALID)
            return False

        success = self.tree.root.status == py_trees.common.Status.SUCCESS
        log.info(
            "ExecutionEngine completed plan goal=%r success=%s ticks=%s",
            plan.goal if plan is not None else tree.root.name,
            success,
            self.tree.count,
        )
        return success

    def stop(self, wait: bool = False, timeout: float | None = None) -> None:
        """Request cancellation of the active plan."""
        self._stop_requested = True
        self.context.execution.cancel_requested = True
        self.context.world_state.set("scene_navigation_target",None)
        self.context.world_state.set("scene_known_objects",[])
        log.error('Executor stop function called!')
        # if self.tree:
        #     self.tree.root.stop(py_trees.common.Status.FAILURE)
        if wait:
            self.join(timeout=timeout)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the background execution thread to finish."""
        thread = None
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def reset(self) -> None:
        """Reset execution state for the next plan."""
        log.error('Executor stop function called by reset!')
        self.stop(wait=True)
        # self.tree = None
        with self._lock:
            self._snapshot = PlanExecutionSnapshot()
