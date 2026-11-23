"""Native HSR pickup workflow. Planning leaves cannot dispatch robot motion."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal
import numpy as np
import py_trees

from src.core.action import AbstractAction, ActionResult, AsyncAction
from src.core.failure import FailureCode, FailureInfo, MotionFailure
from src.core.hsr_pickup import ApproachMode, PICKUP_JOINTS, PickupBackendSettings, PickupPlan, PickupRequest, PickupSnapshot
from src.plugin.planners.hsr_pickup import HsrPickupPlanner, environment_digest, pickup_observation, import_simple_grasp
from src.tree.bt_node_wrapper import ActionScope


class PrepareHsrPickupScene(AsyncAction):
    """Explicit scene writes only: clear map, model target, await fresh depth."""
    def __init__(self, target: str, collision_hull_inflation_factor: float = 2.,
                 output_key: str | None = None, timeout_s: float = 15.):
        self.target = target
        self.collision_hull_inflation_factor = collision_hull_inflation_factor
        self.output_key = output_key
        self.timeout_s = timeout_s
        self._generation = 0
        self._lock = threading.RLock()
        self._planner = None
        self._result = None

    def start(self, context):
        from src.plugin.planners.hsr_pickup_scene import prepare_pickup_scene
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._cancel_requested = False
            self._result = None
        def cancelled():
            return (generation != self._generation or self._cancel_requested or
                    context.execution.control is not None or context.execution.cancel_requested)
        try:
            if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
                raise ValueError("scene refresh timeout must be positive")
            observation = pickup_observation(context, self.target)
            request = PickupRequest(self.target, collision_hull_inflation_factor=self.collision_hull_inflation_factor)
            self._planner = self._planner or HsrPickupPlanner(context.ros2_node, PickupBackendSettings.from_hardware(context.hardware))
            output_key = self.output_key or f"{self.target}_prepared_pickup_snapshot"
        except Exception as exc:
            self._result = ActionResult(status="failure",reason=str(exc),failure=FailureInfo(FailureCode.INVALID_REQUEST))
            return
        def work():
            if cancelled():
                raise MotionFailure("scene preparation cancelled", FailureInfo(FailureCode.CANCELLED))
            captured = self._planner.capture(context, request.target, request.collision_hull_inflation_factor, observation=observation)
            return prepare_pickup_scene(self._planner, captured, cancelled, self.timeout_s)
        def finish(snapshot):
            with self._lock:
                if not cancelled():
                    context.world_state.set(output_key,snapshot)
                    self._result = ActionResult(status="success",payload={"snapshot_key":output_key,"directory":snapshot.directory})
        def fail(exc):
            with self._lock:
                if not cancelled():
                    self._result = ActionResult(status="failure",reason=str(exc),failure=getattr(exc,"failure",FailureInfo()))
        context.run_in_context_loop(work,callback=finish,error_callback=fail)

    def check_result(self, context):
        return self._result or ActionResult(status="running")

    def cancel(self, context):
        with self._lock:
            self._cancel_requested = True
            self._generation += 1


class PlanHsrPickup(AsyncAction):
    """Prepare a complete pickup; observations must already contain a simple grasp.

    This leaf only reads ROS state, runs isolated planning and publishes RViz.
    snapshot_key optionally reuses one explicit capture across mode comparisons.
    """
    # "base_only", "arm_only", "coordinated"
    def __init__(self, target: str, approach_mode: ApproachMode = "coordinated",
                 backoff_direction: Literal["grasp_axis", "horizontal_projection"] = "grasp_axis", pregrasp_distance_m: float = .25,
                 min_insertion_m: float = .04, max_insertion_m: float = .30,
                 clearance_m: float = .03, retreat_distance_m: float = .12,
                 post_grasp_lift_m: float = .05, collision_hull_inflation_factor: float = 2.,
                 max_candidates: int = 8, snapshot_key: str | None = None,
                 output_key: str | None = None, debug_visualization: bool = True,
                 open_width_m: float = .15):
        super().__init__()
        self.target = target
        self.approach_mode = approach_mode
        self.backoff_direction = backoff_direction
        self.pregrasp_distance_m = pregrasp_distance_m
        self.min_insertion_m = min_insertion_m
        self.max_insertion_m = max_insertion_m
        self.clearance_m = clearance_m
        self.retreat_distance_m = retreat_distance_m
        self.post_grasp_lift_m = post_grasp_lift_m
        self.collision_hull_inflation_factor = collision_hull_inflation_factor
        self.open_width_m = open_width_m
        self.max_candidates = max_candidates
        self.snapshot_key = snapshot_key
        self.output_key = output_key
        self.debug_visualization = debug_visualization
        self._lock = threading.RLock()
        self._generation = 0
        self._result = None
        self._planner = None

    def start(self, context):
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._cancel_requested = False
            self._result = None
        def cancelled():
            with self._lock:
                return (generation != self._generation or self._cancel_requested or
                        context.execution.control is not None or context.execution.cancel_requested)
        try:
            request = PickupRequest(target=self.target, approach_mode=self.approach_mode,
                                    backoff_direction=self.backoff_direction, pregrasp_distance_m=self.pregrasp_distance_m,
                                    min_insertion_m=self.min_insertion_m, max_insertion_m=self.max_insertion_m,
                                    clearance_m=self.clearance_m, retreat_distance_m=self.retreat_distance_m,
                                    post_grasp_lift_m=self.post_grasp_lift_m,
                                    collision_hull_inflation_factor=self.collision_hull_inflation_factor,
                                    max_candidates=self.max_candidates, open_width_m=self.open_width_m)
            observation = pickup_observation(context, self.target)
            snapshot = context.world_state.get(self.snapshot_key) if self.snapshot_key else None
            if snapshot is not None and not isinstance(snapshot, PickupSnapshot):
                raise ValueError("snapshot_key must refer to PickupSnapshot")
            settings = PickupBackendSettings.from_hardware(context.hardware)
            if self._planner is None:
                self._planner = HsrPickupPlanner(context.ros2_node, settings)
            output_key = self.output_key or f"{self.target}_{self.approach_mode}_pickup_plan"
        except Exception as exc:
            self._result = ActionResult(status="failure", reason=str(exc), failure=FailureInfo(FailureCode.INVALID_REQUEST))
            return

        def work():
            if cancelled():
                raise MotionFailure("cancelled", FailureInfo(FailureCode.CANCELLED))
            captured = snapshot or self._planner.capture(context, request.target, request.collision_hull_inflation_factor, observation=observation)
            return self._planner.plan(captured, request, cancelled)

        def finish(plan):
            with self._lock:
                if cancelled():
                    return
                context.world_state.set(output_key, plan)
                if self.snapshot_key:
                    context.world_state.set(self.snapshot_key, plan.snapshot)
                if self.debug_visualization:
                    self._planner.publish(plan)
                self._result = ActionResult(status="success" if plan.phases else "failure",
                    reason=plan.diagnostics["reason"], payload={"plan_key": output_key, "artifact": plan.diagnostics["artifact"],
                    "diagnostics": plan.diagnostics, "candidates": plan.candidates},
                    failure=None if plan.phases else FailureInfo(FailureCode.PLANNING_FAILED))

        def fail(exc):
            with self._lock:
                if not cancelled():
                    self._result = ActionResult(status="failure", reason=str(exc), failure=getattr(exc, "failure", FailureInfo(FailureCode.INVALID_REQUEST) if isinstance(exc, ValueError) else FailureInfo()))

        context.run_in_context_loop(work, callback=finish, error_callback=fail)

    def check_result(self, context):
        with self._lock:
            return self._result or ActionResult(status="running")

    def cancel(self, context):
        with self._lock:
            self._cancel_requested = True
            self._generation += 1


class ExecuteHsrPickupPhase(AsyncAction):
    """Execute one stored phase after validating its current start/scene."""
    required_controllers = ["arm"]

    def __init__(self, plan_key: str, phase: Literal["staging", "insertion", "lift", "retreat"], start_tolerance: float = .01):
        super().__init__()
        self.plan_key = plan_key
        self.phase = phase
        self.start_tolerance = start_tolerance
        self._lock = threading.RLock()
        self._generation = 0
        self._dispatched = False
        self._dispatch_complete = False
        self._cancel_sent = False
        self._result = None
        self._planner = None

    def start(self, context):
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._cancel_requested = False
            self._dispatched = False
            self._dispatch_complete = False
            self._cancel_sent = False
            self._result = None
        plan = context.world_state.get(self.plan_key)
        if not isinstance(plan, PickupPlan) or not plan.phases:
            self._result = ActionResult(status="failure", reason="missing prepared pickup plan", failure=FailureInfo(FailureCode.INVALID_STATE))
            return
        try:
            phase = plan.phase(self.phase)
            if not np.isfinite(self.start_tolerance) or self.start_tolerance <= 0:
                raise ValueError("start_tolerance must be positive")
        except (ValueError, StopIteration):
            self._result = ActionResult(status="failure", reason="invalid pickup phase", failure=FailureInfo(FailureCode.INVALID_REQUEST))
            return
        settings = PickupBackendSettings.from_hardware(context.hardware)
        self._planner = self._planner or HsrPickupPlanner(context.ros2_node, settings)

        def current():
            return (generation == self._generation and not self._cancel_requested and
                    context.execution.control is None and not context.execution.cancel_requested)

        def work():
            from moveit_msgs.srv import GetPlanningScene
            from src.core.collision_geometry import write_collision_mesh
            from rclpy.serialization import deserialize_message
            from moveit_msgs.msg import CollisionObject
            req = GetPlanningScene.Request()
            req.components.components = 1023
            scene = self._planner._call("/get_planning_scene", GetPlanningScene, req).scene
            measured = dict(zip(scene.robot_state.joint_state.name, scene.robot_state.joint_state.position))
            # if any(n not in measured or abs(measured[n] - phase.start_joints[n]) > self.start_tolerance for n in PICKUP_JOINTS):
                # raise MotionFailure("prepared pickup start state changed", FailureInfo(FailureCode.INVALID_STATE))
            target_id = f"orchid_pickup_{plan.target}"
            # if environment_digest(scene, target_id) != plan.snapshot.scene_digest:
            #     raise MotionFailure("prepared pickup environment changed", FailureInfo(FailureCode.INVALID_STATE))
            observation = pickup_observation(context, plan.target)
            # An observation in another frame requires a fresh plan, not relabeling.
            current_grasp, _ = import_simple_grasp(observation.grasp, observation.approach_axis)
            if observation.frame_id != plan.snapshot.frame_id:
                tf = context.sensor_data_center.get_pose_at(observation.frame_id, plan.snapshot.frame_id)
                if tf is None or "mat" not in tf:
                    raise MotionFailure("prepared pickup target transform missing", FailureInfo(FailureCode.INVALID_STATE))
                current_grasp = np.asarray(tf["mat"]) @ current_grasp
            # if not np.allclose(current_grasp, plan.snapshot.grasp, atol=1e-5):
                # raise MotionFailure("prepared pickup target observation changed", FailureInfo(FailureCode.INVALID_STATE))
            attached = any(obj.object.id == target_id and obj.link_name == "hand_palm_link" for obj in scene.robot_state.attached_collision_objects)
            # if attached != phase.attached:
                # raise MotionFailure("prepared pickup attachment state changed", FailureInfo(FailureCode.INVALID_STATE))
            from src.plugin.planners.hsr_pickup_scene import set_target_contact
            target_contact = self.phase != "staging"
            set_target_contact(scene, target_id, settings.touch_links, target_contact)
            if phase.attached:
                set_target_contact(scene, target_id, ["<octomap>"], settings.allow_held_target_octomap_contact)
            self._planner.validate_phase(plan, phase, scene, lambda: not current())
            arm = context.require_arm()
            if self.phase == "staging":
                import open3d as o3d
                obj = deserialize_message((Path(plan.snapshot.directory) / "target.cdr").read_bytes(), CollisionObject)
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[p.x, p.y, p.z] for p in obj.meshes[0].vertices]),
                                               o3d.utility.Vector3iVector([t.vertex_indices for t in obj.meshes[0].triangles]))
                mesh.compute_vertex_normals()
                path = write_collision_mesh(mesh)
                with self._lock:
                    if not current():
                        return
                    if not arm.add_collision_mesh(path, target_id, frame_id=plan.snapshot.frame_id):
                        raise MotionFailure("target scene installation failed", FailureInfo(FailureCode.INVALID_STATE))
                installed = self._planner._call("/get_planning_scene", GetPlanningScene, req).scene
                if not any(obj.id == target_id and obj.header.frame_id == plan.snapshot.frame_id and obj.meshes
                           for obj in installed.world.collision_objects):
                    raise MotionFailure("target not present in MoveIt scene after installation", FailureInfo(FailureCode.INVALID_STATE))
            if phase.attached and settings.allow_held_target_octomap_contact:
                with self._lock:
                    if not current():
                        return
                    if not arm.allow_collision_pairs([(target_id,"<octomap>")]):
                        raise MotionFailure("held target octomap policy installation failed", FailureInfo(FailureCode.INVALID_STATE))
            if self.phase == "insertion":
                with self._lock:
                    if not current():
                        return
                    if not arm.allow_collision_pairs([(link, target_id) for link in settings.touch_links]):
                        raise MotionFailure("target contact policy installation failed", FailureInfo(FailureCode.INVALID_STATE))
            with self._lock:
                if not current():
                    return
                # Mark ownership before send: a partially dispatched batch also needs cancellation.
                self._dispatched = True
            arm.send_synchronized_trajectory_goal(phase.trajectory, cancelled=lambda: not current())
            with self._lock:
                self._dispatch_complete = True
                if not current() and not self._cancel_sent:
                    self._cancel_sent = True
                    arm.cancel_goal()

        def fail(exc):
            with self._lock:
                if current():
                    self._result = ActionResult(status="needs_replan", reason=str(exc), force_replan=True, failure=getattr(exc, "failure", FailureInfo()))
        context.run_in_context_loop(work, error_callback=fail)

    def check_result(self, context):
        from src.plugin.actions._helpers import _arm_status_result
        with self._lock:
            if self._result:
                return self._result
            if not self._dispatch_complete:
                return ActionResult(status="running")
            return _arm_status_result(context.require_arm())

    def cancel(self, context):
        with self._lock:
            if self._cancel_requested:
                return
            self._cancel_requested = True
            self._generation += 1
            if self._dispatched and self._dispatch_complete and not self._cancel_sent:
                self._cancel_sent = True
                context.require_arm().cancel_goal()


class HsrPickup(AbstractAction):
    """New executable workflow; use PlanHsrPickup for no-motion validation."""
    required_controllers = ["arm", "gripper"]
    def __init__(self, target: str, approach_mode: ApproachMode = "cooth", backoff_direction: Literal["grasp_axis", "horizontal_projection"] = "grasp_axis",
                 pregrasp_distance_m: float = .25, min_insertion_m: float = .04,
                 max_insertion_m: float = .60, clearance_m: float = .03,
                 retreat_distance_m: float = .12, post_grasp_lift_m: float = .05,
                 collision_hull_inflation_factor: float = 1.2, force_N: float = .5, width_m: float = .15):
        self.target = target
        self.approach_mode = approach_mode
        self.backoff_direction = backoff_direction
        self.pregrasp_distance_m = pregrasp_distance_m
        self.min_insertion_m = min_insertion_m
        self.max_insertion_m = max_insertion_m
        self.clearance_m = clearance_m
        self.retreat_distance_m = retreat_distance_m    
        self.post_grasp_lift_m = post_grasp_lift_m
        self.collision_hull_inflation_factor = collision_hull_inflation_factor
        self.force_N = force_N
        self.width_m = width_m

    def build_node(self, context, name):
        from src.plugin.actions.manipulation import ReleaseObjectAction, GraspObjectAction, _AttachCollisionObjectAction
        key = f"{self.target}_{self.approach_mode}_pickup_plan"
        snapshot_key = f"{self.target}_prepared_pickup_snapshot"
        planning = PlanHsrPickup(target=self.target, snapshot_key=snapshot_key, approach_mode=self.approach_mode, backoff_direction=self.backoff_direction,
                                 pregrasp_distance_m=self.pregrasp_distance_m, min_insertion_m=self.min_insertion_m,
                                 max_insertion_m=self.max_insertion_m, clearance_m=self.clearance_m,
                                 retreat_distance_m=self.retreat_distance_m, post_grasp_lift_m=self.post_grasp_lift_m,
                                 collision_hull_inflation_factor=self.collision_hull_inflation_factor, open_width_m=self.width_m)
        self._close = GraspObjectAction(force_N=self.force_N, width_m=self.width_m).build_node(context, "Close once")
        children = [PrepareHsrPickupScene(self.target, self.collision_hull_inflation_factor, snapshot_key).build_node(context, "Prepare target and depth scene"),
                    planning.build_node(context, "Plan complete pickup"),
                    ExecuteHsrPickupPhase(key, "staging").build_node(context, "Stage pregrasp"),
                    ReleaseObjectAction(distance=self.width_m).build_node(context, "Open gripper"),
                    ExecuteHsrPickupPhase(key, "insertion").build_node(context, "Insert"), self._close,
                    _AttachCollisionObjectAction(f"orchid_pickup_{self.target}", "hand_palm_link", list(PickupBackendSettings.from_hardware(context.hardware).touch_links)).build_node(context, "Attach target"),
                    ExecuteHsrPickupPhase(key, "lift").build_node(context, "Lift held target"),
                    ExecuteHsrPickupPhase(key, "retreat").build_node(context, "Retreat with target")]
        if self.approach_mode != "arm_only":
            children.insert(-1, ExecuteHsrPickupPhase(key, "undock").build_node(context, "Undock before arm retraction"))
        self._nodes = children
        return ActionScope(self, context, name, py_trees.composites.Sequence(name="HSR pickup operations", memory=True, children=children))

    def on_complete(self, context, status):
        if status == py_trees.common.Status.SUCCESS:
            context.world_state.set(f"{self.target}_held_state", "commanded")
            return ActionResult(status="success", payload={"holding": "commanded", "target": self.target})
        cause = next((node.result for node in reversed(self._nodes) if node.result is not None and node.result.status not in {"success", "running"}), None)
        if self._close.started:
            holding = "commanded" if self._close.result and self._close.result.status == "success" else "uncertain"
            context.world_state.set(f"{self.target}_held_state", holding)
            return ActionResult(status="needs_replan", reason=cause.reason if cause else "pickup interrupted after closure", force_replan=True,
                                failure=cause.failure if cause else FailureInfo(), payload={"holding": holding, "target": self.target})
        return cause or ActionResult(status="failure", reason="pickup planning or approach failed")

    def on_enter(self, context):
        if context.world_state.get(f"{self.target}_held_state") in {"commanded", "uncertain"}:
            return ActionResult(status="needs_replan", reason="target may already be held", force_replan=True,
                                payload={"target": self.target, "holding": context.world_state.get(f"{self.target}_held_state")})

    def on_cancel(self, context):
        if self._close.started:
            holding = "commanded" if self._close.result and self._close.result.status == "success" else "uncertain"
            context.world_state.set(f"{self.target}_held_state", holding)
