"""Pickup-local ROS planning client; no controller clients or file IPC."""
from __future__ import annotations
from dataclasses import fields, replace
import copy
import threading
import time
import xml.etree.ElementTree as ET
import numpy as np
import src.core.logger as log
from src.core.failure import FailureCode, FailureInfo, MotionFailure
from .model import (ARM_JOINTS, BASE_JOINTS, PICKUP_JOINTS, TargetCollisionGeometry, PickupBackendSettings, PickupInputs,
                    PickupObservation, PickupPlan, PickupRequest, PreparedMotion, pickup_geometry)

def add_pickup_groups(srdf: str, robot_name: str | None = None) -> str:
    root = ET.fromstring(srdf)
    if robot_name:
        root.set("name", robot_name)
    for name, names in (("pickup_arm", ARM_JOINTS), ("pickup_arm_yaw", ("odom_t",) + ARM_JOINTS), ("pickup_whole_body", PICKUP_JOINTS)):
        if root.find(f"group[@name='{name}']") is not None:
            raise ValueError(f"reserved pickup group already exists: {name}")
        group = ET.SubElement(root, "group", name=name)
        # Explicit active joints exclude HSR-C's passive wrist force-sensor joint.
        for joint in names:
            ET.SubElement(group, "joint", name=joint)
    return ET.tostring(root, encoding="unicode")


def pickup_observation(context, target: str) -> PickupObservation:
    ws = context.world_state
    return PickupObservation(
        np.array(ws.get(f"{target}_estimated_pick_pose_mat"), dtype=float, copy=True),
        np.array(ws.get(f"{target}_object_pcd"), dtype=float, copy=True),
        np.array(ws.get(f"{target}_surround_pcd") if ws.get(f"{target}_surround_pcd") is not None else [], dtype=float, copy=True),
        ws.get(f"{target}_estimated_pick_pose_frame") or ws.get(f"{target}_pcd_frame"),
        (ws.get(f"{target}_simple_grasp_metadata") or {}).get("approach_axis", "+z"),
    )


def import_simple_grasp(pose, approach_axis: str):
    """Convert the legacy simple side-grasp basis into HSR hand_palm_link.

    Legacy side estimates use lateral/up/approach (+z), whereas the HSR palm
    uses up/finger-opening/approach. The legacy FK workflow fixes wrist roll
    and aligns approach, so it never enforces that full estimator basis.
    Do not repair handedness while preserving the wrong finger-opening axis.
    Proper palm poses and top-down poses retain their orientation.
    """
    result = np.array(pose, dtype=float, copy=True)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("invalid simple grasp transform")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("invalid simple grasp rotation")
    if approach_axis not in {"+x", "-x", "+z", "-z"}:
        raise ValueError("unsupported simple-estimator approach axis")
    axis = "xyz".index(approach_axis[1])
    approach = rotation[:, axis] * (-1 if approach_axis[0] == "-" else 1)
    up = rotation[:, 1 if axis == 2 else 2]
    legacy_side = abs(approach[2]) < 1e-5 and abs(up[2]) > 1.-1e-5
    if legacy_side:
        # HSR hand +X points up for a horizontal, wrist-roll-zero pickup.
        hand_x = up.copy()
        hand_y = np.cross(approach, hand_x)
        result[:3, :3] = np.column_stack((hand_x, hand_y, approach))
        return result, True
    if approach_axis != "+z" or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
        raise ValueError("unrecognized simple-estimator frame convention")
    return result, False


def trajectory_message(data):
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    result = JointTrajectory()
    result.joint_names = data["joint_names"]
    last = -1.
    for raw in data["points"]:
        stamp = float(raw["time_from_start"])
        if not np.isfinite(stamp) or stamp < 0 or stamp <= last:
            raise ValueError("trajectory timing must be finite and strictly increasing")
        last = stamp
        point = JointTrajectoryPoint()
        for key in ("positions", "velocities", "accelerations"):
            values = [float(v) for v in raw.get(key, [])]
            if (values and len(values) != len(result.joint_names)) or not np.isfinite(values).all():
                raise ValueError("invalid prepared trajectory")
            setattr(point, key, values)
        if len(point.positions) != len(result.joint_names):
            raise ValueError("trajectory positions are required")
        ns = round(stamp * 1e9)
        point.time_from_start = Duration(sec=ns // 10**9, nanosec=ns % 10**9)
        result.points.append(point)
    if not result.points:
        raise ValueError("empty prepared trajectory")
    return result


def duration_seconds(value):
    return value.sec + value.nanosec*1e-9


def candidate_dict(c):
    from scipy.spatial.transform import Rotation
    def pose(p):
        q=p.orientation
        m=np.eye(4)
        if q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w > 0:
            m[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        m[:3,3]=[p.position.x,p.position.y,p.position.z]
        return m.tolist()
    classification = ("collision" if c.contacts else "backend_error" if c.backend_code else
                      {"pregrasp_no_ik":"unreachable","endpoint_no_compatible_ik":"unreachable",
                       "incompatible_base_freedom":"incompatible_base_freedom","endpoint_joint_bounds":"joint_bounds",
                       "pregrasp_joint_bounds":"joint_bounds","joint_bounds":"joint_bounds"}.get(c.reason,"planning_failed"))
    rejected={"joints":dict(zip(c.rejected_joints.name,c.rejected_joints.position)),
              "hand_pose":pose(c.rejected_hand_pose),
              "contacts":[[p.first,p.second] for p in c.contacts],
              "contact_details":[{"first":p.first,"second":p.second,"position":[p.position.x,p.position.y,p.position.z],"depth":p.depth} for p in c.contacts]}
    return {"id":c.id,"mode":c.mode,"distance_m":c.backoff_m,"status":c.status,"pregrasp":pose(c.pregrasp),
            "joints":dict(zip(c.joints.name,c.joints.position)),"staging_order":c.staging_order,
            "failure":{"classification":classification,"retryable":False,"phase":c.operation,"reason":c.reason,"backend_code":c.backend_code,"fraction":c.fraction,
                       "contacts":rejected["contacts"],"rejected_state":rejected}}


class PickupPlanningClient:
    def __init__(self, node, settings: PickupBackendSettings):
        self.node, self.settings = node, settings
        self._clients, self._publishers, self._service_counts = {}, {}, {}
        self._action_client = None

    def _call(self, service, kind, request, timeout=15.):
        self._service_counts[service] = self._service_counts.get(service, 0) + 1
        client = self._clients.get(service)
        if client is None:
            client = self._clients[service] = self.node.create_client(kind, service)
        if not client.wait_for_service(timeout_sec=min(timeout, 5.)):
            raise MotionFailure(f"service unavailable: {service}", FailureInfo(FailureCode.NOT_FOUND))
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            client.remove_pending_request(future)
            raise MotionFailure(f"service timeout: {service}", FailureInfo(FailureCode.PLANNING_FAILED, backend_code="service_timeout"))
        return future.result()

    def collect_inputs(self, context, target: str, inflation: float,
                       observation: PickupObservation | None = None, *,
                       target_collision_geometry: TargetCollisionGeometry = "mesh") -> PickupInputs:
        began = time.monotonic()
        from moveit_msgs.srv import GetPlanningScene
        from rcl_interfaces.srv import GetParameters, ListParameters
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import Mesh, MeshTriangle
        from geometry_msgs.msg import Point, Pose
        from src.core.collision_geometry import inflated_convex_hull, voxel_collision_mesh

        observation = observation or pickup_observation(context, target)
        grasp, corrected_rotation = import_simple_grasp(observation.grasp, observation.approach_axis)
        points = observation.target_points.copy()
        if target_collision_geometry not in {"mesh", "convex_hull"}:
            raise ValueError("target_collision_geometry must be mesh or convex_hull")
        target_mesh = (inflated_convex_hull(points, inflation) if target_collision_geometry == "convex_hull"
                       else voxel_collision_mesh(points))
        log.info("Pickup target collision geometry=%s hull_inflation=%s", target_collision_geometry,
                 inflation if target_collision_geometry == "convex_hull" else "not applied")
        surround = observation.surround_points.copy()
        frame = observation.frame_id
        if not frame or grasp.shape != (4, 4) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("simple grasp estimate, target cloud and frame are required")
        params = self._call("/move_group/get_parameters", GetParameters,
                            GetParameters.Request(names=["robot_description", "robot_description_semantic"]))
        if len(params.values) != 2 or not all(value.string_value for value in params.values):
            raise ValueError("MoveIt robot descriptions missing")
        names = self._call("/move_group/list_parameters", ListParameters,
                           ListParameters.Request(prefixes=["robot_description_planning.joint_limits"], depth=0)).result.names
        limits = {}
        if names:
            values = self._call("/move_group/get_parameters", GetParameters, GetParameters.Request(names=names)).values
            for name, value in zip(names, values):
                if value.type == 1:
                    limits[name] = value.bool_value
                elif value.type == 2:
                    limits[name] = value.integer_value
                elif value.type == 3:
                    limits[name] = value.double_value
        req = GetPlanningScene.Request()
        req.components.components = 1023
        scene = self._call("/get_planning_scene", GetPlanningScene, req).scene
        start = dict(zip(scene.robot_state.joint_state.name, scene.robot_state.joint_state.position))
        if any(name not in start for name in PICKUP_JOINTS):
            raise ValueError("captured robot state is missing pickup joints")
        # Require a known transform; never reinterpret coordinates by relabeling.
        root = ET.fromstring(params.values[0].string_value)
        child_links = {j.find("child").attrib["link"] for j in root.findall("joint")}
        model_frame = next(link.attrib["name"] for link in root.findall("link") if link.attrib["name"] not in child_links)
        transform = np.eye(4)
        if frame != model_frame:
            from scipy.spatial.transform import Rotation
            tf = context.sensor_data_center.get_pose_at(frame, model_frame)
            if tf is None:
                raise ValueError(f"missing transform: {frame} -> {model_frame}")
            position, orientation = tf["position"], tf["orientation"]
            transform[:3, :3] = Rotation.from_quat([orientation[k] for k in ("x", "y", "z", "w")]).as_matrix()
            transform[:3, 3] = [position[k] for k in ("x", "y", "z")]
            grasp = transform @ grasp
            points = points @ transform[:3, :3].T + transform[:3, 3]
            if surround.size:
                surround = surround @ transform[:3, :3].T + transform[:3, 3]
            target_mesh.transform(transform)
        mesh = Mesh()
        mesh.vertices = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in np.asarray(target_mesh.vertices)]
        mesh.triangles = [MeshTriangle(vertex_indices=[int(v) for v in t]) for t in np.asarray(target_mesh.triangles)]
        obj = CollisionObject(id=f"orchid_pickup_{target}")
        obj.header.frame_id = model_frame
        obj.meshes = [mesh]
        obj.mesh_poses = [Pose()]
        obj.mesh_poses[0].orientation.w = 1.
        obj.operation = CollisionObject.ADD
        arm = context.hardware.get("arm", {})
        named = arm.get("named_targets", {}).get("go", {})
        go = named.get("joints")
        if not isinstance(go, dict) or set(go) != set(ARM_JOINTS):
            raise ValueError("HSR pickup requires a complete arm named target 'go'")
        log.info("Pickup inputs: frame=%s model=%s target=%s inflation=%.3f converted_frame=%s collected_s=%.3f",
                 model_frame, root.attrib["name"], target, inflation, corrected_rotation, time.monotonic()-began)
        return PickupInputs(model_frame, start, grasp, points, surround,
                            params.values[0].string_value,
                            add_pickup_groups(params.values[1].string_value, root.attrib["name"]),
                            limits, scene, obj, {k: float(v) for k, v in go.items()})

    def plan_post_lift(self, context, previous, cancelled=lambda: False):
        """Replan only retreat/undock using measured post-lift joints and live ACM/attachment."""
        from moveit_msgs.srv import GetPlanningScene, GetPositionFK
        from scipy.spatial.transform import Rotation
        from src.plugin.actions.manipulation import _latest_joint_positions
        request = GetPlanningScene.Request()
        request.components.components = 1023
        scene = self._call("/get_planning_scene", GetPlanningScene, request).scene
        measured = dict(zip(scene.robot_state.joint_state.name, scene.robot_state.joint_state.position))
        measured.update(_latest_joint_positions(context))
        if any(n not in measured for n in PICKUP_JOINTS):
            raise ValueError("post-lift measured state missing pickup joints")
        scene.robot_state.joint_state.name = list(measured)
        scene.robot_state.joint_state.position = list(measured.values())
        scene.robot_state.joint_state.velocity = []
        scene.robot_state.joint_state.effort = []
        fk = GetPositionFK.Request()
        fk.header.frame_id = previous.inputs.frame_id
        fk.fk_link_names = ["hand_palm_link"]
        fk.robot_state = scene.robot_state
        response = self._call("/compute_fk", GetPositionFK, fk)
        if response.error_code.val != 1 or not response.pose_stamped:
            raise MotionFailure("post-lift measured FK failed", FailureInfo(FailureCode.INVALID_STATE))
        pose = response.pose_stamped[0].pose
        insertion = next(m for m in previous.motions if m.name == "insertion")
        if not insertion.hand_path:
            raise ValueError("pickup plan missing insertion hand endpoints")
        retreat = np.eye(4)
        preview = next(m for m in previous.motions if m.name == "retreat")
        fk.robot_state = copy.deepcopy(scene.robot_state)
        endpoint = dict(measured)
        endpoint.update(preview.end_joints)
        fk.robot_state.joint_state.name = list(endpoint)
        fk.robot_state.joint_state.position = list(endpoint.values())
        response = self._call("/compute_fk", GetPositionFK, fk)
        if response.error_code.val != 1 or not response.pose_stamped:
            raise MotionFailure("original retreat endpoint FK failed", FailureInfo(FailureCode.INVALID_STATE))
        q = response.pose_stamped[0].pose.orientation
        retreat[:3, :3] = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        retreat[:3, 3] = insertion.hand_path[0]
        retreat[2, 3] += pose.position.z - insertion.hand_path[-1][2]
        inputs = replace(previous.inputs, start_joints=measured, scene=scene)
        undock = next(m for m in previous.motions if m.name == "undock")
        request = PickupRequest(target=previous.target, approach_mode=previous.selected_mode)
        log.info("Post-lift planning measured_base=%s measured_lift=%.6f measured_hand_z=%.6f retreat_xyz=%s mode=%s",
                 {n:measured[n] for n in BASE_JOINTS},measured["arm_lift_joint"],pose.position.z,retreat[:3,3].tolist(),previous.selected_mode)
        return self.plan(inputs, request, cancelled, post_lift=(retreat, undock.end_joints))

    def plan(self, inputs: PickupInputs, request: PickupRequest, cancelled=lambda: False, *, post_lift=None):
        from rclpy.action import ActionClient
        from rclpy.parameter import Parameter
        from orchid_hsr_pickup_interfaces.action import PlanPickup
        from orchid_hsr_pickup_interfaces.msg import PlanningSettings
        from geometry_msgs.msg import PoseStamped, Vector3
        from scipy.spatial.transform import Rotation
        if cancelled():
            raise MotionFailure("pickup planning cancelled", FailureInfo(FailureCode.CANCELLED))
        geometry = pickup_geometry(request, inputs.grasp, inputs.target_points, inputs.frame_id)
        goal = PlanPickup.Goal()
        goal.robot_description, goal.robot_description_semantic = inputs.robot_description, inputs.robot_description_semantic
        goal.scene, goal.target = inputs.scene, inputs.target
        goal.joint_limits = [Parameter(name, value=value).to_parameter_msg() for name, value in inputs.joint_limits.items()]
        goal.grasp = PoseStamped()
        goal.grasp.header.frame_id = inputs.frame_id
        goal.grasp.pose.position.x, goal.grasp.pose.position.y, goal.grasp.pose.position.z = map(float, inputs.grasp[:3, 3])
        q = Rotation.from_matrix(inputs.grasp[:3, :3]).as_quat()
        goal.grasp.pose.orientation.x, goal.grasp.pose.orientation.y, goal.grasp.pose.orientation.z, goal.grasp.pose.orientation.w = map(float, q)
        goal.direction = Vector3(x=geometry.direction[0], y=geometry.direction[1], z=geometry.direction[2])
        goal.backoff_distances = list(geometry.distances)
        goal.approach_mode, goal.max_candidates = request.approach_mode, request.max_candidates
        goal.open_width_m, goal.post_grasp_lift_m = request.open_width_m, request.post_grasp_lift_m
        goal.go_joints.name, goal.go_joints.position = list(inputs.go_joints), list(inputs.go_joints.values())
        goal.settings = PlanningSettings(**{f.name: (list(getattr(self.settings, f.name)) if f.name == "touch_links" else getattr(self.settings, f.name))
                                           for f in fields(self.settings) if f.name not in {"action_name", "timeout_s"}})
        goal.timeout_s = self.settings.timeout_s
        if post_lift is not None:
            from geometry_msgs.msg import Pose
            retreat, undock = post_lift
            goal.operation = "post_lift"
            goal.retreat_pose = Pose()
            goal.retreat_pose.position.x, goal.retreat_pose.position.y, goal.retreat_pose.position.z = map(float,retreat[:3,3])
            q = Rotation.from_matrix(retreat[:3,:3]).as_quat()
            goal.retreat_pose.orientation.x, goal.retreat_pose.orientation.y, goal.retreat_pose.orientation.z, goal.retreat_pose.orientation.w = map(float,q)
            goal.undock_joints.name = list(BASE_JOINTS)
            goal.undock_joints.position = [undock[n] for n in BASE_JOINTS]
        log.info("Pickup planning: requested_mode=%s distances=%s settings=%s", request.approach_mode, geometry.distances, self.settings)
        if self._action_client is None:
            self._action_client = ActionClient(self.node, PlanPickup, self.settings.action_name)
        client = self._action_client
        if not client.wait_for_server(timeout_sec=5.):
            raise MotionFailure(f"pickup planning action unavailable: {self.settings.action_name}", FailureInfo(FailureCode.NOT_FOUND))
        if cancelled():
            raise MotionFailure("pickup planning cancelled", FailureInfo(FailureCode.CANCELLED))
        began = time.monotonic()
        sent = client.send_goal_async(goal, feedback_callback=lambda msg: self.feedback(msg.feedback) if not cancelled() else None)
        handle = None
        def cancel_late(future):
            accepted = future.result()
            if accepted is not None and accepted.accepted:
                accepted.cancel_goal_async()
        try:
            self._wait(sent, cancelled, began)
            handle = sent.result()
            if not handle.accepted:
                raise MotionFailure("native pickup planning request rejected (server busy or invalid inputs)", FailureInfo(FailureCode.PLANNING_FAILED))
            result_future = handle.get_result_async()
            self._wait(result_future, cancelled, began)
            response = result_future.result()
            if cancelled():
                raise MotionFailure("pickup planning cancelled", FailureInfo(FailureCode.CANCELLED))
            result = response.result
        except Exception:
            if handle is not None:
                handle.cancel_goal_async()
            else:
                sent.add_done_callback(cancel_late)
            raise
        if response.status == 5:
            raise MotionFailure("pickup planning cancelled", FailureInfo(FailureCode.CANCELLED))
        accepted_success = response.status == 4 and result.success
        motions = tuple(PreparedMotion(m.name, m.trajectory, dict(zip(m.start_joints.name, m.start_joints.position)),
                      dict(zip(m.end_joints.name, m.end_joints.position)), m.attached,
                      tuple((p.x,p.y,p.z) for p in m.hand_path), tuple((p.x,p.y,p.z) for p in m.base_path)) for m in result.motions) if accepted_success else ()
        candidates = tuple(candidate_dict(c) for c in result.candidates)
        plan = PickupPlan(request.target, request.approach_mode, result.selected_mode, result.staging_order,
                          inputs, motions, candidates, {"reason":result.reason, "planning_s":result.planning_s,
                          "timings":dict(zip(result.timing_names,result.timing_seconds)),"service_calls":dict(self._service_counts),
                          "counts":{"ik_calls":result.ik_calls,"planning_calls":result.planning_calls,"validity_calls":result.validity_calls},
                          "robot_model":ET.fromstring(inputs.robot_description).attrib["name"]})
        if accepted_success:
            log.info("Pickup selected mode=%s staging=%s planning_s=%.3f motion_durations=%s", result.selected_mode,
                     result.staging_order, result.planning_s, {m.name: duration_seconds(m.trajectory.points[-1].time_from_start) for m in motions})
        else:
            log.warn("Pickup planning FAILED: %s; requested=%s; rejections=%s", result.reason, request.approach_mode,
                     [(c["id"],c["failure"]["phase"],c["failure"]["reason"],c["failure"]["contacts"]) for c in candidates])
        return plan

    def _wait(self, future, cancelled, began):
        last = began
        while not future.done():
            if cancelled():
                raise MotionFailure("pickup planning cancelled", FailureInfo(FailureCode.CANCELLED))
            now = time.monotonic()
            if now-began > self.settings.timeout_s+5.:
                raise MotionFailure("pickup planning response timeout", FailureInfo(FailureCode.PLANNING_FAILED, backend_code="timeout"))
            if now-last >= 2.:
                log.info("Pickup planning still running elapsed_s=%.1f", now-began)
                last = now
            time.sleep(.02)

    @staticmethod
    def feedback(f):
        message = "Pickup %s mode=%s candidate=%s operation=%s reason=%s fraction=%.3f contacts=%s elapsed_s=%.3f"
        logger = log.warn if (f.reason or f.contacts) and f.event != "yaw_constraints" else log.info
        logger(message, f.event, f.mode, f.candidate_id, f.operation, f.reason, f.fraction,
               [(c.first,c.second,round(c.depth,6)) for c in f.contacts], f.elapsed_s)

    def publish(self, plan: PickupPlan):
        """Publish retained markers, robot endpoints and planned trajectories only."""
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
        from moveit_msgs.msg import DisplayTrajectory, DisplayRobotState, RobotTrajectory
        from rclpy.qos import QoSProfile, DurabilityPolicy

        def publisher(topic, kind):
            if topic not in self._publishers:
                self._publishers[topic] = self.node.create_publisher(kind, topic, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            return self._publishers[topic]

        prefix = f"/orchid/hsr_pickup/{plan.selected_mode}"
        for mode in ("auto", "base_only", "arm_only", "arm_yaw", "coordinated"):
            publisher(f"/orchid/hsr_pickup/{mode}/markers", MarkerArray).publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        hidden = DisplayRobotState(hide=True)
        hidden.state.joint_state.name = list(plan.inputs.start_joints)
        hidden.state.joint_state.position = list(plan.inputs.start_joints.values())
        for topic, previous in list(self._publishers.items()):
            if "/candidate_" in topic or topic.endswith(("/selected_state", "/failure_state")):
                previous.publish(hidden)
        for index in range(24):
            publisher(prefix + f"/candidate_{index}", DisplayRobotState).publish(hidden)
        publisher(prefix + "/selected_state", DisplayRobotState).publish(hidden)
        publisher(prefix + "/failure_state", DisplayRobotState).publish(hidden)
        markers = MarkerArray()
        clear = Marker(action=Marker.DELETEALL)
        markers.markers.append(clear)
        frame = plan.inputs.frame_id
        def pose_axes(pose, label, selected, offset):
            pose = np.asarray(pose)
            for axis in range(3):
                marker = Marker(id=offset + axis, ns=label, type=Marker.ARROW, action=Marker.ADD)
                marker.header.frame_id = frame
                marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (pose[:3, 3], pose[:3, 3] + .08 * pose[:3, axis])]
                marker.scale.x, marker.scale.y, marker.scale.z = .006, .012, .015
                marker.color.a = 1. if selected else .45
                setattr(marker.color, "rgb"[axis], 1.)
                markers.markers.append(marker)
            text = Marker(id=offset + 3, ns=label, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            text.header.frame_id = frame
            text.pose.position = Point(x=float(pose[0, 3]), y=float(pose[1, 3]), z=float(pose[2, 3] + .06))
            text.pose.orientation.w = 1.
            text.scale.z = .025
            text.color.r = text.color.g = text.color.b = text.color.a = 1.
            text.text = label
            markers.markers.append(text)
        pose_axes(plan.inputs.grasp, "grasp", True, 0)
        failure_shown = False
        for index, candidate in enumerate(plan.candidates):
            label = f"{candidate['id']} {candidate['status']} {candidate['distance_m']:.3f}m {candidate.get('reason', candidate.get('failure', {}).get('reason', ''))}"
            pose_axes(candidate["pregrasp"], label, candidate["status"] == "fully_planned", 10 + index * 10)
            failure = candidate.get("failure", {})
            rejected = failure.get("rejected_state", failure)
            if rejected.get("joints") and "hand_pose" in rejected:
                pose_axes(rejected["hand_pose"], candidate["id"] + " failure: " + failure.get("reason", ""), True, 850 + index * 10)
            if not failure_shown and rejected.get("joints"):
                robot = DisplayRobotState()
                robot.state.joint_state.name = list(rejected["joints"])
                robot.state.joint_state.position = list(rejected["joints"].values())
                publisher(prefix + "/failure_state", DisplayRobotState).publish(robot)
                failure_shown = True
            for contact_index, contact in enumerate(rejected.get("contact_details", [])):
                marker = Marker(ns="collision_contact", id=2000+index*100+contact_index, type=Marker.SPHERE, action=Marker.ADD)
                marker.header.frame_id = frame
                marker.pose.orientation.w = 1.
                marker.pose.position = Point(x=contact["position"][0],y=contact["position"][1],z=contact["position"][2])
                marker.scale.x = marker.scale.y = marker.scale.z = .015
                marker.color.r = marker.color.a = 1.
                markers.markers.append(marker)
            if "retreat" in candidate:
                pose_axes(candidate["retreat"], candidate["id"] + " retreat", True, 500 + index * 10)
            if candidate.get("joints"):
                q = candidate["joints"]
                base_pose = np.eye(4)
                yaw = q["odom_t"]
                base_pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
                base_pose[:3, 3] = [q["odom_x"], q["odom_y"], .025]
                pose_axes(base_pose, candidate["id"] + " base", candidate["status"] == "fully_planned", 700 + index * 10)
                robot = DisplayRobotState()
                robot.state.joint_state.name = list(candidate["joints"])
                robot.state.joint_state.position = list(candidate["joints"].values())
                publisher(prefix + f"/candidate_{index}", DisplayRobotState).publish(robot)
                if candidate["status"] == "fully_planned":
                    publisher(prefix + "/selected_state", DisplayRobotState).publish(robot)
        captured = plan.inputs.scene
        for phase in plan.motions:
            paths = {"hand_path": phase.hand_path, "base_path": phase.base_path}
            for index, key in enumerate(("hand_path", "base_path")):
                line = Marker(ns=phase.name + key, id=1000 + len(markers.markers), type=Marker.LINE_STRIP, action=Marker.ADD)
                line.header.frame_id = frame
                line.scale.x = .005
                line.color.a = 1.
                line.color.g = 1. if index == 0 else .4
                line.color.b = .4 if index == 0 else 1.
                line.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in paths.get(key) or []]
                markers.markers.append(line)
            message = DisplayTrajectory()
            message.model_id = plan.diagnostics["robot_model"]
            message.trajectory_start = copy.deepcopy(captured.robot_state)
            message.trajectory_start.joint_state.name = list(phase.start_joints)
            message.trajectory_start.joint_state.position = list(phase.start_joints.values())
            rt = RobotTrajectory()
            rt.joint_trajectory = phase.trajectory
            message.trajectory = [rt]
            publisher(prefix + "/" + phase.name, DisplayTrajectory).publish(message)
        publisher(prefix + "/markers", MarkerArray).publish(markers)
        staging = [m for m in plan.motions if m.name.startswith("stag")]
        if staging:
            message = DisplayTrajectory(model_id=plan.diagnostics["robot_model"], trajectory_start=captured.robot_state)
            message.trajectory = [RobotTrajectory(joint_trajectory=m.trajectory) for m in staging]
            publisher(prefix + "/staging", DisplayTrajectory).publish(message)
