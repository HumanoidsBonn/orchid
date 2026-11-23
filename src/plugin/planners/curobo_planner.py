"""Optional in-process cuRobo manipulation planner backend."""

from __future__ import annotations

import importlib
import json
import math
import os
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.planners import BaseManipulationPlanner, PlannerResult
from src.core.geometry import planar_pose_matrix, point_relative_to_planar_base, rotation_matrix_to_quaternion
from src.core.serialization import to_json_safe
import src.core.logger as log


class CuroboManipulationPlanner(BaseManipulationPlanner):
    """cuRobo-backed manipulation planner.

    This backend imports cuRobo lazily so Orchid can run before cuRobo is
    installed. The first supported surface is IK feasibility; trajectory
    planning is wired through the same interface and can be filled once the
    robot-specific world/robot config is reviewed.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._curobo_module = None
        self._available: bool | None = None
        self._import_error: str | None = None
        self._solver = None
        self._tensor_args = None
        self._robot_config_data: dict[str, Any] | None = None
        self._config_error: str | None = None
        self._last_pose_metadata: dict[str, Any] = {}
        self._last_target_error: str | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_import()
        return bool(self._available)

    def check_ik(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        if not self.available:
            return PlannerResult(
                success=False,
                reason="curobo_not_available",
                metadata={"import_error": self._import_error},
            )
        solver = self._get_ik_solver()
        if solver is None:
            return PlannerResult(
                success=False,
                reason="curobo_ik_not_configured",
                metadata={
                    "planner": "curobo",
                    "target": self._summarize_target(target),
                    "config_error": self._config_error,
                    "config_keys": sorted(self.config.keys()),
                },
            )
        self._last_pose_metadata = {}
        self._last_target_error = None
        pose_list = self._target_to_pose_list(target, context=context)
        if pose_list is None:
            return PlannerResult(
                success=False,
                reason=self._last_target_error or "curobo_ik_target_unavailable",
                metadata={
                    "planner": "curobo",
                    "target": self._summarize_target(target),
                    "expected": "target must provide ee_pose or object_position/base_pose",
                    "pose_metadata": to_json_safe(self._last_pose_metadata),
                },
            )
        validation_error = self._validate_pose_list(pose_list)
        if validation_error is not None:
            return PlannerResult(
                success=False,
                reason=validation_error,
                metadata={
                    "planner": "curobo",
                    "target": self._summarize_target(target),
                    "pose_wxyz": pose_list,
                    "pose_metadata": to_json_safe(self._last_pose_metadata),
                },
            )
        seed_state = self._seed_state_metadata(context, target=target, options=options)
        try:
            pose_cls = importlib.import_module("curobo._src.types.pose").Pose
            goal_pose_cls = importlib.import_module("curobo._src.types.tool_pose").GoalToolPose
            pose = pose_cls.from_list(pose_list, device_cfg=self._tensor_args)
            tool_frames = list(getattr(getattr(solver, "kinematics", None), "tool_frames", []) or [self._ee_link()])
            goal_tool_poses = goal_pose_cls.from_poses(
                {tool_frames[0]: pose.unsqueeze(1)},
                ordered_tool_frames=tool_frames,
            )
            current_state = self._current_state_from_seed(solver, seed_state)
            solve_kwargs = {"goal_tool_poses": goal_tool_poses, "return_seeds": 1}
            if current_state is not None:
                solve_kwargs["current_state"] = current_state
            result = solver.solve_pose(**solve_kwargs)
            success = self._as_bool(getattr(result, "success", False))
            joints = self._extract_joint_positions(result)
            score = self._extract_score(result)
            reason = None if success else "curobo_ik_failed"
            solver_errors = self._solver_error_metadata(result)
            compact_metadata = {
                "planner": "curobo",
                "pose_wxyz": pose_list,
                "pose_metadata": to_json_safe(self._last_pose_metadata),
                "config_summary": self.config_summary(),
                "seed_state": seed_state,
                "solver_errors": solver_errors,
                "raw_success": self._jsonish(getattr(result, "success", None)),
            }
            debug_json = self._write_debug_json(
                {
                    **compact_metadata,
                    "success": success,
                    "reason": reason,
                    "target": self._summarize_target(target),
                    "joint_positions": joints,
                    "score": score,
                    "raw_result": self._jsonish(result),
                }
            )
            if debug_json:
                compact_metadata["debug_json"] = debug_json
                log.info(
                    "cuRobo IK %s for target=%s debug_json=%s",
                    "succeeded" if success else "failed",
                    self._target_name(target),
                    debug_json,
                )
            return PlannerResult(
                success=success,
                reason=reason,
                joint_positions=joints,
                score=score,
                metadata=compact_metadata,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("cuRobo IK solve failed: %s", exc)
            return PlannerResult(
                success=False,
                reason="curobo_ik_exception",
                metadata={
                    "planner": "curobo",
                    "error": str(exc),
                    "target": self._summarize_target(target),
                    "pose_metadata": to_json_safe(self._last_pose_metadata),
                },
            )

    def plan_pose(self, target: Any, *, context: Any | None = None, **options: Any) -> PlannerResult:
        if not self.available:
            return PlannerResult(
                success=False,
                reason="curobo_not_available",
                metadata={"import_error": self._import_error},
            )
        return PlannerResult(
            success=False,
            reason="curobo_pose_planning_not_configured",
            metadata={"planner": "curobo", "target": self._summarize_target(target)},
        )

    def _try_import(self) -> bool:
        try:
            self._curobo_module = importlib.import_module("curobo")
            importlib.import_module("curobo._src.types.device_cfg")
            importlib.import_module("curobo._src.types.pose")
            importlib.import_module("curobo._src.types.tool_pose")
            importlib.import_module("curobo._src.solver.solver_ik")
            importlib.import_module("curobo._src.solver.solver_ik_cfg")
            importlib.import_module("curobo.config_io")
            log.info("cuRobo module imported successfully")
            return True
        except Exception as exc:  # noqa: BLE001
            self._import_error = str(exc)
            log.warn("cuRobo unavailable: %s", exc)
            return False

    def _get_ik_solver(self):
        if self._solver is not None:
            return self._solver
        try:
            self._load_robot_config()
            device_cfg_cls = importlib.import_module("curobo._src.types.device_cfg").DeviceCfg
            ik_solver_module = importlib.import_module("curobo._src.solver.solver_ik")
            ik_cfg_module = importlib.import_module("curobo._src.solver.solver_ik_cfg")
            try:
                util_config = importlib.import_module("curobo._src.util.config_io")
            except Exception:
                util_config = importlib.import_module("curobo.config_io")
            self._tensor_args = device_cfg_cls()
            metrics_rollout = "metrics_base.yml"
            if self._metrics_self_collision_disabled():
                content = importlib.import_module("curobo.content")
                metrics_rollout = self._metrics_without_self_collision(
                    util_config,
                    content,
                )
            ik_config = ik_cfg_module.IKSolverCfg.create(
                robot=self._expanded_robot_config_path(),
                device_cfg=self._tensor_args,
                metrics_rollout=metrics_rollout,
                **self._ik_options(),
            )
            self._solver = ik_solver_module.IKSolver(ik_config)
            log.info("cuRobo IK solver initialized from %s", self.config.get("robot_config"))
            return self._solver
        except Exception as exc:  # noqa: BLE001
            self._config_error = str(exc)
            log.warn("cuRobo IK solver not configured: %s", exc)
            return None

    def _load_robot_config(self) -> dict[str, Any]:
        if self._robot_config_data is not None:
            return self._robot_config_data
        path = self.config.get("robot_config")
        if not path:
            raise ValueError("missing robot_config path")
        expanded = os.path.expandvars(os.path.expanduser(str(path)))
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"robot_config not found: {expanded}")
        util_file = importlib.import_module("curobo.config_io")
        data = util_file.load_yaml(expanded)
        if not isinstance(data, dict):
            raise ValueError(f"robot_config must load to a dict: {expanded}")
        self._robot_config_data = data
        return data

    def _expanded_robot_config_path(self) -> str:
        path = self.config.get("robot_config")
        if not path:
            raise ValueError("missing robot_config path")
        return os.path.expandvars(os.path.expanduser(str(path)))

    def _ik_options(self) -> dict[str, Any]:
        config_data = self._robot_config_data or {}
        yaml_options = config_data.get("ik", {}) if isinstance(config_data.get("ik"), dict) else {}
        inline_options = self.config.get("ik", {}) if isinstance(self.config.get("ik"), dict) else {}
        merged = {**yaml_options, **inline_options}
        merged.setdefault("random_seed", 0)
        merged.setdefault("use_cuda_graph", False)
        allowed = {
            "position_tolerance",
            "orientation_tolerance",
            "num_seeds",
            "self_collision_check",
            "use_cuda_graph",
            "random_seed",
            "optimizer_collision_activation_distance",
            "store_debug",
            "success_requires_convergence",
            "seed_position_weight",
            "seed_orientation_weight",
            "seed_velocity_weight",
            "seed_acceleration_weight",
            "seed_solver_num_seeds",
            "max_batch_size",
            "max_goalset",
        }
        if "position_threshold" in merged and "position_tolerance" not in merged:
            merged["position_tolerance"] = merged["position_threshold"]
        if "rotation_threshold" in merged and "orientation_tolerance" not in merged:
            merged["orientation_tolerance"] = merged["rotation_threshold"]
        return {key: value for key, value in merged.items() if key in allowed}

    def _write_debug_json(self, payload: dict[str, Any]) -> str | None:
        if self.config.get("debug_json_enabled", True) is False:
            return None
        directory = Path(
            os.path.expandvars(
                os.path.expanduser(
                    str(self.config.get("debug_dir", "/tmp/orchid_debug/curobo"))
                )
            )
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            suffix = int((time.time() % 1.0) * 1_000_000)
            path = directory / f"ik_{stamp}_{suffix:06d}.json"
            path.write_text(json.dumps(to_json_safe(payload), indent=2, default=str))
            return str(path)
        except Exception as exc:  # noqa: BLE001
            log.warn("Failed to write cuRobo IK debug JSON: %s", exc)
            return None

    @staticmethod
    def _target_name(target: Any) -> str:
        if isinstance(target, dict):
            return str(target.get("target") or target.get("object_id") or "unknown")
        return "unknown"

    def _target_to_pose_list(self, target: Any, *, context: Any | None = None) -> list[float] | None:
        data = target if isinstance(target, dict) else {}
        ee_pose = data.get("ee_pose") or data.get("pose")
        if ee_pose is not None:
            pose = self._pose_to_list(ee_pose)
            if pose is not None:
                return pose
        target_pose_matrix = data.get("target_pose_matrix")
        base_pose = data.get("base_pose")
        if target_pose_matrix is not None and base_pose is not None:
            pose = self._target_matrix_to_base_pose(target_pose_matrix, base_pose, data=data, context=context)
            if pose is not None:
                return pose
        object_position = data.get("object_position")
        if object_position is None and data.get("object_xy") is not None:
            object_position = [data["object_xy"][0], data["object_xy"][1], self._target_z_offset()]
        if object_position is None or base_pose is None:
            return None
        try:
            ox, oy, oz = self._xyz(object_position)
            x_base, y_base, z_base = point_relative_to_planar_base((ox, oy, oz), base_pose)
            z_base = oz + self._target_z_offset()
            pose = [x_base, y_base, z_base, *self._default_orientation_wxyz()]
            self._last_pose_metadata = {
                "conversion": "object_position_candidate_base",
                "curobo_base_frame": self.robot_base_frame(),
                "quaternion_order": "wxyz",
            }
            return pose
        except Exception:  # noqa: BLE001
            return None

    def _target_matrix_to_base_pose(
        self,
        target_pose_matrix: Any,
        base_pose: Any,
        *,
        data: dict[str, Any] | None = None,
        context: Any | None = None,
    ) -> list[float] | None:
        try:
            data = data or {}
            target = np.asarray(target_pose_matrix, dtype=np.float64)
            if target.shape != (4, 4):
                self._last_target_error = "invalid_target_pose_matrix"
                return None
            bx = float(base_pose["x"])
            by = float(base_pose["y"])
            byaw = float(base_pose.get("yaw", 0.0))
            candidate_frame = str(base_pose.get("frame_id") or data.get("candidate_frame") or data.get("frame_id") or "")
            target_frame = str(data.get("target_pose_frame") or data.get("object_pose_frame") or candidate_frame)
            nav_base_frame = str(data.get("nav_base_frame") or base_pose.get("base_frame") or "")
            curobo_base_frame = self.robot_base_frame()
            if not candidate_frame:
                candidate_frame = target_frame or "map"
            if not target_frame:
                target_frame = candidate_frame
            if not nav_base_frame:
                nav_base_frame = curobo_base_frame

            target_in_candidate = target
            if target_frame != candidate_frame:
                tf_mat = self._lookup_tf_matrix(context, target_frame, candidate_frame)
                if tf_mat is None:
                    self._last_target_error = "missing_object_to_candidate_frame_tf"
                    self._last_pose_metadata = {
                        "target_pose_frame": target_frame,
                        "candidate_frame": candidate_frame,
                        "nav_base_frame": nav_base_frame,
                        "curobo_base_frame": curobo_base_frame,
                    }
                    return None
                target_in_candidate = tf_mat @ target

            candidate_to_nav_base = planar_pose_matrix(bx, by, byaw)
            nav_base_to_curobo_base = np.eye(4, dtype=np.float64)
            if nav_base_frame != curobo_base_frame:
                tf_mat = self._lookup_tf_matrix(context, curobo_base_frame, nav_base_frame)
                if tf_mat is None:
                    self._last_target_error = "missing_nav_base_to_curobo_base_tf"
                    self._last_pose_metadata = {
                        "target_pose_frame": target_frame,
                        "candidate_frame": candidate_frame,
                        "nav_base_frame": nav_base_frame,
                        "curobo_base_frame": curobo_base_frame,
                    }
                    return None
                nav_base_to_curobo_base = tf_mat

            candidate_to_curobo_base = candidate_to_nav_base @ nav_base_to_curobo_base
            pose_in_base = np.linalg.inv(candidate_to_curobo_base) @ target_in_candidate
            qx, qy, qz, qw = rotation_matrix_to_quaternion(pose_in_base[:3, :3])
            qnorm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if qnorm > 1e-9:
                qw, qx, qy, qz = qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm
            pose = [
                float(pose_in_base[0, 3]),
                float(pose_in_base[1, 3]),
                float(pose_in_base[2, 3]),
                float(qw),
                float(qx),
                float(qy),
                float(qz),
            ]
            self._last_pose_metadata = {
                "conversion": "target_matrix_candidate_to_curobo_base",
                "target_pose_frame": target_frame,
                "candidate_frame": candidate_frame,
                "nav_base_frame": nav_base_frame,
                "curobo_base_frame": curobo_base_frame,
                "quaternion_order": "wxyz",
                "candidate_pose": {"x": bx, "y": by, "yaw": byaw},
                "pose_wxyz": pose,
            }
            return pose
        except Exception:  # noqa: BLE001
            self._last_target_error = "target_pose_transform_failed"
            return None

    def robot_base_frame(self) -> str:
        data = self._robot_config_data or self._load_robot_config()
        kinematics = data.get("robot_cfg", {}).get("kinematics", {})
        robot_base = str(kinematics.get("base_link") or "")
        orchid_base = str(self._orchid_config().get("base_frame") or "")
        if robot_base and orchid_base and robot_base != orchid_base:
            log.warn(
                "cuRobo robot base frame %s differs from orchid.base_frame %s; using robot_cfg.kinematics.base_link",
                robot_base,
                orchid_base,
            )
        return robot_base or orchid_base or "base_link"

    def config_summary(self) -> dict[str, Any]:
        data = self._robot_config_data or self._load_robot_config()
        kinematics = data.get("robot_cfg", {}).get("kinematics", {})
        cspace = kinematics.get("cspace", {}) if isinstance(kinematics.get("cspace"), dict) else {}
        return {
            "robot_config": self._expanded_robot_config_path(),
            "curobo_base_frame": self.robot_base_frame(),
            "orchid_base_frame": self._orchid_config().get("base_frame"),
            "tool_frames": kinematics.get("tool_frames"),
            "cspace_joints": cspace.get("joint_names"),
            "collision_links": kinematics.get("collision_link_names"),
            "ik_options": self._ik_options(),
            "robot_config_profile": self._orchid_config().get("robot_config_profile"),
            "metrics_self_collision_enabled": not self._metrics_self_collision_disabled(),
        }

    def _metrics_self_collision_disabled(self) -> bool:
        if self.config.get("metrics_self_collision_enabled") is False:
            return True
        if self.config.get("debug_collision_disabled_metrics") is True:
            return True
        if self._ik_options().get("self_collision_check") is False and self.config.get(
            "disable_metrics_self_collision_with_ik", False
        ):
            return True
        return False

    @staticmethod
    def _metrics_without_self_collision(util_config: Any, content: Any) -> dict[str, Any]:
        metrics = copy.deepcopy(
            util_config.resolve_config(
                util_config.join_path(content.get_task_configs_path(), "metrics_base.yml")
            )
        )
        constraint_cfg = (
            metrics.get("rollout", {}).get("constraint_cfg", {})
            if isinstance(metrics, dict)
            else {}
        )
        if isinstance(constraint_cfg, dict):
            constraint_cfg.pop("self_collision_cfg", None)
        return metrics

    @staticmethod
    def _lookup_tf_matrix(context: Any | None, source_frame: str, target_frame: str) -> np.ndarray | None:
        sdc = getattr(context, "sensor_data_center", None)
        if sdc is None:
            return None
        tf = sdc.get_pose_at(source_frame, target_frame)
        if tf is None:
            return None
        mat = tf.get("mat")
        if mat is None:
            return None
        array = np.asarray(mat, dtype=np.float64)
        return array if array.shape == (4, 4) else None

    @staticmethod
    def _validate_pose_list(pose: list[float]) -> str | None:
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            return "invalid_curobo_pose"
        qnorm = float(np.linalg.norm(values[3:7]))
        if qnorm < 0.95 or qnorm > 1.05:
            return "invalid_curobo_quaternion"
        if float(np.linalg.norm(values[:3])) > 3.0:
            return "target_out_of_workspace"
        return None

    def _seed_state_metadata(
        self,
        context: Any | None,
        *,
        target: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cspace = self.config_summary().get("cspace_joints") or []
        provided = self._provided_joint_seed(target=target, options=options)
        if provided is not None:
            return self._seed_metadata_from_values(provided, cspace, source="provided")
        sdc = getattr(context, "sensor_data_center", None)
        if sdc is None:
            return {"available": False, "reason": "missing_sensor_data_center"}
        msg = sdc.get_latest("joint_states", timeout_s=0.05)
        if msg is None or msg.data is None:
            return {"available": False, "reason": "missing_joint_states"}
        names = list(getattr(msg.data, "name", []) or [])
        positions = list(getattr(msg.data, "position", []) or [])
        values = {name: float(pos) for name, pos in zip(names, positions)}
        return self._seed_metadata_from_values(values, cspace, source="sensor_data_center")

    @staticmethod
    def _provided_joint_seed(*, target: Any | None, options: dict[str, Any] | None) -> dict[str, float] | None:
        for source in (options or {}, target if isinstance(target, dict) else {}):
            seed = source.get("joint_state_seed") or source.get("seed_state")
            if seed is None:
                continue
            if isinstance(seed, dict) and "joints" in seed and isinstance(seed["joints"], dict):
                return {str(name): float(value) for name, value in seed["joints"].items()}
            names = list(seed.get("name", []) or []) if isinstance(seed, dict) else []
            positions = list(seed.get("position", []) or []) if isinstance(seed, dict) else []
            if names and positions:
                return {str(name): float(pos) for name, pos in zip(names, positions, strict=False)}
            if isinstance(seed, dict):
                try:
                    return {str(name): float(value) for name, value in seed.items()}
                except Exception:  # noqa: BLE001
                    return None
        return None

    @staticmethod
    def _seed_metadata_from_values(values: dict[str, float], cspace: list[str], *, source: str) -> dict[str, Any]:
        missing = [joint for joint in cspace if joint not in values]
        if missing:
            return {"available": False, "reason": "incomplete_joint_states", "missing": missing, "source": source}
        return {"available": True, "source": source, "joints": {joint: values[joint] for joint in cspace}}

    def _current_state_from_seed(self, solver: Any, seed_state: dict[str, Any]) -> Any | None:
        if not seed_state.get("available") or not isinstance(seed_state.get("joints"), dict):
            return None
        try:
            torch = importlib.import_module("torch")
            joint_state_cls = importlib.import_module("curobo._src.state.state_joint").JointState
            joints = seed_state["joints"]
            cspace = list(self.config_summary().get("cspace_joints") or joints.keys())
            values = [float(joints[name]) for name in cspace]
            device = getattr(self._tensor_args, "device", "cuda")
            dtype = getattr(self._tensor_args, "dtype", torch.float32)
            tensor = torch.tensor([values], device=device, dtype=dtype)
            joint_state = joint_state_cls.from_position(tensor, joint_names=cspace)
            if hasattr(solver, "get_active_js"):
                joint_state = solver.get_active_js(joint_state)
            if hasattr(joint_state, "unsqueeze") and len(getattr(joint_state, "shape", [1])) == 1:
                joint_state = joint_state.unsqueeze(0)
            return joint_state
        except Exception as exc:  # noqa: BLE001
            log.debug("Unable to build cuRobo current_state from seed: %s", exc)
            return None

    def _solver_error_metadata(self, result: Any) -> dict[str, Any]:
        fields = {}
        for name in (
            "position_error",
            "rotation_error",
            "goalset_index",
            "solve_time",
            "ik_time",
            "debug_info",
        ):
            if hasattr(result, name):
                fields[name] = self._numeric_or_jsonish(getattr(result, name))
        return fields

    @staticmethod
    def _numeric_or_jsonish(value: Any) -> Any:
        try:
            if hasattr(value, "detach"):
                array = value.detach().cpu().reshape(-1)
                return float(array[0].item()) if array.numel() == 1 else array.tolist()
            if hasattr(value, "item"):
                return value.item()
        except Exception:  # noqa: BLE001
            pass
        return CuroboManipulationPlanner._jsonish(value)

    @staticmethod
    def _pose_to_list(value: Any) -> list[float] | None:
        if isinstance(value, dict):
            if "position" in value:
                position = value["position"]
            else:
                position = value
            try:
                x, y, z = CuroboManipulationPlanner._xyz(position)
                if "orientation_wxyz" in value:
                    qw, qx, qy, qz = [float(v) for v in value["orientation_wxyz"]]
                elif "quaternion" in value:
                    quat = value["quaternion"]
                    if isinstance(quat, dict):
                        qw = float(quat.get("w", 1.0))
                        qx = float(quat.get("x", 0.0))
                        qy = float(quat.get("y", 0.0))
                        qz = float(quat.get("z", 0.0))
                    else:
                        qx, qy, qz, qw = [float(v) for v in quat]
                else:
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                return [x, y, z, qw, qx, qy, qz]
            except Exception:  # noqa: BLE001
                return None
        try:
            values = [float(v) for v in value]
            return values if len(values) == 7 else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _xyz(value: Any) -> tuple[float, float, float]:
        if isinstance(value, dict):
            return float(value["x"]), float(value["y"]), float(value.get("z", 0.0))
        return float(value[0]), float(value[1]), float(value[2] if len(value) > 2 else 0.0)

    def _target_z_offset(self) -> float:
        return float(self._orchid_config().get("target_z_offset", self.config.get("target_z_offset", 0.05)))

    def _default_orientation_wxyz(self) -> list[float]:
        orientation = self._orchid_config().get("orientation_wxyz", self.config.get("orientation_wxyz", [1, 0, 0, 0]))
        values = [float(value) for value in orientation]
        return values if len(values) == 4 else [1.0, 0.0, 0.0, 0.0]

    def _ee_link(self) -> str:
        return str(self._orchid_config().get("ee_link", self.config.get("ee_link", "hand_palm_link")))

    def _orchid_config(self) -> dict[str, Any]:
        try:
            data = self._robot_config_data or self._load_robot_config()
        except Exception:
            data = {}
        orchid = data.get("orchid", {}) if isinstance(data.get("orchid"), dict) else {}
        return orchid

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if hasattr(value, "item"):
            return bool(value.item())
        if hasattr(value, "detach"):
            return bool(value.detach().cpu().item())
        if isinstance(value, (list, tuple)):
            return any(CuroboManipulationPlanner._as_bool(item) for item in value)
        return bool(value)

    @staticmethod
    def _extract_score(result: Any) -> float | None:
        for name in ("score", "position_error", "error"):
            value = getattr(result, name, None)
            if value is None:
                continue
            try:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().reshape(-1)[0].item()
                elif hasattr(value, "item"):
                    value = value.item()
                return float(value)
            except Exception:  # noqa: BLE001
                continue
        return None

    def _extract_joint_positions(self, result: Any) -> dict[str, float] | None:
        solution = getattr(result, "solution", None)
        if solution is None:
            js_solution = getattr(result, "js_solution", None)
            solution = getattr(js_solution, "position", None)
        joint_names = getattr(self._solver, "joint_names", None)
        if solution is None or joint_names is None:
            return None
        try:
            if hasattr(solution, "detach"):
                values = solution.detach().cpu().reshape(-1).tolist()
            elif hasattr(solution, "tolist"):
                values = np.asarray(solution.tolist()).reshape(-1).tolist()  # type: ignore[name-defined]
            else:
                values = list(solution)
            return {
                str(name): float(value)
                for name, value in zip(joint_names, values, strict=False)
            }
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _jsonish(value: Any) -> Any:
        return to_json_safe(value)

    @staticmethod
    def _summarize_target(target: Any) -> Any:
        if hasattr(target, "to_dict"):
            return target.to_dict()
        if hasattr(target, "__dict__"):
            return {
                key: value
                for key, value in vars(target).items()
                if not key.startswith("_")
            }
        return str(target)
