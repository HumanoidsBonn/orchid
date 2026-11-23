"""PluginRegistry: discovers and caches action and controller plugins."""

from __future__ import annotations

import src.core.logger as log


class PluginRegistry:
    """Registry for discovering actions and controllers via entry points."""

    def __init__(self):
        """Initialize empty registry."""
        self._actions: dict[str, type] = {}
        self._controllers: dict[str, type] = {}
        self._planners: dict[str, type] = {}
        self._conditions: dict[str, type] = {}

    def discover(self) -> None:
        """Discover plugins via entry points.
        
        Scans for:
        - mm.actions
        - mm.controllers
        - mm.planners
        - mm.conditions
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points

        for ep in entry_points(group="mm.actions"):
            self._actions.setdefault(ep.name, ep.load())

        for ep in entry_points(group="mm.controllers"):
            self._controllers.setdefault(ep.name, ep.load())

        for ep in entry_points(group="mm.planners"):
            self._planners.setdefault(ep.name, ep.load())

        for ep in entry_points(group="mm.conditions"):
            self._conditions.setdefault(ep.name, ep.load())

        self._register_builtins()
        log.info(
            f"PluginRegistry discovered: {len(self._actions)} actions, "
            f"{len(self._controllers)} controllers, {len(self._planners)} planners, "
            f"{len(self._conditions)} conditions"
        )

    def _register_builtins(self) -> None:
        """Register in-repo plugins when entry points are unavailable."""
        from src.plugin.actions.fetch import FetchObject
        from src.plugin.actions.hsr_pickup import PlanHsrPickup, HsrPickup
        from src.plugin.actions.trajectory import ExecuteJointTrajectory
        for action in (PlanHsrPickup, ExecuteJointTrajectory, HsrPickup, FetchObject):
            self._actions.setdefault(action.__name__, action)
        from src.plugin.conditions import WorldStateExists
        self._conditions.setdefault("world_state_exists", WorldStateExists)
        from src.plugin.actions.manipulation import (
            GraspObjectAction,
            MoveArmToState,
            PickTarget,
            PickAtPose,
            PlaceAtPose,
            ReleaseObjectAction,
        )
        from src.plugin.actions.navigation import MoveBaseWithWholeBody, MoveToManipulationPose, MoveToPose, MoveToTarget
        from src.plugin.actions.observe import ObserveScene, ScanScene, DescribeScene
        from src.plugin.actions.inspect import InspectScene
        from src.plugin.actions.estimate_pick_pose import EstimatePickPose
        from src.plugin.actions.estimate_pick_pose_ros2 import EstimatePickPoseRos2Service
        from src.plugin.actions.estimate_pick_pose_ggx import EstimatePickPoseGgx
        from src.plugin.actions.grasp_alignment import (
            ExecuteHsrCartesianGraspMotion,
            EstimateSimpleCylindricalGrasp,
            EstimateSimplePcaGrasp,
            MoveToSelectedGraspBasePose,
            OmniBaseServoToPose,
            PickTargetWithGraspAlignment,
            SelectHsrGraspMotion,
            SelectFkArmPosture,
            SolveTmcGraspIk,
        )
        from src.plugin.actions.user_input import AskUser
        from src.plugin.actions.head import MoveHeadToState, MoveHeadToPanTilt
        from src.plugin.controllers.moveit import MoveItArmController, MoveItGripperController, MoveItHeadController
        from src.plugin.controllers.nav2 import Nav2Controller
        # HSR specific controllers
        from src.plugin.controllers.hsrb_nav import HsrbNavController
        from src.plugin.controllers.hsrb_arm import HsrbArmController
        from src.plugin.controllers.hsrb_head import HsrbHeadController
        from src.plugin.controllers.hsrb_gripper import HsrbGripperController
        from src.plugin.controllers.ros2_control_trajectory import Ros2ControlController
        from src.plugin.planners.moveit_planner import MoveItManipulationPlanner
        from src.plugin.planners.curobo_planner import CuroboManipulationPlanner
        builtins_actions = {
            MoveToPose.__name__: MoveToPose,
            MoveBaseWithWholeBody.__name__: MoveBaseWithWholeBody,
            MoveToTarget.__name__: MoveToTarget,
            MoveToManipulationPose.__name__: MoveToManipulationPose,
            GraspObjectAction.__name__: GraspObjectAction,
            ReleaseObjectAction.__name__: ReleaseObjectAction,
            MoveArmToState.__name__: MoveArmToState,
            MoveHeadToPanTilt.__name__: MoveHeadToPanTilt,
            MoveHeadToState.__name__: MoveHeadToState, 
            PickTarget.__name__:PickTarget,
            PickAtPose.__name__:PickAtPose,
            PlaceAtPose.__name__: PlaceAtPose,
            ObserveScene.__name__: ObserveScene,
            ScanScene.__name__: ScanScene,
            InspectScene.__name__: InspectScene,
            DescribeScene.__name__: DescribeScene,
            EstimatePickPose.__name__:EstimatePickPose,
            EstimatePickPoseRos2Service.__name__: EstimatePickPoseRos2Service,
            EstimatePickPoseGgx.__name__: EstimatePickPoseGgx,
            EstimateSimpleCylindricalGrasp.__name__: EstimateSimpleCylindricalGrasp,
            EstimateSimplePcaGrasp.__name__: EstimateSimplePcaGrasp,
            SelectHsrGraspMotion.__name__: SelectHsrGraspMotion,
            SolveTmcGraspIk.__name__: SolveTmcGraspIk,
            SelectFkArmPosture.__name__: SelectFkArmPosture,
            MoveToSelectedGraspBasePose.__name__: MoveToSelectedGraspBasePose,
            OmniBaseServoToPose.__name__: OmniBaseServoToPose,
            ExecuteHsrCartesianGraspMotion.__name__: ExecuteHsrCartesianGraspMotion,
            PickTargetWithGraspAlignment.__name__: PickTargetWithGraspAlignment,
            AskUser.__name__: AskUser,
        }
        builtins_controllers = {
            "nav2": Nav2Controller,
            "moveit": MoveItArmController,
            "moveit_arm": MoveItArmController,
            "moveit_head": MoveItHeadController,
            "moveit_gripper": MoveItGripperController,
            "hsr_gripper": MoveItGripperController,
            # HSR specific entries
            "hsrb_nav": HsrbNavController,
            "hsrb_arm": HsrbArmController,
            "hsrb_head": HsrbHeadController,
            "hsrb_gripper": HsrbGripperController,
            "ros2_control": Ros2ControlController,
            "ros2control": Ros2ControlController,
        }
        for name, cls in builtins_actions.items():
            self._actions.setdefault(name, cls)
        for name, cls in builtins_controllers.items():
            self._controllers.setdefault(name, cls)
        self._planners.setdefault("curobo", CuroboManipulationPlanner)
        self._planners.setdefault("moveit", MoveItManipulationPlanner)

    def register_action(self, name: str, action_cls: type) -> None:
        """Manually register an action."""
        self._actions[name] = action_cls

    def register_controller(self, name: str, controller_cls: type) -> None:
        """Manually register a controller."""
        self._controllers[name] = controller_cls

    def register_condition(self, name: str, condition_cls: type) -> None:
        """Manually register a condition."""
        self._conditions[name] = condition_cls

    def register_planner(self, name: str, planner_cls: type) -> None:
        """Manually register a planner."""
        self._planners[name] = planner_cls

    def get_action(self, name: str) -> type | None:
        """Get action class by name."""
        return self._actions.get(name)

    def get_controller(self, name: str) -> type | None:
        """Get controller class by name."""
        return self._controllers.get(name)

    def get_condition(self, name: str) -> type | None:
        """Get condition class by name."""
        return self._conditions.get(name)

    def get_planner(self, name: str) -> type | None:
        """Get planner class by name."""
        return self._planners.get(name)

    @property
    def available_actions(self) -> list[str]:
        """List all available action names."""
        return list(self._actions.keys())

    @property
    def available_controllers(self) -> list[str]:
        """List all available controller names."""
        return list(self._controllers.keys())

    @property
    def available_conditions(self) -> list[str]:
        """List all available condition names."""
        return list(self._conditions.keys())

    @property
    def available_planners(self) -> list[str]:
        """List all available planner names."""
        return list(self._planners.keys())
