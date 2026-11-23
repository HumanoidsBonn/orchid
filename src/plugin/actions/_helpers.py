"""Shared helpers for parametric plugin actions."""

from __future__ import annotations

from typing import Any
from dataclasses import replace
from src.core.failure import FailureInfo, FailureCode, FailureOrigin

from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
from src.core.controllers import BaseArmController, BaseGripperController, BaseHeadController, BaseNavController, Pose3D
import src.core.logger as log 


def _nav_status_result(nav: BaseNavController) -> ActionResult:
    status = nav.get_status()
    if status == "succeeded":
        return ActionResult(status="success", payload={"pose": nav.get_current_pose().__dict__})
    if status == "in_progress":
        return ActionResult(status="running")
    return ActionResult(status="failure", reason=nav.get_error() or f"navigation {status}",
                        failure=FailureInfo(FailureCode.CONTROLLER_FAILED, origin=FailureOrigin.CONTROLLER))


def _arm_status_result(arm: BaseArmController) -> ActionResult:
    status = arm.get_status()
    if status == "succeeded":
        log.info("ARM STATUS RESULT: %s", status)
        return ActionResult(status="success")
    elif status == "in_progress":
        log.debug("ARM STATUS RESULT: %s", status)
        return ActionResult(status="running")
    reason = arm.get_error() or f"arm {status} (controller supplied no error detail)"
    failure = arm.get_failure()
    failure = replace(failure if isinstance(failure, FailureInfo) else FailureInfo(FailureCode.CONTROLLER_FAILED), origin=FailureOrigin.CONTROLLER)
    log.warn("ARM STATUS RESULT: %s; reason=%s; failure=%s", status, reason, failure)
    return ActionResult(status="failure", reason=reason, failure=failure)


def _gripper_status_result(gripper: BaseGripperController) -> ActionResult:
    status = gripper.get_status()
    if status == "succeeded":
        log.info("GRIPPER STATUS RESULT: %s", status)
        return ActionResult(status="success", payload={"state": gripper.get_state()})
    if status == "in_progress":
        log.debug("GRIPPER STATUS RESULT: %s", status)
        return ActionResult(status="running")
    log.warn("GRIPPER STATUS RESULT: %s", status)
    return ActionResult(status="failure", reason=gripper.get_error() or f"gripper {status}",
                        failure=FailureInfo(FailureCode.CONTROLLER_FAILED, retryable=True, origin=FailureOrigin.CONTROLLER))


def _head_status_result(head: BaseHeadController) -> ActionResult:
    status = head.get_status()
    if status == "succeeded":
        log.info("HEAD STATUS RESULT: %s", status)
        return ActionResult(status="success")
    if status == "in_progress":
        log.debug("HEAD STATUS RESULT: %s", status)
        return ActionResult(status="running")
    log.warn("HEAD STATUS RESULT: %s", status)
    return ActionResult(status="failure", reason=head.get_error() or f"head {status}",
                        failure=FailureInfo(FailureCode.CONTROLLER_FAILED, origin=FailureOrigin.CONTROLLER))


def _pose_from_params(params: dict[str, Any], default_frame: str = "base_link") -> Pose3D:
    return Pose3D(
        x=float(params.get("x", 0.0)),
        y=float(params.get("y", 0.0)),
        z=float(params.get("z", 0.0)),
        roll=float(params.get("roll", 0.0)),
        pitch=float(params.get("pitch", 0.0)),
        yaw=float(params.get("yaw", 0.0)),
        frame_id=str(params.get("frame_id", default_frame)),
    )

class ControllerAsyncAction(AsyncAction):
    """Executable leaf with cancellation for its declared controller roles."""
    
    def cancel(self, context: Context) -> None:
        if "nav" in self.required_controllers and context.nav:
            context.nav.cancel_goal()
        if "arm" in self.required_controllers and context.arm:
            context.arm.cancel_goal()
        if "head" in self.required_controllers and context.head:
            context.head.cancel_goal()
        if "gripper" in self.required_controllers and context.gripper:
            context.gripper.cancel_goal()
def scene_location_description(context, target: str) -> tuple[str, str | None]:
    """Human-readable furniture and room names for action-owned speech."""
    import re
    def spoken_name(value):
        name=str(value).replace("_"," ").strip()
        return re.sub(r"(?:[\s-]*\d+)+$","",name).strip() or name
    from src.core.scene_graph_query import SceneGraphQuery
    graph = context.scene_graph
    if graph is None:
        return spoken_name(target), None
    query = SceneGraphQuery(graph)
    node = graph.get_node(target)
    if node is not None and node.type == "object":
        target = query.get_supporting_furniture(target) or target
        node = graph.get_node(target)
    name = (node.attributes.get("name") or node.attributes.get("label") or node.id) if node else target
    room_id = query.get_containing_room(target) if node else None
    room = graph.get_node(room_id) if room_id else None
    room_name = (room.attributes.get("room_type") or room.attributes.get("name") or room.id) if room else None
    return spoken_name(name), spoken_name(room_name) if room_name else None


def scene_node_reference(context, node_id):
    node=context.scene_graph.get_node(node_id) if context.scene_graph else None
    return {"id":node_id,"boxer_id":node.attributes.get("boxer_id")} if node else None

