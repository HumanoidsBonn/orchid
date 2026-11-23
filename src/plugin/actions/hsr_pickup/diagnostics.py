"""Optional offline exports/replays; never used to communicate with the server."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .model import PickupInputs


def export_inputs(inputs, directory, target):
    from rclpy.serialization import serialize_message
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/"robot.urdf").write_text(inputs.robot_description)
    (directory/"robot.srdf").write_text(inputs.robot_description_semantic)
    (directory/"joint_limits.json").write_text(json.dumps(inputs.joint_limits,indent=2))
    (directory/"scene.cdr").write_bytes(serialize_message(inputs.scene))
    (directory/"target.cdr").write_bytes(serialize_message(inputs.target))
    np.savez_compressed(directory/"observations.npz",grasp=inputs.grasp,target_points=inputs.target_points,surround_points=inputs.surround_points)
    (directory/"inputs.json").write_text(json.dumps({"target":target,"frame_id":inputs.frame_id,"start_joints":inputs.start_joints,"go_joints":inputs.go_joints},indent=2))


def load_inputs(directory, go_joints):
    """Read new exports or earlier captured files for offline regression only."""
    from rclpy.serialization import deserialize_message
    from moveit_msgs.msg import PlanningScene, CollisionObject
    directory=Path(directory)
    metadata=json.loads((directory/("inputs.json" if (directory/"inputs.json").exists() else "snapshot.json")).read_text())
    observed=np.load(directory/"observations.npz")
    scene=deserialize_message((directory/"scene.cdr").read_bytes(),PlanningScene)
    return PickupInputs(metadata["frame_id"],metadata["start_joints"],observed["grasp"],observed["target_points"],observed["surround_points"],
                        (directory/"robot.urdf").read_text(),(directory/"robot.srdf").read_text(),json.loads((directory/"joint_limits.json").read_text()),
                        scene,deserialize_message((directory/"target.cdr").read_bytes(),CollisionObject),metadata.get("go_joints",go_joints))


def plan_report(plan):
    return {"target":plan.target,"requested_mode":plan.requested_mode,"selected_mode":plan.selected_mode,"staging_order":plan.staging_order,
            "success":bool(plan.motions),"diagnostics":plan.diagnostics,"candidates":plan.candidates,
            "motions":[{"name":m.name,"start_joints":m.start_joints,"end_joints":m.end_joints,"attached":m.attached,
                "trajectory":{"joint_names":list(m.trajectory.joint_names),"points":[{"positions":list(p.positions),"velocities":list(p.velocities),"accelerations":list(p.accelerations),"time_from_start":p.time_from_start.sec+p.time_from_start.nanosec*1e-9} for p in m.trajectory.points]},
                "hand_path":m.hand_path,"base_path":m.base_path} for m in plan.motions]}
