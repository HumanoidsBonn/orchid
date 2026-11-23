#!/usr/bin/env python3
"""Capture the first valid robot pose in a SLAM map frame."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Output start-pose YAML path")
    parser.add_argument("--parent-frame", default="map")
    parser.add_argument("--child-frame", default="base_footprint")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing output file"
    )
    return parser.parse_args()


def transform_to_pose(transform, parent_frame):
    position = transform.transform.translation
    orientation = transform.transform.rotation
    values = (
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("TF contains a non-finite value")

    norm = math.sqrt(
        orientation.x**2
        + orientation.y**2
        + orientation.z**2
        + orientation.w**2
    )
    if norm <= 1e-12:
        raise ValueError("TF contains a zero-length quaternion")

    return {
        "frame_id": parent_frame,
        "position": {
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
        },
        "orientation": {
            "x": float(orientation.x / norm),
            "y": float(orientation.y / norm),
            "z": float(orientation.z / norm),
            "w": float(orientation.w / norm),
        },
    }


def write_pose(output, pose):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(pose, stream, sort_keys=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()
    output = Path(args.output).expanduser().resolve()

    if args.timeout_sec <= 0:
        raise SystemExit("ERROR: --timeout-sec must be greater than zero")
    if output.exists() and not args.force:
        raise SystemExit(
            f"ERROR: output already exists: {output}\n"
            "Pass --force to replace it for a new SLAM run."
        )

    rclpy.init()
    node = Node(
        "capture_slam_start_pose",
        parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True)
        ],
    )
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    deadline = time.monotonic() + args.timeout_sec
    last_error = None

    print(
        f"Waiting for {args.parent_frame} -> {args.child_frame}...",
        flush=True,
    )

    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.get_clock().now().nanoseconds <= 0:
                continue

            try:
                transform = buffer.lookup_transform(
                    args.parent_frame, args.child_frame, Time()
                )
                pose = transform_to_pose(transform, args.parent_frame)
            except (TransformException, ValueError) as error:
                last_error = error
                continue

            write_pose(output, pose)
            print(yaml.safe_dump(pose, sort_keys=False), end="")
            print(f"Saved: {output}")
            return 0
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()

    detail = f" Last TF error: {last_error}" if last_error else ""
    raise SystemExit(
        f"ERROR: timed out waiting for {args.parent_frame} -> "
        f"{args.child_frame}.{detail}"
    )


if __name__ == "__main__":
    sys.exit(main())
