#!/usr/bin/env bash
set -eo pipefail

# Offline AMCL using the pre-cleaned localization bag.
#
# Default inputs:
#   Bag:        /bag/localization_clean
#   Map:        /bag/orchid_map.yaml
#   Start pose: /bag/orchid_map_start_pose.yaml
#   Output:     /bag/amcl_localization_clean
#   Rate:       1.0x
#
# Usage:
#   ./run_offline_amcl.sh
#
# Or:
#   ./run_offline_amcl.sh BAG MAP START_POSE OUTPUT_BAG RATE

BAG="${1:-/bag/localization_clean}"
MAP="${2:-/bag/orchid_map.yaml}"
START_POSE="${3:-/bag/orchid_map_start_pose.yaml}"
OUTPUT_BAG="${4:-/bag/amcl_localization_clean}"
RATE="${5:-1.0}"

AMCL_PARAMS="/home/orchid/ros2_ws/src/orchid_mapping_config/config/amcl.yaml"
RECORD_LOG="/bag/amcl_record_clean.log"
PLAY_LOG="/bag/amcl_play_clean.log"

source /opt/ros/humble/setup.bash
source /home/orchid/ros2_ws/install/setup.bash

MAP_PID=""
AMCL_PID=""
LIFECYCLE_PID=""
REC_PID=""
PLAY_PID=""

cleanup() {
    echo
    echo "Cleaning up..."

    if [[ -n "${PLAY_PID}" ]]; then
        kill -INT "${PLAY_PID}" 2>/dev/null || true
        wait "${PLAY_PID}" 2>/dev/null || true
        PLAY_PID=""
    fi

    if [[ -n "${REC_PID}" ]]; then
        kill -INT "${REC_PID}" 2>/dev/null || true
        wait "${REC_PID}" 2>/dev/null || true
        REC_PID=""
    fi

    if [[ -n "${LIFECYCLE_PID}" ]]; then
        kill -INT "${LIFECYCLE_PID}" 2>/dev/null || true
        wait "${LIFECYCLE_PID}" 2>/dev/null || true
        LIFECYCLE_PID=""
    fi

    if [[ -n "${AMCL_PID}" ]]; then
        kill -INT "${AMCL_PID}" 2>/dev/null || true
        wait "${AMCL_PID}" 2>/dev/null || true
        AMCL_PID=""
    fi

    if [[ -n "${MAP_PID}" ]]; then
        kill -INT "${MAP_PID}" 2>/dev/null || true
        wait "${MAP_PID}" 2>/dev/null || true
        MAP_PID=""
    fi

    sleep 1

    # Catch child processes that may remain after `ros2 run`.
    pkill -INT -f "/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager" 2>/dev/null || true
    pkill -INT -f "/opt/ros/humble/lib/nav2_amcl/amcl" 2>/dev/null || true
    pkill -INT -f "/opt/ros/humble/lib/nav2_map_server/map_server" 2>/dev/null || true
}

trap cleanup EXIT

echo "========================================"
echo "Offline AMCL localization"
echo "========================================"
echo "Bag:        ${BAG}"
echo "Map:        ${MAP}"
echo "Start pose: ${START_POSE}"
echo "Output:     ${OUTPUT_BAG}"
echo "Rate:       ${RATE}x"
echo

if [[ ! -d "${BAG}" ]]; then
    echo "ERROR: bag does not exist: ${BAG}"
    exit 1
fi

if [[ ! -f "${MAP}" ]]; then
    echo "ERROR: map YAML does not exist: ${MAP}"
    exit 1
fi

if [[ ! -f "${START_POSE}" ]]; then
    echo "ERROR: start pose file does not exist: ${START_POSE}"
    exit 1
fi

if [[ ! -f "${AMCL_PARAMS}" ]]; then
    echo "ERROR: AMCL parameter file does not exist: ${AMCL_PARAMS}"
    exit 1
fi

echo "Removing stale localization processes..."

pkill -INT -f "ros2 bag play.*localization_clean" 2>/dev/null || true
pkill -INT -f "ros2 bag record.*amcl_localization" 2>/dev/null || true
pkill -INT -f "/opt/ros/humble/lib/nav2_map_server/map_server" 2>/dev/null || true
pkill -INT -f "/opt/ros/humble/lib/nav2_amcl/amcl" 2>/dev/null || true
pkill -INT -f "/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager" 2>/dev/null || true
pkill -INT -f "ros2 run nav2_map_server map_server" 2>/dev/null || true
pkill -INT -f "ros2 run nav2_amcl amcl" 2>/dev/null || true
pkill -INT -f "ros2 run nav2_lifecycle_manager lifecycle_manager" 2>/dev/null || true

sleep 2

echo "Starting map server..."
ros2 run nav2_map_server map_server \
    --ros-args \
    -p yaml_filename:="${MAP}" \
    -p use_sim_time:=true \
    </dev/null &
MAP_PID=$!

echo "Starting AMCL..."
ros2 run nav2_amcl amcl \
    --ros-args \
    --params-file "${AMCL_PARAMS}" \
    -p use_sim_time:=true \
    </dev/null &
AMCL_PID=$!

echo "Waiting for map_server and AMCL nodes..."

for i in {1..30}; do
    NODES="$(ros2 node list 2>/dev/null || true)"

    if echo "${NODES}" | grep -qx "/map_server" && \
       echo "${NODES}" | grep -qx "/amcl"; then
        echo "map_server and AMCL are available."
        break
    fi

    if [[ "${i}" -eq 30 ]]; then
        echo "ERROR: map_server or AMCL did not appear."
        exit 1
    fi

    sleep 0.5
done

sleep 1

echo "Starting lifecycle manager..."
ros2 run nav2_lifecycle_manager lifecycle_manager \
    --ros-args \
    -p autostart:=true \
    -p use_sim_time:=true \
    -p node_names:="['map_server','amcl']" \
    </dev/null &
LIFECYCLE_PID=$!

echo "Waiting for AMCL to become active..."

for i in {1..40}; do
    STATE="$(ros2 lifecycle get /amcl 2>/dev/null || true)"

    if echo "${STATE}" | grep -Eqi '^active([[:space:]]|\[|$)'; then
        echo "AMCL is active."
        break
    fi

    if [[ "${i}" -eq 40 ]]; then
        echo "ERROR: AMCL failed to become active."
        echo "Last state: ${STATE}"
        exit 1
    fi

    sleep 0.5
done

MAP_STATE="$(ros2 lifecycle get /map_server 2>/dev/null || true)"

if ! echo "${MAP_STATE}" | grep -Eqi '^active([[:space:]]|\[|$)'; then
    echo "ERROR: map_server is not active."
    echo "State: ${MAP_STATE}"
    exit 1
fi

echo "map_server is active."

echo "Removing previous output..."
rm -rf "${OUTPUT_BAG}"
rm -f "${RECORD_LOG}" "${PLAY_LOG}"

echo "Starting localization recorder..."
ros2 bag record \
    /amcl_pose \
    /tf \
    /tf_static \
    -o "${OUTPUT_BAG}" \
    </dev/null > "${RECORD_LOG}" 2>&1 &
REC_PID=$!

sleep 1

echo "Starting clean localization bag at ${RATE}x..."
ros2 bag play "${BAG}" \
    --clock \
    --rate "${RATE}" \
    </dev/null > "${PLAY_LOG}" 2>&1 &
PLAY_PID=$!

echo "Waiting for simulated time and publishing initial pose..."

python3 - "${START_POSE}" <<'PY'
import sys
import yaml
import rclpy

from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseWithCovarianceStamped

pose_file = sys.argv[1]

with open(pose_file, "r") as f:
    data = yaml.safe_load(f)

p = data["position"]
q = data["orientation"]

rclpy.init()

node = Node(
    "offline_initial_pose",
    parameter_overrides=[
        Parameter("use_sim_time", Parameter.Type.BOOL, True)
    ],
)

pub = node.create_publisher(
    PoseWithCovarianceStamped,
    "/initialpose",
    10,
)

# Wait for /clock.
for _ in range(100):
    rclpy.spin_once(node, timeout_sec=0.1)

    if node.get_clock().now().nanoseconds > 0:
        break
else:
    raise RuntimeError("Timed out waiting for simulated /clock.")

# Allow TF to populate briefly.
for _ in range(10):
    rclpy.spin_once(node, timeout_sec=0.1)

msg = PoseWithCovarianceStamped()
msg.header.frame_id = "map"

msg.pose.pose.position.x = float(p["x"])
msg.pose.pose.position.y = float(p["y"])
msg.pose.pose.position.z = float(p.get("z", 0.0))

msg.pose.pose.orientation.x = float(q["x"])
msg.pose.pose.orientation.y = float(q["y"])
msg.pose.pose.orientation.z = float(q["z"])
msg.pose.pose.orientation.w = float(q["w"])

# Planar initial uncertainty:
# x std = 0.5 m, y std = 0.5 m, yaw std ~= 15 degrees.
msg.pose.covariance[0] = 0.25
msg.pose.covariance[7] = 0.25
msg.pose.covariance[35] = 0.0685

for _ in range(5):
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)

print(
    "Initial pose published at simulated time:",
    msg.header.stamp.sec,
    msg.header.stamp.nanosec,
)

node.destroy_node()
rclpy.shutdown()
PY

echo
echo "========================================"
echo "Localization running"
echo "========================================"
echo "Waiting for bag playback to finish..."
echo

wait "${PLAY_PID}"
PLAY_PID=""

echo "Bag playback finished."

echo "Stopping localization recorder..."
kill -INT "${REC_PID}" 2>/dev/null || true
wait "${REC_PID}" 2>/dev/null || true
REC_PID=""

echo
echo "========================================"
echo "Localization complete"
echo "========================================"
echo "Output bag:   ${OUTPUT_BAG}"
echo "Recorder log: ${RECORD_LOG}"
echo "Playback log: ${PLAY_LOG}"
