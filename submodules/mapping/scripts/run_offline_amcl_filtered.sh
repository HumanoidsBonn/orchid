#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/orchid/ros2_ws/install/setup.bash

BAG="${1:-/bag/playable_bag}"
MAP="${2:-/bag/orchid_map.yaml}"
START_POSE="${3:-/bag/orchid_map_start_pose.yaml}"
OUTPUT_BAG="${4:-/bag/amcl_localization_run}"

FILTER="/home/orchid/orchid_server/submodules/mapping/scripts/filter_recorded_tf.py"
AMCL_PARAMS="/home/orchid/ros2_ws/src/orchid_mapping_config/config/amcl.yaml"

FILTER_PID=""
MAP_SERVER_PID=""
AMCL_PID=""
LIFECYCLE_PID=""
RECORDER_PID=""
BAG_PID=""

cleanup() {
    echo
    echo "Stopping localization processes..."

    for pid in \
        "${BAG_PID}" \
        "${RECORDER_PID}" \
        "${FILTER_PID}" \
        "${MAP_SERVER_PID}" \
        "${AMCL_PID}" \
        "${LIFECYCLE_PID}"
    do
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT INT TERM

echo "========================================"
echo " Orchid Offline AMCL Localization"
echo "========================================"
echo "Bag:        ${BAG}"
echo "Map:        ${MAP}"
echo "Start pose: ${START_POSE}"
echo "Output:     ${OUTPUT_BAG}"
echo

echo "[1/7] Starting TF filter..."
python3 "${FILTER}" &
FILTER_PID=$!

sleep 1

echo "[2/7] Starting map server..."
ros2 run nav2_map_server map_server \
    --ros-args \
    -p yaml_filename:="${MAP}" \
    -p use_sim_time:=true &
MAP_SERVER_PID=$!

sleep 1

echo "[3/7] Starting AMCL..."
ros2 run nav2_amcl amcl \
    --ros-args \
    --params-file "${AMCL_PARAMS}" \
    -p use_sim_time:=true &
AMCL_PID=$!

sleep 1

echo "[4/7] Starting lifecycle manager..."
ros2 run nav2_lifecycle_manager lifecycle_manager \
    --ros-args \
    -p use_sim_time:=true \
    -p autostart:=true \
    -p node_names:="['map_server','amcl']" &
LIFECYCLE_PID=$!

sleep 2

echo "[5/7] Starting localization recorder..."

rm -rf "${OUTPUT_BAG}"

ros2 bag record \
    /amcl_pose \
    /tf \
    -o "${OUTPUT_BAG}" &
RECORDER_PID=$!

sleep 1

echo "[6/7] Starting rosbag playback..."

ros2 bag play "${BAG}" \
    --clock \
    --remap /tf:=/tf_bag &
BAG_PID=$!

# Give /clock, odom, TF and scans a moment to start flowing.
sleep 5

echo "[7/7] Publishing initial pose..."

python3 - "${START_POSE}" <<'PY'
import sys
import yaml
import subprocess

path = sys.argv[1]

with open(path, "r") as f:
    pose = yaml.safe_load(f)

x = float(pose["position"]["x"])
y = float(pose["position"]["y"])
z = float(pose["position"].get("z", 0.0))

qx = float(pose["orientation"].get("x", 0.0))
qy = float(pose["orientation"].get("y", 0.0))
qz = float(pose["orientation"]["z"])
qw = float(pose["orientation"]["w"])

msg = f"""
{{
  header: {{frame_id: 'map'}},
  pose: {{
    pose: {{
      position: {{x: {x}, y: {y}, z: {z}}},
      orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}
    }},
    covariance: [
      0.25,0,0,0,0,0,
      0,0.25,0,0,0,0,
      0,0,0,0,0,0,
      0,0,0,0,0,0,
      0,0,0,0,0,0,
      0,0,0,0,0,0.0685
    ]
  }}
}}
"""

print(
    f"Initial pose: "
    f"x={x:.3f}, y={y:.3f}, "
    f"qz={qz:.4f}, qw={qw:.4f}"
)

subprocess.run(
    [
        "ros2",
        "topic",
        "pub",
        "--once",
        "/initialpose",
        "geometry_msgs/msg/PoseWithCovarianceStamped",
        msg,
    ],
    check=True,
)
PY

echo
echo "Initial pose published."
echo "Waiting for bag playback to finish..."
echo

wait "${BAG_PID}"

echo
echo "Bag playback finished."
echo "Stopping localization recorder..."

kill "${RECORDER_PID}" 2>/dev/null || true
wait "${RECORDER_PID}" 2>/dev/null || true
RECORDER_PID=""

echo
echo "========================================"
echo " Localization run complete"
echo "========================================"
echo
echo "Recorded localization:"
echo "  ${OUTPUT_BAG}"
echo
echo "This bag contains:"
echo "  /amcl_pose"
echo "  /tf"
echo
echo "Press Ctrl+C to stop AMCL/map server."
echo

wait