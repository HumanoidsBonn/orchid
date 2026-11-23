#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/orchid/ros2_ws/install/setup.bash

BAG="${1:-/bag/playable_bag}"

PARAMS="/home/orchid/ros2_ws/src/orchid_mapping_config/config/slam_params.yaml"
FILTER="/home/orchid/orchid_server/submodules/mapping/scripts/filter_recorded_tf.py"

FILTER_PID=""
SLAM_PID=""

cleanup() {
    echo
    echo "Stopping mapping processes..."

    if [[ -n "${FILTER_PID}" ]]; then
        kill "${FILTER_PID}" 2>/dev/null || true
    fi

    if [[ -n "${SLAM_PID}" ]]; then
        kill "${SLAM_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "========================================"
echo " Orchid Offline Mapping"
echo "========================================"
echo "Bag: ${BAG}"
echo

echo "[1/3] Starting TF filter..."
python3 "${FILTER}" &
FILTER_PID=$!

sleep 1

echo "[2/3] Starting SLAM Toolbox..."
ros2 run slam_toolbox async_slam_toolbox_node \
    --ros-args \
    --params-file "${PARAMS}" \
    -p use_sim_time:=true &
SLAM_PID=$!

sleep 2

echo "[3/3] Playing rosbag..."
echo

ros2 bag play "${BAG}" \
    --clock \
    --remap /tf:=/tf_bag

echo
echo "========================================"
echo " Bag playback finished"
echo "========================================"
echo
echo "SLAM Toolbox is still running."
echo
echo "Open another terminal and save the map with:"
echo
echo "  ros2 run nav2_map_server map_saver_cli -f /bag/orchid_map"
echo
echo "Press Ctrl+C here AFTER the map has been saved."
echo

wait