#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-maps/orchid_pose_graph}"

source /opt/ros/humble/setup.bash
if [[ -f /home/orchid/ros2_ws/install/setup.bash ]]; then
  source /home/orchid/ros2_ws/install/setup.bash
fi

mkdir -p "$(dirname "$OUT")"
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '${OUT}'}"
