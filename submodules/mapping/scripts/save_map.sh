#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-maps/orchid_map}"

source /opt/ros/humble/setup.bash
if [[ -f /home/orchid/ros2_ws/install/setup.bash ]]; then
  source /home/orchid/ros2_ws/install/setup.bash
fi

mkdir -p "$(dirname "$OUT")"
ros2 run nav2_map_server map_saver_cli -f "$OUT"
