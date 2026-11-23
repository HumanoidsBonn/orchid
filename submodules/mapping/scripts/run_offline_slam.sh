#!/usr/bin/env bash
set -eo pipefail

# Run the repository's offline SLAM launch directly on the pre-cleaned bag.
# No live Python TF relay and no /tf remapping are used here.
#
# Usage:
#   ./run_offline_slam.sh
#   ./run_offline_slam.sh /bag/localization_clean

BAG="${1:-/bag/localization_clean}"

source /opt/ros/humble/setup.bash
source /home/orchid/ros2_ws/install/setup.bash

if [[ ! -d "${BAG}" ]]; then
    echo "ERROR: bag does not exist: ${BAG}"
    exit 1
fi

echo "========================================"
echo "Offline SLAM"
echo "========================================"
echo "Bag: ${BAG}"
echo
echo "This uses the cleaned bag directly."
echo "No runtime TF filtering is performed."
echo

exec ros2 launch orchid_mapping_config offline_slam.launch.py \
    bag_path:="${BAG}"
