#!/usr/bin/env bash
set -eo pipefail
ui_root=${ORCHID_UI_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$ui_root"
lock_file=${ORCHID_UI_LOCK_FILE:-/tmp/orchid-ui-${UID}.lock}
if [[ $(readlink "/proc/$$/fd/9" 2>/dev/null || true) != "$lock_file" ]]; then exec 9>"$lock_file"; fi
if ! flock -n 9; then echo "Dashboard is already building or running ($lock_file)" >&2; exit 1; fi
echo "Building Orchid HRI Dashboard..."
for tool in cargo rustc; do
    if ! version=$("$tool" --version 9>&- 2>/dev/null); then
        echo "$tool is missing. Rebuild the UI image or install Rust 1.85.1 with rustup inside the container." >&2
        exit 1
    fi
    if [[ ! "$version" =~ ^${tool}[[:space:]]+([0-9]+)\.([0-9]+) ]] ||
       ! ((BASH_REMATCH[1]>1 || (BASH_REMATCH[1]==1 && BASH_REMATCH[2]>=85))); then
        echo "Unsupported $version; Rust/Cargo >=1.85 is required. Rebuild the UI image or install Rust 1.85.1 with rustup inside the container." >&2
        exit 1
    fi
done
source "${ORCHID_UI_ROS_SETUP:-/opt/ros/humble/setup.bash}"
cd "$ui_root/ros2_ws"
colcon build --packages-select orchid_ros_interfaces 9>&-
source install/setup.bash
cd "$ui_root"
cargo build --release --locked 9>&-
echo "Build complete: $ui_root/target/release/orchid_ui"
