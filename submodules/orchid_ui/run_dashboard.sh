#!/usr/bin/env bash
set -eo pipefail
ui_root=${ORCHID_UI_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$ui_root"
lock_file=${ORCHID_UI_LOCK_FILE:-/tmp/orchid-ui-${UID}.lock}
if [[ $(readlink "/proc/$$/fd/9" 2>/dev/null || true) != "$lock_file" ]]; then exec 9>"$lock_file"; fi
if ! flock -n 9; then echo "Dashboard is already building or running ($lock_file)" >&2; exit 1; fi
source "${ORCHID_UI_ROS_SETUP:-/opt/ros/humble/setup.bash}"
binary=${ORCHID_UI_BINARY:-$ui_root/target/release/orchid_ui}
workspace_ready() (
    [[ -f ros2_ws/install/setup.bash ]] || exit 1
    [[ -f ros2_ws/install/orchid_ros_interfaces/share/ament_index/resource_index/packages/orchid_ros_interfaces ||
       -f ros2_ws/install/share/ament_index/resource_index/packages/orchid_ros_interfaces ]] || exit 1
    source ros2_ws/install/setup.bash || exit 1
    python3 -c 'from orchid_ros_interfaces.action import ExecuteTask, ExecuteWorkflow; from orchid_ros_interfaces.srv import GetUserInput; from rclpy.type_support import check_for_type_support; [check_for_type_support(message) for message in (ExecuteTask.Goal, ExecuteWorkflow.Goal, GetUserInput.Request)]' 9>&-
)
needs_build=0
if ! workspace_ready >/dev/null 2>&1; then echo "ROS2 workspace missing or incomplete; building dashboard"; needs_build=1; fi
if [[ ! -x "$binary" ]]; then echo "Dashboard release binary missing; building dashboard"; needs_build=1; fi
build_pid=
stop_build() {
    trap - EXIT
    trap '' INT TERM
    if [[ -n "$build_pid" ]]; then
        kill -TERM -- "-$build_pid" 2>/dev/null || true
        local deadline=$((SECONDS+5))
        while kill -0 -- "-$build_pid" 2>/dev/null && ((SECONDS<deadline)); do sleep .1; done
        if kill -0 -- "-$build_pid" 2>/dev/null; then kill -KILL -- "-$build_pid" 2>/dev/null || true; fi
        wait "$build_pid" 2>/dev/null || true
    fi
}
trap stop_build EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$needs_build" == 1 ]]; then
    set +m
    setsid bash "$ui_root/build.sh" &
    build_pid=$!
    if wait "$build_pid"; then build_pid=; else result=$?; echo "Dashboard build failed (status $result); not starting" >&2; exit "$result"; fi
    if ! workspace_ready; then echo "ROS2 interfaces are still unavailable after build; not starting" >&2; exit 1; fi
    if [[ ! -x "$binary" ]]; then echo "Dashboard binary is still missing after build: $binary" >&2; exit 1; fi
fi
trap - EXIT INT TERM
exec bash "$ui_root/start_dashboard.sh"
