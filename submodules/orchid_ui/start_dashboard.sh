#!/usr/bin/env bash
set -eo pipefail
ui_root=${ORCHID_UI_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
cd "$ui_root"
lock_file=${ORCHID_UI_LOCK_FILE:-/tmp/orchid-ui-${UID}.lock}
if [[ $(readlink "/proc/$$/fd/9" 2>/dev/null || true) != "$lock_file" ]]; then exec 9>"$lock_file"; fi
if ! flock -n 9; then echo "Dashboard is already building or running ($lock_file)" >&2; exit 1; fi
children=(); names=(); logs=()
cleanup_started=0
cleanup() {
    ((cleanup_started==0)) || return 0
    cleanup_started=1
    trap - EXIT
    trap '' INT TERM
    local pid deadline alive
    for pid in "${children[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    deadline=$((SECONDS+5))
    while :; do
        alive=0
        for pid in "${children[@]}"; do if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi; done
        [[ "$alive" == 1 ]] && ((SECONDS<deadline)) || break
        sleep .1
    done
    for pid in "${children[@]}"; do
        if kill -0 -- "-$pid" 2>/dev/null; then
            echo "Force-stopping dashboard process group $pid" >&2
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    done
    for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
source "${ORCHID_UI_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f ros2_ws/install/setup.bash ]]; then source ros2_ws/install/setup.bash; fi
set -u
binary=${ORCHID_UI_BINARY:-$ui_root/target/release/orchid_ui}
if [[ ! -x "$binary" ]]; then echo "Dashboard binary missing; use ./run_dashboard.sh to build and start" >&2; exit 1; fi
log_dir=${ORCHID_UI_LOG_DIR:-/tmp}
mkdir -p "$log_dir"
set +m
start_component() {
    local name=$1 logfile=$2
    shift 2
    setsid -- "$@" 9>&- >"$logfile" 2>&1 &
    children+=("$!"); names+=("$name"); logs+=("$logfile")
}
check_children() {
    local index
    for index in "${!children[@]}"; do
        if ! kill -0 "${children[index]}" 2>/dev/null; then
            echo "Dashboard component ${names[index]} exited during startup; log: ${logs[index]}" >&2
            return 1
        fi
    done
}
if [[ "${ORCHID_UI_HTTP_ONLY:-0}" != 1 ]]; then
    start_component 'rosbridge/rosapi' "$log_dir/rosbridge.log" ros2 launch rosbridge_server rosbridge_websocket_launch.xml send_action_goals_in_new_thread:=true
    start_component 'camera preview' "$log_dir/orchid_ui_camera.log" python3 scripts/camera_preview.py
fi
start_component 'web server' "$log_dir/webserver.log" "$binary"
ready=0
for _ in {1..50}; do
    check_children || exit 1
    if curl -fsS --max-time 1 "http://127.0.0.1:${ORCHID_UI_PORT:-8000}/healthz" 9>&- >/dev/null 2>&1; then ready=1; break; fi
    sleep .1
done
if [[ "$ready" != 1 ]]; then echo "Dashboard HTTP readiness timed out; log: $log_dir/webserver.log" >&2; exit 1; fi
check_children || exit 1
echo "Dashboard ready at http://0.0.0.0:${ORCHID_UI_PORT:-8000}"
set +e
wait -n -p exited_pid "${children[@]}"
result=$?
for index in "${!children[@]}"; do
    if [[ "${children[index]}" == "${exited_pid:-}" ]]; then
        echo "Dashboard component ${names[index]} exited (status $result); log: ${logs[index]}; stopping owned processes" >&2
    fi
done
exit 1
