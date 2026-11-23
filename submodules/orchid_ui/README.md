# Orchid HRI Web Dashboard

Orchid UI is the browser dashboard for the Orchid ROS2 orchestration stack. It provides a command surface for sending robot tasks, live execution feedback, inline user follow-up prompts, operation timing, behavior-tree visualization, ROS logs, map/pose display, and camera display.

## Quick Start

Run the UI from the Orchid UI Docker environment. From the Orchid server repository root:

```bash
cd submodules/orchid_ui/humble
docker compose up -d
```

Compose runs `run_dashboard.sh`. It checks the installed ROS2 interfaces and
release binary, builds missing/incomplete artifacts with `build.sh`, then starts
the dashboard. No separate initial build is required.

If running the container manually instead, use:

```bash
cd /home/orchid/orchid_ui
./run_dashboard.sh
```

For source updates, stop the launcher before explicitly running `./build.sh`,
then run `./run_dashboard.sh` again. Startup checks detect missing/incomplete
builds; they do not rebuild merely because sources changed.

Open the UI at:

```text
http://localhost:8000
```

Use the default connection values unless your ROS bridge or robot ID differs:

```text
Bridge IP: ws://<dashboard-host>:9090
Robot ID: hsrb_bot
```

The startup script launches:

- `rosbridge_websocket` on `ws://localhost:9090`
- the Rust/Axum dashboard server on `http://localhost:8000`

## Current UI Layout

- **Map** (`/#map`) fills the workspace with a draggable camera picture-in-picture.
- **Behaviour Tree** (`/#tree`) fills the workspace; leaf taps open a draggable details panel.
- Shared robot/task/battery ribbon and command/question bar preserve state across pages.
- Connection address/robot ID and display settings are in side-menu dropdowns. Compact connection/theme controls are beside the menu. Logs start closed and unsubscribed.
- One-finger pan and two-finger pinch operate inside each viewport.