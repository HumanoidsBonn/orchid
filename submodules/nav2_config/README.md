# Orchid Nav2 configuration

Build/source `orchid_nav2_config`, then launch with two map YAML paths:

```bash
ros2 launch orchid_nav2_config navigation.launch.py \
  localization_map:=/absolute/path/localization.yaml \
  navigation_map:=/absolute/path/navigation.yaml \
  use_sim_time:=false
```

The launch owns:

- `localization_map_server`, publishing `/map_localization`, and AMCL subscribed
  to that map. The localization lifecycle manager activates both.
- `navigation_map_server`, publishing `/map_navigation`. The navigation lifecycle
  manager activates it along with the navigation nodes. Both costmaps' enabled
  static layers use this topic; local static-layer enablement is unchanged.

Both maps must be registered in the same `map` frame. Image editing must preserve
the corresponding resolution, origin and yaw. Paths may be relative to the launch
working directory; they are resolved to absolute paths before starting nodes.
`namespace:=robot` scopes topics/nodes under `/robot`.

`map:=/path/shared.yaml` remains a shorthand for the same YAML in both servers.
`navigation_map` defaults to `localization_map`, and `localization_map` falls back
to `map`. With `use_localization:=false`, AMCL/localization server are not started;
a supplied `navigation_map` still starts the navigation map server. If no map
paths are supplied in external-localization mode, the launch retains the map
topics in your params file (default `/map`) and starts no map servers.

When using the two built-in servers, RViz/the dashboard's ROS-topic display should
select `/map_navigation` or `/map_localization`; this launch no longer publishes
the old `/map` alias. Dashboard PNG display is independent of ROS map selection.

After updating this package:

```bash
cd /home/orchid/ros2_ws
colcon build --packages-select orchid_nav2_config
source install/setup.bash
```

Stop the previous navigation launch before starting the revised one. No live
stack is restarted automatically by installing this package.
