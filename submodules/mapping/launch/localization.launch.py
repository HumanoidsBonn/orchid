"""Launch Nav2 map server and AMCL localization for a saved Orchid map."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("orchid_mapping_config")
    default_map = os.path.join(pkg_share, "config", "slam_data", "my_map.yaml")
    default_amcl = os.path.join(pkg_share, "config", "amcl.yaml")

    map_file = LaunchConfiguration("map")
    use_sim_time = LaunchConfiguration("use_sim_time")
    amcl_params = LaunchConfiguration("amcl_params")

    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("amcl_params", default_value=default_amcl),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[{"yaml_filename": map_file, "use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=[amcl_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_localization",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "autostart": True,
                        "node_names": ["map_server", "amcl"],
                    }
                ],
            ),
        ]
    )
