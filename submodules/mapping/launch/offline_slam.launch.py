"""Launch SLAM Toolbox with rosbag playback for offline map building."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("orchid_mapping_config")
    default_params = os.path.join(pkg_share, "config", "slam_params.yaml")

    bag_path = LaunchConfiguration("bag_path")
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_path", description="Path to a ROS2 bag directory."),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("params_file", default_value=default_params),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[params_file, {"use_sim_time": use_sim_time}],
            ),
            ExecuteProcess(
                cmd=["ros2", "bag", "play", bag_path, "--clock"],
                output="screen",
            ),
        ]
    )
