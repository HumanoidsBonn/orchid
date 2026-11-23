"""Launch Orchid-owned HSR-C MoveIt configuration."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                choices=["true", "false"],
                description="Use simulated time.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([ThisLaunchFileDir(), "/demo.py"]),
                launch_arguments={
                    "description_package": "hsrc_description",
                    "description_file": "hsrc1s.urdf.xacro",
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }.items(),
            ),
        ]
    )
