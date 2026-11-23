#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def declare_arguments():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Automatically startup the nav2 stack'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_localization', default_value='true',
            description='Launch the localization map server and AMCL'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'map',
            default_value='',
            description='Legacy shorthand: YAML used for both maps unless overridden'))
    declared_arguments.extend([
        DeclareLaunchArgument('localization_map', default_value='',
                              description='Localization map YAML; defaults to map'),
        DeclareLaunchArgument('navigation_map', default_value='',
                              description='Navigation map YAML; defaults to localization_map/map'),
    ])

    declared_arguments.append(
        DeclareLaunchArgument(
            'params_file',
                default_value=os.path.join(get_package_share_directory('orchid_nav2_config'),
                                       'config', 'nav2_params.yaml'),
            description='Full path to the ROS2 parameters file to use'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'default_bt_xml_filename',
            default_value=os.path.join(
                get_package_share_directory('nav2_bt_navigator'),
                'behavior_trees', 'navigate_w_replanning_and_recovery.xml'),
            description='Full path to the behavior tree xml file to use'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'map_subscribe_transient_local', default_value='true',
            description='Whether to set the map subscriber QoS to transient local'))

    declared_arguments.append(
        DeclareLaunchArgument(
            'odom_topic', default_value='omni_base_controller/wheel_odom',
            description='Odometry topic name'))

    return declared_arguments


def _launch_setup(context):
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_localization = LaunchConfiguration('use_localization')
    localization_enabled = IfCondition(use_localization).evaluate(context)
    legacy_map = LaunchConfiguration('map').perform(context).strip()
    localization_map = LaunchConfiguration('localization_map').perform(context).strip() or legacy_map
    navigation_map = LaunchConfiguration('navigation_map').perform(context).strip() or localization_map
    if localization_enabled and not localization_map:
        raise RuntimeError('Pass localization_map:=... and navigation_map:=... (or map:=... for a shared map)')
    for label, path in [('localization_map',localization_map if localization_enabled else ''),('navigation_map',navigation_map)]:
        if path and not os.path.isfile(os.path.expanduser(path)):
            raise RuntimeError(f'{label} YAML does not exist: {path}')
    localization_map = os.path.abspath(os.path.expanduser(localization_map)) if localization_map else ''
    navigation_map = os.path.abspath(os.path.expanduser(navigation_map)) if navigation_map else ''
    ns = namespace.perform(context).strip('/')
    topic_prefix = '/' + ns if ns else ''
    localization_topic = topic_prefix + '/map_localization'
    navigation_topic = topic_prefix + '/map_navigation'
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    lifecycle_nodes = ['controller_server',
                       'planner_server',
                       'behavior_server',
                       'bt_navigator',
                       'waypoint_follower']
    if navigation_map:
        lifecycle_nodes.insert(0,'navigation_map_server')

    tf_remappings = [('/tf', 'tf'),
                     ('/tf_static', 'tf_static')]
    # velocity_remappings = [('cmd_vel', '/cmd_vel_nav')]
    velocity_remappings = [('cmd_vel', '/omni_base_controller/cmd_vel')]
    remappings = tf_remappings + velocity_remappings

    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_bt_xml_filename': default_bt_xml_filename,
        'autostart': autostart,
        'map_subscribe_transient_local': map_subscribe_transient_local,
        'odom_topic': LaunchConfiguration('odom_topic')}
    if navigation_map:
        for costmap in ['global_costmap','local_costmap']:
            param_substitutions[f'{costmap}.{costmap}.ros__parameters.static_layer.map_topic'] = navigation_topic

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    env = SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    map_nodes = []
    if localization_enabled:
        map_nodes.extend([
            Node(package='nav2_map_server', executable='map_server',
                 name='localization_map_server',namespace=namespace,output='screen',
                 parameters=[{'yaml_filename':localization_map,'topic_name':localization_topic,
                              'frame_id':'map','use_sim_time':use_sim_time}]),
            Node(package='nav2_amcl',executable='amcl',name='amcl',namespace=namespace,output='screen',
                 parameters=[configured_params,{'map_topic':localization_topic}],
                 remappings=tf_remappings+[('map',localization_topic)]),
            Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',
                 name='lifecycle_manager_localization',namespace=namespace,output='screen',
                 parameters=[{'use_sim_time':use_sim_time,'autostart':autostart,
                              'node_names':['localization_map_server','amcl']}]),
        ])
    if navigation_map:
        map_nodes.append(Node(package='nav2_map_server',executable='map_server',
                              name='navigation_map_server',namespace=namespace,output='screen',
                              parameters=[{'yaml_filename':navigation_map,'topic_name':navigation_topic,
                                           'frame_id':'map','use_sim_time':use_sim_time}]))

    controller_server_node = Node(
        namespace=namespace,
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    planner_server_node = Node(
        namespace=namespace,
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[configured_params],
        remappings=tf_remappings)

    behavior_server_node = Node(
        namespace=namespace,
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[configured_params],
        remappings=remappings)

    bt_navigator_node = Node(
        namespace=namespace,
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params],
        remappings=tf_remappings)

    waypoint_follower_node = Node(
        namespace=namespace,
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[configured_params],
        remappings=tf_remappings)

    lifecycle_manager_node = Node(
        namespace=namespace,
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': lifecycle_nodes}])

    nodes = [
        env,
        *map_nodes,
        controller_server_node,
        planner_server_node,
        behavior_server_node,
        bt_navigator_node,
        waypoint_follower_node,
        lifecycle_manager_node
    ]

    return nodes


def generate_launch_description():
    return LaunchDescription(declare_arguments() + [OpaqueFunction(function=_launch_setup)])
