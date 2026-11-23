cmake_minimum_required(VERSION 3.8)
project(orchid_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/odom_joint_states_publisher.py
  	   scripts/depth_to_filtered_pcd.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
