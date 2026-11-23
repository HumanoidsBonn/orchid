cmake_minimum_required(VERSION 3.8)
project(orchid_mapping_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  maps
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(PROGRAMS
  scripts/unknown_to_free.py
  scripts/save_map.sh
  scripts/serialize_slam_toolbox_map.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
