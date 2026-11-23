cmake_minimum_required(VERSION 3.8)
project(orchid_nav2_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
