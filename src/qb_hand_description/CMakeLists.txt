cmake_minimum_required(VERSION 3.8)
project(qb_hand_description VERSION 3.0.3 LANGUAGES CXX)

# Dependency Settings
find_package(ament_cmake REQUIRED)

# Installation
install(
  DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
