cmake_minimum_required(VERSION 3.8)
project(schunk_egk50_description)

find_package(ament_cmake REQUIRED)

# Install URDF/Xacro files
install(DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
