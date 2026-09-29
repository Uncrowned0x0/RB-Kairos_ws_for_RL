cmake_minimum_required(VERSION 3.8)
project(rbkairos_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes urdf robots config scripts
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts as executables
install(
  PROGRAMS
    scripts/twist_relay.py
    scripts/kairos_teleop_keyboard.py
    scripts/kairos_teleop_keyboard
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
