cmake_minimum_required(VERSION 3.8)
project(kairos_bringup)

find_package(ament_cmake REQUIRED)

# Install launch files so they are found by FindPackageShare("kairos_bringup")
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Install Python scripts as executables
install(
  PROGRAMS
    scripts/kairos_teleop_keyboard.py
    scripts/kairos_teleop_keyboard
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
