cmake_minimum_required(VERSION 3.4)
project(franka_joint_replay)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  hardware_interface
  pluginlib
  realtime_tools
  roscpp
  trajectory_msgs
  sensor_msgs
  joint_trajectory_controller
  franka_hw
  franka_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Franka REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    controller_interface
    hardware_interface
    pluginlib
    realtime_tools
    roscpp
    trajectory_msgs
    sensor_msgs
    joint_trajectory_controller
    franka_hw
    franka_msgs
  DEPENDS Franka
)

add_library(${PROJECT_NAME}
  src/joint_impedance_trajectory_controller.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Franka_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${Franka_LIBRARIES}
)

## Install
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(FILES franka_joint_replay_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
catkin_install_python(PROGRAMS scripts/replay_lerobot_episode.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
