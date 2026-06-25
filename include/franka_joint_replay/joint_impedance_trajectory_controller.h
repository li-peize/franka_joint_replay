// Copyright (c) 2026 Peize Li. Apache-2.0.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>

#include <trajectory_interface/quintic_spline_segment.h>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace franka_joint_replay {

// A joint-impedance controller that follows a trajectory_msgs/JointTrajectory.
//
// Torque law (from franka_example_controllers/JointImpedanceExampleController):
//   tau = coriolis_factor*coriolis + k*(q_d - q) + d*(dq_d - dq_filtered),
// torque-rate saturated. The desired (q_d, dq_d) are sampled at the 1 kHz loop
// from quintic-spline segments built between the (low-rate) waypoints, reusing
// trajectory_interface::QuinticSplineSegment for the interpolation math. With
// no command yet it holds the joint configuration captured in starting().
class JointImpedanceTrajectoryController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaModelInterface,
          franka_hw::FrankaStateInterface,
          hardware_interface::EffortJointInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;

 private:
  using Segment = trajectory_interface::QuinticSplineSegment<double>;
  using SegmentState = Segment::State;        // PosVelAccState<double>
  using Trajectory = std::vector<Segment>;    // multi-joint segments (dim 7)

  // Clamp the per-tick change of the commanded torque (matches the franka
  // examples), relative to the last desired torque tau_J_d.
  std::array<double, 7> saturateTorqueRate(const std::array<double, 7>& tau_d_calculated,
                                           const std::array<double, 7>& tau_J_d);

  // Non-RT: parse a JointTrajectory into spline segments and hand it to the RT
  // loop via the realtime buffer.
  void commandCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg);

  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  std::vector<std::string> joint_names_;
  std::array<double, 7> k_gains_{};
  std::array<double, 7> d_gains_{};
  double coriolis_factor_{1.0};
  double delta_tau_max_{1.0};

  std::array<double, 7> dq_filtered_{};
  std::array<double, 7> q_hold_{};

  realtime_tools::RealtimeBuffer<std::shared_ptr<Trajectory>> trajectory_buffer_;
  const Trajectory* active_trajectory_{nullptr};  // RT-only: identity of loaded traj
  std::size_t segment_index_{0};                  // RT-only: cursor into segments

  ros::Subscriber command_sub_;
  std::unique_ptr<realtime_tools::RealtimePublisher<sensor_msgs::JointState>> desired_pub_;
};

}  // namespace franka_joint_replay
