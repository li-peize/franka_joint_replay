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

#include <ruckig/ruckig.hpp>
#include <trajectory_interface/quintic_spline_segment.h>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

namespace franka_joint_replay {

// A joint-impedance controller that follows a trajectory_msgs/JointTrajectory.
//
// Torque law (from franka_example_controllers/JointImpedanceExampleController):
//   tau = coriolis_factor*coriolis + k*(q_d - q) + d*(dq_d - dq_filtered),
// torque-rate saturated. The desired (q_d, dq_d) are produced at the 1 kHz loop
// from the low-rate waypoints by one of four selectable bridging modes
// (~interpolation_mode): none (zero-order hold), lowpass, spline (quintic), or
// ruckig (online jerk-limited OTG). With no command yet it holds the joint
// configuration captured in starting().
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
  enum class Mode { kNone, kLowPass, kSpline, kRuckig };

  using Segment = trajectory_interface::QuinticSplineSegment<double>;
  using SegmentState = Segment::State;  // PosVelAccState<double>

  // A parsed command: raw waypoints (for none/lowpass/ruckig) plus pre-built
  // spline segments (for spline mode), handed to the RT loop as one unit.
  struct CommandTrajectory {
    std::vector<double> times;                      // absolute sec, size N
    std::vector<std::array<double, 7>> positions;   // size N
    std::vector<std::array<double, 7>> velocities;  // size N, or empty
    bool has_velocity{false};
    std::vector<Segment> segments;                  // size N-1 (spline mode)
  };

  std::array<double, 7> saturateTorqueRate(const std::array<double, 7>& tau_d_calculated,
                                           const std::array<double, 7>& tau_J_d);
  void commandCallback(const trajectory_msgs::JointTrajectory::ConstPtr& msg);
  void seedFromCurrent(const std::array<double, 7>& q);  // re-anchor mode state on (re)load

  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  std::vector<std::string> joint_names_;
  std::array<double, 7> k_gains_{};
  std::array<double, 7> d_gains_{};
  double coriolis_factor_{1.0};
  double delta_tau_max_{1.0};

  // --- bridging mode + per-mode params ---
  Mode mode_{Mode::kSpline};
  double lp_alpha_{1.0};                       // low-pass coefficient (per tick)
  std::array<double, 7> ruckig_max_velocity_{};
  std::array<double, 7> ruckig_max_acceleration_{};
  std::array<double, 7> ruckig_max_jerk_{};

  std::array<double, 7> dq_filtered_{};
  std::array<double, 7> q_hold_{};

  // --- RT-only state ---
  std::array<double, 7> q_lp_{};      // low-pass filter state
  std::array<double, 7> q_d_prev_{};  // previous desired (LP velocity FF)
  SegmentState scratch_state_;        // reused spline sample target (pre-sized; no RT alloc)
  ruckig::Ruckig<7> otg_{0.001};      // OTG at the 1 kHz control period
  ruckig::InputParameter<7> ruckig_input_;
  ruckig::OutputParameter<7> ruckig_output_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<CommandTrajectory>> trajectory_buffer_;
  const CommandTrajectory* active_trajectory_{nullptr};  // RT-only: identity of loaded traj
  std::size_t waypoint_index_{0};                        // RT-only: cursor into waypoints

  ros::Subscriber command_sub_;
  std::unique_ptr<realtime_tools::RealtimePublisher<sensor_msgs::JointState>> desired_pub_;
};

}  // namespace franka_joint_replay
