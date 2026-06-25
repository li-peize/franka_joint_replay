// Copyright (c) 2026 Peize Li. Apache-2.0.
#include <franka_joint_replay/joint_impedance_trajectory_controller.h>

#include <algorithm>
#include <cmath>

#include <franka/robot_state.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>

namespace franka_joint_replay {

bool JointImpedanceTrajectoryController::init(hardware_interface::RobotHW* robot_hw,
                                              ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointImpedanceTrajectoryController: missing parameter 'arm_id'.");
    return false;
  }
  if (!node_handle.getParam("joint_names", joint_names_) || joint_names_.size() != 7) {
    ROS_ERROR("JointImpedanceTrajectoryController: 'joint_names' must list exactly 7 joints.");
    return false;
  }

  std::vector<double> k_gains, d_gains;
  if (!node_handle.getParam("k_gains", k_gains) || k_gains.size() != 7) {
    ROS_ERROR("JointImpedanceTrajectoryController: 'k_gains' must have 7 entries.");
    return false;
  }
  if (!node_handle.getParam("d_gains", d_gains) || d_gains.size() != 7) {
    ROS_ERROR("JointImpedanceTrajectoryController: 'd_gains' must have 7 entries.");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    k_gains_[i] = k_gains[i];
    d_gains_[i] = d_gains[i];
  }
  node_handle.param("coriolis_factor", coriolis_factor_, 1.0);
  node_handle.param("delta_tau_max", delta_tau_max_, 1.0);

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("JointImpedanceTrajectoryController: no FrankaStateInterface.");
    return false;
  }
  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR("JointImpedanceTrajectoryController: no FrankaModelInterface.");
    return false;
  }
  auto* effort_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_interface == nullptr) {
    ROS_ERROR("JointImpedanceTrajectoryController: no EffortJointInterface.");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
    for (size_t i = 0; i < 7; ++i) {
      joint_handles_.push_back(effort_interface->getHandle(joint_names_[i]));
    }
  } catch (const hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("JointImpedanceTrajectoryController: handle error: " << ex.what());
    return false;
  }

  command_sub_ = node_handle.subscribe(
      "command", 1, &JointImpedanceTrajectoryController::commandCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  desired_pub_ = std::make_unique<realtime_tools::RealtimePublisher<sensor_msgs::JointState>>(
      node_handle, "desired_joint_states", 1);
  desired_pub_->lock();
  desired_pub_->msg_.name = joint_names_;
  desired_pub_->msg_.position.resize(7);
  desired_pub_->msg_.velocity.resize(7);
  desired_pub_->unlock();

  return true;
}

void JointImpedanceTrajectoryController::starting(const ros::Time& /*time*/) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  for (size_t i = 0; i < 7; ++i) {
    q_hold_[i] = robot_state.q[i];
    dq_filtered_[i] = 0.0;
  }
  trajectory_buffer_.initRT(std::shared_ptr<Trajectory>());  // hold until a command arrives
  active_trajectory_ = nullptr;
  segment_index_ = 0;
}

void JointImpedanceTrajectoryController::update(const ros::Time& time,
                                                const ros::Duration& /*period*/) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();

  for (size_t i = 0; i < 7; ++i) {
    dq_filtered_[i] = 0.99 * robot_state.dq[i] + 0.01 * dq_filtered_[i];
  }

  // Default: hold the configuration captured in starting().
  std::array<double, 7> q_d = q_hold_;
  std::array<double, 7> dq_d{};

  std::shared_ptr<Trajectory> traj;
  if (auto* slot = trajectory_buffer_.readFromRT()) {
    traj = *slot;  // copy shared_ptr: keep the trajectory alive for this tick
  }
  if (traj && !traj->empty()) {
    if (traj.get() != active_trajectory_) {
      active_trajectory_ = traj.get();
      segment_index_ = 0;
    }
    const Trajectory& segments = *traj;
    const double now = time.toSec();
    while (segment_index_ + 1 < segments.size() && now > segments[segment_index_].endTime()) {
      ++segment_index_;
    }
    SegmentState desired;
    segments[segment_index_].sample(now, desired);  // clamps to endpoints outside the segment
    if (desired.position.size() == 7) {
      for (size_t i = 0; i < 7; ++i) {
        q_d[i] = desired.position[i];
      }
      if (desired.velocity.size() == 7) {
        for (size_t i = 0; i < 7; ++i) {
          dq_d[i] = desired.velocity[i];
        }
      }
    }
  }

  std::array<double, 7> tau_d;
  for (size_t i = 0; i < 7; ++i) {
    tau_d[i] = coriolis_factor_ * coriolis[i] + k_gains_[i] * (q_d[i] - robot_state.q[i]) +
               d_gains_[i] * (dq_d[i] - dq_filtered_[i]);
  }
  std::array<double, 7> tau_cmd = saturateTorqueRate(tau_d, robot_state.tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_cmd[i]);
  }

  if (desired_pub_ && desired_pub_->trylock()) {
    desired_pub_->msg_.header.stamp = time;
    for (size_t i = 0; i < 7; ++i) {
      desired_pub_->msg_.position[i] = q_d[i];
      desired_pub_->msg_.velocity[i] = dq_d[i];
    }
    desired_pub_->unlockAndPublish();
  }
}

std::array<double, 7> JointImpedanceTrajectoryController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; ++i) {
    const double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void JointImpedanceTrajectoryController::commandCallback(
    const trajectory_msgs::JointTrajectory::ConstPtr& msg) {
  const std::size_t n = msg->points.size();
  if (n == 0) {
    ROS_WARN("JointImpedanceTrajectoryController: ignoring empty trajectory.");
    return;
  }

  // Map each controller joint to its index in the message.
  std::array<int, 7> idx{};
  for (size_t i = 0; i < 7; ++i) {
    auto it = std::find(msg->joint_names.begin(), msg->joint_names.end(), joint_names_[i]);
    if (it == msg->joint_names.end()) {
      ROS_ERROR_STREAM("JointImpedanceTrajectoryController: command is missing joint '"
                       << joint_names_[i] << "'.");
      return;
    }
    idx[i] = static_cast<int>(std::distance(msg->joint_names.begin(), it));
  }

  const double start =
      (msg->header.stamp.toSec() == 0.0) ? ros::Time::now().toSec() : msg->header.stamp.toSec();

  std::vector<SegmentState> states(n);
  std::vector<double> times(n);
  for (size_t k = 0; k < n; ++k) {
    const auto& pt = msg->points[k];
    if (pt.positions.size() != msg->joint_names.size()) {
      ROS_ERROR("JointImpedanceTrajectoryController: point %zu positions size mismatch.", k);
      return;
    }
    times[k] = start + pt.time_from_start.toSec();
    states[k].position.resize(7);
    for (size_t i = 0; i < 7; ++i) {
      states[k].position[i] = pt.positions[idx[i]];
    }
    if (pt.velocities.size() == pt.positions.size()) {
      states[k].velocity.resize(7);
      for (size_t i = 0; i < 7; ++i) {
        states[k].velocity[i] = pt.velocities[idx[i]];
      }
    }
  }

  auto traj = std::make_shared<Trajectory>();
  if (n == 1) {
    traj->emplace_back(times[0], states[0], times[0], states[0]);
  } else {
    traj->reserve(n - 1);
    for (size_t k = 0; k + 1 < n; ++k) {
      const double t_end = std::max(times[k + 1], times[k]);  // guard non-monotonic input
      traj->emplace_back(times[k], states[k], t_end, states[k + 1]);
    }
  }
  trajectory_buffer_.writeFromNonRT(traj);
  ROS_INFO("JointImpedanceTrajectoryController: loaded %zu-point trajectory spanning %.2f s.",
           n, times.back() - times.front());
}

}  // namespace franka_joint_replay

PLUGINLIB_EXPORT_CLASS(franka_joint_replay::JointImpedanceTrajectoryController,
                       controller_interface::ControllerBase)
