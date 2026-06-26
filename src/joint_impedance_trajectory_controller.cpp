// Copyright (c) 2026 Peize Li. Apache-2.0.
#include <franka_joint_replay/joint_impedance_trajectory_controller.h>

#include <algorithm>
#include <cmath>

#include <franka/robot_state.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>

namespace franka_joint_replay {

namespace {
// Panda joint kinematic limits (rad/s, rad/s^2, rad/s^3) used as Ruckig
// defaults; generous since recorded replay is slow.
const std::array<double, 7> kDefaultMaxVel{2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61};
const std::array<double, 7> kDefaultMaxAcc{15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0};
const std::array<double, 7> kDefaultMaxJerk{7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0};

void read7(ros::NodeHandle& nh, const std::string& name, const std::array<double, 7>& def,
           std::array<double, 7>& out) {
  std::vector<double> v;
  if (nh.getParam(name, v) && v.size() == 7) {
    for (size_t i = 0; i < 7; ++i) out[i] = v[i];
  } else {
    out = def;
  }
}
}  // namespace

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

  // --- bridging mode ---
  std::string mode_str;
  node_handle.param<std::string>("interpolation_mode", mode_str, "spline");
  if (mode_str == "none") {
    mode_ = Mode::kNone;
  } else if (mode_str == "lowpass") {
    mode_ = Mode::kLowPass;
  } else if (mode_str == "ruckig") {
    mode_ = Mode::kRuckig;
  } else {
    if (mode_str != "spline") {
      ROS_WARN_STREAM("JointImpedanceTrajectoryController: unknown interpolation_mode '"
                      << mode_str << "'; defaulting to 'spline'.");
    }
    mode_ = Mode::kSpline;
  }

  // Low-pass coefficient from a cutoff frequency, exact first-order at 1 kHz.
  double lowpass_cutoff_hz;
  node_handle.param("lowpass_cutoff_hz", lowpass_cutoff_hz, 10.0);
  const double control_dt = 0.001;
  lp_alpha_ = 1.0 - std::exp(-2.0 * M_PI * lowpass_cutoff_hz * control_dt);

  // Ruckig limits.
  read7(node_handle, "ruckig_max_velocity", kDefaultMaxVel, ruckig_max_velocity_);
  read7(node_handle, "ruckig_max_acceleration", kDefaultMaxAcc, ruckig_max_acceleration_);
  read7(node_handle, "ruckig_max_jerk", kDefaultMaxJerk, ruckig_max_jerk_);
  ruckig_input_.max_velocity = ruckig_max_velocity_;
  ruckig_input_.max_acceleration = ruckig_max_acceleration_;
  ruckig_input_.max_jerk = ruckig_max_jerk_;

  ROS_INFO_STREAM("JointImpedanceTrajectoryController: interpolation_mode=" << mode_str);

  // Pre-size the spline sample target so sample()'s resize is a no-op (no
  // heap allocation on the 1 kHz RT thread).
  scratch_state_.position.resize(7);
  scratch_state_.velocity.resize(7);
  scratch_state_.acceleration.resize(7);

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

void JointImpedanceTrajectoryController::seedFromCurrent(const std::array<double, 7>& q) {
  q_lp_ = q;
  q_d_prev_ = q;
  ruckig_input_.current_position = q;
  ruckig_input_.current_velocity = {};
  ruckig_input_.current_acceleration = {};
  ruckig_input_.target_position = q;
  ruckig_input_.target_velocity = {};
  ruckig_input_.target_acceleration = {};
}

void JointImpedanceTrajectoryController::starting(const ros::Time& /*time*/) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  for (size_t i = 0; i < 7; ++i) {
    q_hold_[i] = robot_state.q[i];
    dq_filtered_[i] = 0.0;
  }
  seedFromCurrent(q_hold_);
  trajectory_buffer_.initRT(std::shared_ptr<CommandTrajectory>());  // hold until a command arrives
  active_trajectory_ = nullptr;
  waypoint_index_ = 0;
}

void JointImpedanceTrajectoryController::update(const ros::Time& time,
                                                const ros::Duration& period) {
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis = model_handle_->getCoriolis();
  const double dt = period.toSec();

  for (size_t i = 0; i < 7; ++i) {
    dq_filtered_[i] = 0.99 * robot_state.dq[i] + 0.01 * dq_filtered_[i];
  }

  // Default: hold the configuration captured in starting().
  std::array<double, 7> q_d = q_hold_;
  std::array<double, 7> dq_d{};

  std::shared_ptr<CommandTrajectory> traj;
  if (auto* slot = trajectory_buffer_.readFromRT()) {
    traj = *slot;  // copy shared_ptr: keep the trajectory alive for this tick
  }

  if (traj && !traj->positions.empty()) {
    if (traj.get() != active_trajectory_) {
      active_trajectory_ = traj.get();
      waypoint_index_ = 0;
      seedFromCurrent(robot_state.q);  // re-anchor mode state to current pose
    }
    const double now = time.toSec();
    while (waypoint_index_ + 1 < traj->times.size() && now >= traj->times[waypoint_index_ + 1]) {
      ++waypoint_index_;
    }
    const std::size_t i_active = waypoint_index_;

    switch (mode_) {
      case Mode::kNone: {
        q_d = traj->positions[i_active];  // zero-order hold; dq_d stays 0
        break;
      }
      case Mode::kLowPass: {
        const auto& target = traj->positions[i_active];
        for (size_t i = 0; i < 7; ++i) {
          q_lp_[i] += lp_alpha_ * (target[i] - q_lp_[i]);
          q_d[i] = q_lp_[i];
          if (dt > 0.0) dq_d[i] = (q_d[i] - q_d_prev_[i]) / dt;
        }
        break;
      }
      case Mode::kSpline: {
        if (!traj->segments.empty()) {
          const std::size_t seg = std::min(i_active, traj->segments.size() - 1);
          traj->segments[seg].sample(now, scratch_state_);  // clamps to endpoints outside
          if (scratch_state_.position.size() == 7) {
            for (size_t i = 0; i < 7; ++i) q_d[i] = scratch_state_.position[i];
            if (scratch_state_.velocity.size() == 7) {
              for (size_t i = 0; i < 7; ++i) dq_d[i] = scratch_state_.velocity[i];
            }
          }
        } else {
          q_d = traj->positions[i_active];
        }
        break;
      }
      case Mode::kRuckig: {
        ruckig_input_.target_position = traj->positions[i_active];
        ruckig_input_.target_velocity =
            traj->has_velocity ? traj->velocities[i_active] : std::array<double, 7>{};
        ruckig_input_.target_acceleration = {};
        const auto result = otg_.update(ruckig_input_, ruckig_output_);
        if (result == ruckig::Result::Working || result == ruckig::Result::Finished) {
          q_d = ruckig_output_.new_position;
          dq_d = ruckig_output_.new_velocity;
          ruckig_output_.pass_to_input(ruckig_input_);
        } else {
          // Planner error: hold the last OTG state rather than jump.
          q_d = ruckig_input_.current_position;
          dq_d = ruckig_input_.current_velocity;
          ROS_WARN_THROTTLE(1.0, "JointImpedanceTrajectoryController: ruckig update failed (%d).",
                            static_cast<int>(result));
        }
        break;
      }
    }
  }

  q_d_prev_ = q_d;  // for the low-pass velocity estimate next tick

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

  auto traj = std::make_shared<CommandTrajectory>();
  traj->times.resize(n);
  traj->positions.resize(n);
  traj->velocities.resize(n);
  bool all_have_velocity = true;
  for (size_t k = 0; k < n; ++k) {
    const auto& pt = msg->points[k];
    if (pt.positions.size() != msg->joint_names.size()) {
      ROS_ERROR("JointImpedanceTrajectoryController: point %zu positions size mismatch.", k);
      return;
    }
    traj->times[k] = start + pt.time_from_start.toSec();
    for (size_t i = 0; i < 7; ++i) traj->positions[k][i] = pt.positions[idx[i]];
    if (pt.velocities.size() == pt.positions.size()) {
      for (size_t i = 0; i < 7; ++i) traj->velocities[k][i] = pt.velocities[idx[i]];
    } else {
      all_have_velocity = false;
    }
  }
  traj->has_velocity = all_have_velocity;
  if (!all_have_velocity) traj->velocities.clear();

  // Pre-build spline segments (used by spline mode; cheap, off the RT path).
  auto make_state = [&](std::size_t k) {
    SegmentState s;
    s.position.assign(traj->positions[k].begin(), traj->positions[k].end());
    if (traj->has_velocity) {
      s.velocity.assign(traj->velocities[k].begin(), traj->velocities[k].end());
    }
    return s;
  };
  if (n == 1) {
    traj->segments.emplace_back(traj->times[0], make_state(0), traj->times[0], make_state(0));
  } else {
    traj->segments.reserve(n - 1);
    for (size_t k = 0; k + 1 < n; ++k) {
      const double t_end = std::max(traj->times[k + 1], traj->times[k]);  // guard non-monotonic
      traj->segments.emplace_back(traj->times[k], make_state(k), t_end, make_state(k + 1));
    }
  }

  trajectory_buffer_.writeFromNonRT(traj);
  ROS_INFO("JointImpedanceTrajectoryController: loaded %zu-point trajectory spanning %.2f s.",
           n, traj->times.back() - traj->times.front());
}

}  // namespace franka_joint_replay

PLUGINLIB_EXPORT_CLASS(franka_joint_replay::JointImpedanceTrajectoryController,
                       controller_interface::ControllerBase)
