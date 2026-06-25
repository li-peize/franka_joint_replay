# franka_joint_replay

Replay a **LeRobot episode** (produced by `bag2lerobot`) on a Franka by
commanding the recorded **7 joint positions (+ gripper)** through a custom
**joint-impedance controller**.

Two parts:

1. **`JointImpedanceTrajectoryController`** (C++ pluginlib controller).
   Torque law from `franka_example_controllers/JointImpedanceExampleController`:
   `τ = coriolis_factor·coriolis + k_gains·(q_d−q) + d_gains·(dq_d−dq_filtered)`,
   torque-rate saturated. Unlike the example (which runs an internal Cartesian
   circle), this one **subscribes to a `trajectory_msgs/JointTrajectory`** on
   `~command` and samples `q_d, dq_d` at the 1 kHz loop from quintic-spline
   segments (reusing `trajectory_interface::QuinticSplineSegment`). Low-rate
   (~20–30 Hz) waypoints are therefore interpolated to 1 kHz **without a
   low-pass** — no velocity-proportional lag. It also publishes the desired
   joints on `~desired_joint_states` for plotting.

2. **`replay_lerobot_episode.py`** (ROS 1 Python node). Reads the LeRobot v2.1
   parquet **directly** (pandas + pyarrow — no `lerobot`/torch, so it runs in
   Noetic's Python), extracts the arm-joint trajectory from `observation.state`,
   prepends the robot's current pose, and publishes the whole episode as one
   `JointTrajectory` (with finite-difference velocities → cubic spline). The
   gripper is replayed by thresholding the recorded width and firing
   `franka_gripper` Move/Grasp at the open/close transitions.

Because the whole episode is known up front, sending it as one trajectory gives
**zero causal lag** — the controller always interpolates between known points.

## Why a new controller

The stock `joint_impedance_example_controller` generates its own trajectory and
exposes no command input, so it can't follow recorded joints. This controller
keeps its torque law but takes joints from a topic. (`franka_spacemouse_teleop`
deliberately ships no C++, so this lives in its own package.)

## Build

```bash
cd <ws> && catkin_make && source devel/setup.bash
# the replay node needs parquet support in the ROS Python:
pip install pyarrow pandas        # for the Noetic python3 interpreter
```

Confirm the plugin is found:
```bash
rospack plugins --attrib=plugin controller_interface | grep franka_joint_replay
```

## Run

First export an episode with `bag2lerobot` (e.g. to `/tmp/teleop_ds`).

**Gazebo (validate here first):**
```bash
roslaunch franka_joint_replay replay_gazebo.launch dataset:=/tmp/teleop_ds episode:=0
```
**Real robot:**
```bash
roslaunch franka_joint_replay replay.launch robot_ip:=<ip> dataset:=/tmp/teleop_ds episode:=0
```

Args: `episode`, `approach_time` (ramp from current pose to the episode start,
default 5 s), `with_velocity` (cubic vs linear interpolation), `replay_gripper`.

## Verify

- `rqt_plot` the controller's `~desired_joint_states` vs
  `/franka_state_controller/joint_states`: motion should be smooth (no 20 Hz
  stairs), start with **no jump** (approach ramp), and hold at the end.
- Tune `config/joint_impedance_replay.yaml` `k_gains`/`d_gains` (start soft).

## Safety / notes

- **Only one controller may claim the 7 joints** — don't run the
  Cartesian/teleop controller during replay.
- The `approach_time` prefix + current-pose point 0 prevent a startup jump;
  keep `approach_time` generous.
- Gripper replay is threshold-based (open/close), not continuous width.
- Gains default to the `joint_impedance_example_controller` values; lower them
  for the first hardware run.
