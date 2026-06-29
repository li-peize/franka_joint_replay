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
   `~command` and produces `q_d, dq_d` at the 1 kHz loop from the low-rate
   (~20–30 Hz) waypoints using one of **four selectable bridging modes**
   (`~interpolation_mode`, see below). It also publishes the desired joints on
   `~desired_joint_states` for plotting.

2. **`replay_lerobot_episode.py`** (ROS 1 Python node). On startup it **lists
   the recent replayable episodes** in the dataset (newest first, with task /
   frame count / duration / recorded time) and lets you **choose** one — an
   interactive prompt when run in a terminal, or auto-select-most-recent when
   there's no TTY (e.g. under `roslaunch`), or the exact `~episode` if given.
   It then reads the chosen episode's LeRobot v3.0 parquet **directly** (pandas
   + pyarrow — no `lerobot`/torch, so it runs in Noetic's Python), extracts the
   arm-joint trajectory from `observation.state`, prepends the robot's current
   pose, and publishes the whole episode as one `JointTrajectory` (with
   finite-difference velocities → cubic spline). The gripper is replayed by
   thresholding the recorded width and firing `franka_gripper` Move/Grasp at the
   open/close transitions.

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
# The replay node needs pandas + pyarrow in the SAME interpreter that runs
# rospy (the ROS Python — python3.8 on Noetic). Install there explicitly:
python3.8 -m pip install --user pandas pyarrow
```

> Gotcha: if you have another `python3` (e.g. a 3.10+/conda) shadowing
> python3.8 on PATH, the node may pick an interpreter that has pandas but not
> rospy (or vice-versa). It needs **one** interpreter with both. Verify with
> `python3.8 -c "import rospy, pandas, pyarrow"`.

Confirm the plugin is found:
```bash
rospack plugins --attrib=plugin controller_interface | grep franka_joint_replay
```

## Run

First export an episode with `bag2lerobot`. The `dataset` arg is optional — when
omitted it defaults to **`$LEROBOT_DATA_PATH`**, falling back to
`~/teleop_lerobot` with a warning if that env var is unset. Pass `dataset:=<dir>`
to override. (Examples below use an explicit dir.)

**One-shot (auto-selects the most recent episode, or pass `episode:=N`):**
```bash
# Gazebo (validate here first):
roslaunch franka_joint_replay replay_gazebo.launch dataset:=/tmp/teleop_ds
# Real robot:
roslaunch franka_joint_replay replay.launch robot_ip:=<ip> dataset:=/tmp/teleop_ds
```
Under `roslaunch` the node has no TTY, so it prints the list and auto-selects the
most recent (or the `episode:=N` you pass).

**Interactive menu (choose from a terminal):** bring up the controller without
the replay node, then run the replay node yourself via `rosrun` (which has a
terminal, so you get the prompt):
```bash
# terminal A — controller only:
roslaunch franka_joint_replay replay.launch robot_ip:=<ip> run_replay:=false
# terminal B — interactive episode picker:
rosrun franka_joint_replay replay_lerobot_episode.py _dataset:=/tmp/teleop_ds
```
It prints the recent episodes and prompts for a selection (Enter = most recent).

Args: `episode` (-1 = ask/auto), `run_replay` (false = controller only),
`approach_time` (ramp from current pose to the episode start, default 5 s),
`with_velocity` (cubic vs linear), `replay_gripper`, and `interpolation` (the
bridging mode, below).

## Interpolation modes (`interpolation:=`)

How the low-rate (~20–30 Hz) waypoints are bridged to the 1 kHz loop. Select per
launch, e.g. `roslaunch ... replay_gazebo.launch interpolation:=ruckig ...`:

| mode | what it does | expect |
|---|---|---|
| `none` | zero-order hold — stepped target fed directly | stair-stepped desired, torque buzz, lag (baseline) |
| `lowpass` | first-order low-pass (`lowpass_cutoff_hz`) | smooth but lagged / rounded corners |
| `spline` *(default)* | quintic-spline interpolation between waypoints | smooth, faithful, ~0 lag with velocity FF |
| `ruckig` | online jerk-limited OTG toward the active waypoint (`ruckig_max_*`) | smooth, dynamically feasible; may re-time fast segments |

All modes share one command interface (the whole episode as one
`JointTrajectory`); the mode only changes how the controller samples it at
1 kHz. The approach ramp eases in safely in every mode.

## Verify

- `rqt_plot` the controller's `~desired_joint_states` vs
  `/franka_state_controller/joint_states`: motion should be smooth (no 20 Hz
  stairs), start with **no jump** (approach ramp), and hold at the end.
- Tune `config/joint_impedance_replay.yaml` `k_gains`/`d_gains` (start soft).

## Safety / notes

- **Only one controller may claim the 7 joints** — don't run the
  Cartesian/teleop controller during replay.
- The `approach_time` dense ramp from the current pose prevents a startup jump
  in every mode (incl. `none`); keep `approach_time` generous.
- Gripper replay is threshold-based (open/close), not continuous width.
- Gains default to the `joint_impedance_example_controller` values; lower them
  for the first hardware run.
