#!/usr/bin/env python3
"""Replay a LeRobot episode's 7-joint trajectory (+ gripper) on the Franka.

Reads the LeRobot v2.1 dataset parquet DIRECTLY (pandas + pyarrow; no `lerobot`
dependency, so it runs in ROS Noetic's Python), extracts the recorded arm-joint
trajectory from ``observation.state``, and publishes ONE
``trajectory_msgs/JointTrajectory`` to the joint-impedance controller. Because
the whole episode is known up front, the controller interpolates the low-rate
(~20 Hz) waypoints to its 1 kHz loop with spline segments and no lag.

The trajectory is prefixed with the robot's CURRENT joint configuration at
``time_from_start = 0`` and the episode is offset by ``approach_time``, so the
controller ramps smoothly from wherever the arm is to the episode start (no
jump). The gripper is replayed by thresholding the recorded width and issuing
``franka_gripper`` Move/Grasp goals at the open/close transitions.
"""

import datetime
import glob
import json
import os
import sys

import actionlib
import numpy as np
import pandas as pd
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_gripper.msg import (
    GraspAction,
    GraspEpsilon,
    GraspGoal,
    MoveAction,
    MoveGoal,
)

FRANKA_JOINT_STATES = "/franka_state_controller/joint_states"


def _load_state_names(dataset):
    with open(os.path.join(dataset, "meta", "info.json"), encoding="utf-8") as handle:
        info = json.load(handle)
    return list(info["features"]["observation.state"]["names"])


def _episode_parquet(dataset, episode):
    candidate = os.path.join(
        dataset, "data", "chunk-%03d" % (episode // 1000), "episode_%06d.parquet" % episode)
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(dataset, "data", "*", "episode_%06d.parquet" % episode))
    if not matches:
        raise FileNotFoundError("No parquet for episode %d under %s/data" % (episode, dataset))
    return matches[0]


def _read_episode(dataset, episode):
    """Return (timestamps[N], state[N, D], state_names)."""
    names = _load_state_names(dataset)
    frame = pd.read_parquet(_episode_parquet(dataset, episode))
    state = np.stack([np.asarray(row, dtype=np.float64) for row in frame["observation.state"]])
    if "timestamp" in frame:
        times = np.asarray(frame["timestamp"], dtype=np.float64)
    else:  # fall back to a uniform grid from frame_index + fps
        with open(os.path.join(dataset, "meta", "info.json"), encoding="utf-8") as handle:
            fps = float(json.load(handle).get("fps", 20.0))
        times = np.arange(len(state), dtype=np.float64) / fps
    return times - times[0], state, names


def _joint_indices(names):
    idx = [names.index("joint%d" % j) for j in range(1, 8)]
    return idx


def _current_q(joint_names, timeout=10.0):
    msg = rospy.wait_for_message(FRANKA_JOINT_STATES, JointState, timeout=timeout)
    name_to_pos = dict(zip(msg.name, msg.position))
    return np.array([name_to_pos[j] for j in joint_names], dtype=np.float64)


def _finite_diff_velocity(times, joints):
    """Central-difference joint velocity; zero at the endpoints."""
    vel = np.zeros_like(joints)
    if len(times) >= 3:
        dt = times[2:] - times[:-2]
        dt[dt == 0.0] = 1e-6
        vel[1:-1] = (joints[2:] - joints[:-2]) / dt[:, None]
    return vel


def _build_trajectory(joint_names, current_q, times, joints, approach_time, start_delay,
                      with_velocity, approach_rate):
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(start_delay)
    traj.joint_names = list(joint_names)
    ndof = len(joint_names)

    vel = _finite_diff_velocity(times, joints) if with_velocity else None

    # Approach ramp: a dense linear ramp from the current pose to the episode
    # start over approach_time, so EVERY mode eases in (a single far point would
    # make the zero-order-hold mode step the whole gap at once). Covers
    # [0, approach_time); the episode then continues from approach_time.
    n_ramp = max(2, int(approach_time * approach_rate))
    for j in range(n_ramp):
        frac = j / float(n_ramp)
        point = JointTrajectoryPoint()
        point.positions = ((1.0 - frac) * current_q + frac * joints[0]).tolist()
        if with_velocity:
            point.velocities = [0.0] * ndof
        point.time_from_start = rospy.Duration.from_sec(approach_time * frac)
        traj.points.append(point)

    # Episode, offset by approach_time.
    for k in range(len(times)):
        point = JointTrajectoryPoint()
        point.positions = joints[k].tolist()
        if with_velocity:
            point.velocities = vel[k].tolist()
        point.time_from_start = rospy.Duration.from_sec(approach_time + float(times[k]))
        traj.points.append(point)
    return traj


def _gripper_events(times, width, open_thr, close_thr, max_width):
    """List of (episode_time, 'open'|'close') at width transitions (hysteresis)."""
    events = []
    state = None
    for k in range(len(times)):
        w = float(width[k])
        if w >= open_thr:
            new = "open"
        elif w <= close_thr:
            new = "close"
        else:
            new = state  # inside hysteresis band: keep current
        if new is not None and new != state:
            events.append((float(times[k]), new))
            state = new
    return events


class GripperClients(object):
    def __init__(self, open_width, open_speed, close_width, close_speed, close_force,
                 eps_inner, eps_outer):
        self._open_width = open_width
        self._open_speed = open_speed
        self._close_width = close_width
        self._close_speed = close_speed
        self._close_force = close_force
        self._eps_inner = eps_inner
        self._eps_outer = eps_outer
        self._move = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self._grasp = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)

    def open(self):
        goal = MoveGoal(width=self._open_width, speed=self._open_speed)
        self._move.send_goal(goal)
        rospy.loginfo("Gripper open (Move width=%.3f)", self._open_width)

    def close(self):
        goal = GraspGoal(width=self._close_width, speed=self._close_speed, force=self._close_force,
                         epsilon=GraspEpsilon(inner=self._eps_inner, outer=self._eps_outer))
        self._grasp.send_goal(goal)
        rospy.loginfo("Gripper close (Grasp force=%.1f)", self._close_force)


def _list_episodes(dataset, limit):
    """Return recent episodes (newest first) as dicts with index/task/length/
    duration/mtime, read from meta/episodes.jsonl + the parquet files.
    """
    fps = 20.0
    try:
        with open(os.path.join(dataset, "meta", "info.json"), encoding="utf-8") as handle:
            fps = float(json.load(handle).get("fps", 20.0))
    except Exception:  # noqa: BLE001 - best-effort metadata
        pass

    meta = {}
    episodes_jsonl = os.path.join(dataset, "meta", "episodes.jsonl")
    if os.path.exists(episodes_jsonl):
        with open(episodes_jsonl, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = d.get("episode_index")
                if idx is None:
                    continue
                tasks = d.get("tasks") or []
                meta[int(idx)] = {"task": tasks[0] if tasks else "", "length": d.get("length")}

    episodes = []
    for parquet in glob.glob(os.path.join(dataset, "data", "*", "episode_*.parquet")):
        base = os.path.basename(parquet)
        try:
            idx = int(base[len("episode_"):-len(".parquet")])
        except ValueError:
            continue
        info = meta.get(idx, {})
        length = info.get("length")
        episodes.append({
            "index": idx,
            "task": info.get("task", ""),
            "length": length,
            "duration": (length / fps) if length else None,
            "mtime": os.path.getmtime(parquet),
        })
    episodes.sort(key=lambda e: e["mtime"], reverse=True)
    return episodes[:limit]


def _choose_episode(episodes, requested):
    """Print the recent episodes and return the chosen episode_index.

    If ``requested`` >= 0 it is honored (non-interactive). Otherwise, with a
    TTY the user is prompted (Enter = most recent); without a TTY (e.g. under
    roslaunch) the most recent is auto-selected.
    """
    print("\nRecent replayable sequences in this dataset:")
    print("  %3s  %7s  %7s  %7s  %-16s  %s"
          % ("sel", "episode", "frames", "dur(s)", "recorded", "task"))
    for i, e in enumerate(episodes):
        ts = datetime.datetime.fromtimestamp(e["mtime"]).strftime("%Y-%m-%d %H:%M")
        frames = str(e["length"]) if e["length"] is not None else "?"
        dur = ("%.1f" % e["duration"]) if e["duration"] is not None else "?"
        print("  %3d) %7d  %7s  %7s  %-16s  %s" % (i, e["index"], frames, dur, ts, e["task"]))

    if requested is not None and requested >= 0:
        print("Episode %d requested via param; replaying it." % requested)
        return requested

    most_recent = episodes[0]["index"]
    if not sys.stdin.isatty():
        rospy.loginfo("Non-interactive (no TTY): auto-selecting most recent episode %d. "
                      "Run via `rosrun` in a terminal to choose, or pass _episode:=N.",
                      most_recent)
        return most_recent

    while True:
        try:
            raw = input("\nSelect [0-%d], Enter = most recent (episode %d): "
                        % (len(episodes) - 1, most_recent)).strip()
        except EOFError:
            return most_recent
        if raw == "":
            return most_recent
        try:
            sel = int(raw)
        except ValueError:
            print("  please enter a number.")
            continue
        if 0 <= sel < len(episodes):
            return episodes[sel]["index"]
        print("  out of range.")


def main():
    rospy.init_node("replay_lerobot_episode")

    dataset = rospy.get_param("~dataset", "")
    if not dataset:
        rospy.logerr("~dataset is not set; nothing to replay.")
        return
    requested = int(rospy.get_param("~episode", -1))  # -1 = ask / auto most recent
    max_list = int(rospy.get_param("~max_list", 20))
    arm_id = rospy.get_param("~arm_id", "panda")
    approach_time = float(rospy.get_param("~approach_time", 5.0))
    approach_rate = float(rospy.get_param("~approach_rate", 20.0))
    start_delay = float(rospy.get_param("~start_delay", 1.0))
    with_velocity = bool(rospy.get_param("~with_velocity", True))
    command_topic = rospy.get_param(
        "~command_topic", "/joint_impedance_replay_controller/command")
    replay_gripper = bool(rospy.get_param("~replay_gripper", True))
    gripper_max_width = float(rospy.get_param("~gripper_max_width", 0.08))
    open_thr = float(rospy.get_param("~gripper_open_threshold", 0.6 * gripper_max_width))
    close_thr = float(rospy.get_param("~gripper_close_threshold", 0.4 * gripper_max_width))

    joint_names = ["%s_joint%d" % (arm_id, j) for j in range(1, 8)]

    episodes = _list_episodes(dataset, max_list)
    if not episodes:
        rospy.logerr("No episodes found under %s/data; nothing to replay.", dataset)
        return
    episode = _choose_episode(episodes, requested)

    times, state, names = _read_episode(dataset, episode)
    joints = state[:, _joint_indices(names)]
    rospy.loginfo("Episode %d: %d frames, %.2f s, %d-DoF joints.",
                  episode, len(times), float(times[-1]), joints.shape[1])

    current_q = _current_q(joint_names)
    traj = _build_trajectory(joint_names, current_q, times, joints, approach_time, start_delay,
                             with_velocity, approach_rate)

    pub = rospy.Publisher(command_topic, JointTrajectory, queue_size=1, latch=True)
    # Wait for the controller to subscribe so the (timed) trajectory is not missed.
    deadline = rospy.Time.now() + rospy.Duration.from_sec(5.0)
    while pub.get_num_connections() == 0 and rospy.Time.now() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.05)
    if pub.get_num_connections() == 0:
        rospy.logwarn("No subscriber on %s; publishing anyway (latched).", command_topic)
    # Re-stamp now that we are about to publish.
    traj.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(start_delay)
    traj_start = traj.header.stamp
    pub.publish(traj)
    rospy.loginfo("Published trajectory (%d points) to %s; starting in %.1f s + %.1f s approach.",
                  len(traj.points), command_topic, start_delay, approach_time)

    # Gripper playback, timed against the same start + approach offset.
    if replay_gripper:
        gripper_idx = names.index("gripper") if "gripper" in names else None
        if gripper_idx is None:
            rospy.logwarn("No 'gripper' in observation.state; skipping gripper replay.")
        else:
            grip = GripperClients(
                open_width=float(rospy.get_param("~open_width", 0.08)),
                open_speed=float(rospy.get_param("~open_speed", 0.1)),
                close_width=float(rospy.get_param("~close_width", 0.0)),
                close_speed=float(rospy.get_param("~close_speed", 0.1)),
                close_force=float(rospy.get_param("~close_force", 30.0)),
                eps_inner=float(rospy.get_param("~close_epsilon_inner", 0.005)),
                eps_outer=float(rospy.get_param("~close_epsilon_outer", 0.08)))
            events = _gripper_events(times, state[:, gripper_idx], open_thr, close_thr,
                                     gripper_max_width)
            for episode_t, action in events:
                exec_time = traj_start + rospy.Duration.from_sec(approach_time + episode_t)
                _sleep_until(exec_time)
                if rospy.is_shutdown():
                    return
                grip.open() if action == "open" else grip.close()

    # Stay alive until the motion finishes (latched trajectory keeps holding).
    end_time = traj_start + rospy.Duration.from_sec(approach_time + float(times[-1]) + 1.0)
    _sleep_until(end_time)
    rospy.loginfo("Replay complete; controller holds the final pose.")


def _sleep_until(stamp):
    while not rospy.is_shutdown() and rospy.Time.now() < stamp:
        rospy.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
