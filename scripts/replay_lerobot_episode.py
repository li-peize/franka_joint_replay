#!/usr/bin/env python3
"""Replay a LeRobot episode's 7-joint trajectory (+ gripper) on the Franka.

Reads the LeRobot v3.0 dataset parquet DIRECTLY (pandas + pyarrow; no `lerobot`
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


def _load_action_names(dataset):
    with open(os.path.join(dataset, "meta", "info.json"), encoding="utf-8") as handle:
        info = json.load(handle)
    return list(info["features"]["action"]["names"])


def _dataset_fps(dataset, default=20.0):
    try:
        with open(os.path.join(dataset, "meta", "info.json"), encoding="utf-8") as handle:
            return float(json.load(handle).get("fps", default))
    except Exception:  # noqa: BLE001 - best-effort metadata
        return default


def _is_dataset(path):
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


def _find_datasets(root):
    """``root`` is either a LeRobot dataset or a container of dataset dirs
    (bag2lerobot writes one dataset per run under $LEROBOT_DATA_PATH)."""
    if _is_dataset(root):
        return [root]
    datasets = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            sub = os.path.join(root, name)
            if _is_dataset(sub):
                datasets.append(sub)
    return datasets


def _episodes_meta(dataset):
    """v3.0 per-episode metadata (meta/episodes/**/*.parquet) as a DataFrame,
    or None if absent."""
    files = sorted(glob.glob(
        os.path.join(dataset, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _data_files(dataset):
    return sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "file-*.parquet")))


def _read_episode(dataset, episode):
    """Return (times[N], state[N,D], state_names, action[N,A], action_names).

    LeRobot v3.0 packs many episodes into shared data files, each row tagged
    with an ``episode_index`` column. We locate the file via meta/episodes when
    possible (else scan all data files), then keep this episode's rows.
    """
    names = _load_state_names(dataset)

    files = []
    meta = _episodes_meta(dataset)
    if meta is not None and "data/chunk_index" in meta.columns:
        row = meta[meta["episode_index"] == episode]
        if len(row):
            chunk = int(row.iloc[0]["data/chunk_index"])
            file_idx = int(row.iloc[0]["data/file_index"])
            cand = os.path.join(dataset, "data", "chunk-%03d" % chunk,
                                "file-%03d.parquet" % file_idx)
            if os.path.exists(cand):
                files = [cand]
    if not files:
        files = _data_files(dataset)
    if not files:
        raise FileNotFoundError("No data parquet under %s/data" % dataset)

    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    frame = frame[frame["episode_index"] == episode].sort_values("frame_index")
    if frame.empty:
        raise ValueError("Episode %d not found in %s" % (episode, dataset))

    state = np.stack([np.asarray(row, dtype=np.float64) for row in frame["observation.state"]])
    actions = np.stack([np.asarray(row, dtype=np.float64) for row in frame["action"]])
    if "timestamp" in frame:
        times = np.asarray(frame["timestamp"], dtype=np.float64)
    else:  # fall back to a uniform grid from frame order + fps
        times = np.arange(len(state), dtype=np.float64) / _dataset_fps(dataset)
    return times - times[0], state, names, actions, _load_action_names(dataset)


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


def _gripper_events(times, signal, open_thr, close_thr):
    """List of (episode_time, 'open'|'close') at command transitions (hysteresis)."""
    events = []
    state = None
    for k in range(len(times)):
        w = float(signal[k])
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


def _list_episodes(root, limit):
    """Recent replayable episodes across the dataset(s) under ``root``, newest
    first. ``root`` may be a single LeRobot dataset or a container of dataset
    dirs. Each entry: {dataset, index, task, length, duration}.
    """
    entries = []
    for dataset in _find_datasets(root):
        fps = _dataset_fps(dataset)
        ds_mtime = os.path.getmtime(dataset)
        meta = _episodes_meta(dataset)
        if meta is not None:
            has_len = "length" in meta.columns
            for _, row in meta.iterrows():
                tasks = row["tasks"] if "tasks" in meta.columns else None
                task = str(tasks[0]) if tasks is not None and len(tasks) else ""
                length = int(row["length"]) if has_len else None
                entries.append({
                    "dataset": dataset, "index": int(row["episode_index"]),
                    "task": task, "length": length,
                    "duration": (length / fps) if length else None,
                    "_sort": (ds_mtime, int(row["episode_index"])),
                })
        else:  # fallback: derive episodes from the data parquet's episode_index
            files = _data_files(dataset)
            if not files:
                continue
            col = pd.concat([pd.read_parquet(p, columns=["episode_index"]) for p in files],
                            ignore_index=True)["episode_index"]
            for idx in sorted(col.unique().tolist()):
                length = int((col == idx).sum())
                entries.append({
                    "dataset": dataset, "index": int(idx), "task": "",
                    "length": length, "duration": length / fps,
                    "_sort": (ds_mtime, int(idx)),
                })
    entries.sort(key=lambda e: e["_sort"], reverse=True)
    return entries[:limit]


def _choose_episode(entries, requested):
    """Print the recent episodes and return the chosen entry {dataset, index}.

    ``requested`` >= 0 selects that episode_index within the most-recent
    dataset (non-interactive). Otherwise, with a TTY the user is prompted
    (Enter = most recent); without a TTY the most recent is auto-selected.
    """
    print("\nRecent replayable sequences:")
    print("  %3s  %-24s  %3s  %7s  %7s  %s"
          % ("sel", "dataset", "ep", "frames", "dur(s)", "task"))
    for i, e in enumerate(entries):
        ds = os.path.basename(e["dataset"].rstrip("/"))[:24]
        frames = str(e["length"]) if e["length"] is not None else "?"
        dur = ("%.1f" % e["duration"]) if e["duration"] is not None else "?"
        print("  %3d) %-24s  %3d  %7s  %7s  %s" % (i, ds, e["index"], frames, dur, e["task"]))

    if requested is not None and requested >= 0:
        recent_ds = entries[0]["dataset"]
        for e in entries:
            if e["dataset"] == recent_ds and e["index"] == requested:
                print("Episode %d requested via param; replaying it." % requested)
                return e
        rospy.logwarn("Requested episode %d not in the most recent dataset; "
                      "using the most recent entry.", requested)
        return entries[0]

    if not sys.stdin.isatty():
        rospy.loginfo("Non-interactive (no TTY): auto-selecting the most recent episode. "
                      "Run via `rosrun` in a terminal to choose, or pass _episode:=N.")
        return entries[0]

    while True:
        try:
            raw = input("\nSelect [0-%d], Enter = most recent: " % (len(entries) - 1)).strip()
        except EOFError:
            return entries[0]
        if raw == "":
            return entries[0]
        try:
            sel = int(raw)
        except ValueError:
            print("  please enter a number.")
            continue
        if 0 <= sel < len(entries):
            return entries[sel]
        print("  out of range.")


def main():
    rospy.init_node("replay_lerobot_episode")

    # Processed episodes live in $LEROBOT_DATA_PATH; an explicit ~dataset
    # overrides it, and the default ~/teleop_lerobot is used (with a warning)
    # only when neither is set.
    # data_root is a LeRobot dataset OR a container of dataset dirs (bag2lerobot
    # writes one dataset per run under $LEROBOT_DATA_PATH).
    dataset_param = rospy.get_param("~dataset", "")
    if dataset_param:
        data_root = os.path.expanduser(dataset_param)
    elif os.environ.get("LEROBOT_DATA_PATH"):
        data_root = os.path.expanduser(os.environ["LEROBOT_DATA_PATH"])
    else:
        rospy.logwarn(
            "LEROBOT_DATA_PATH not set and ~dataset empty; using default "
            "'~/teleop_lerobot'.")
        data_root = os.path.expanduser("~/teleop_lerobot")
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
    # The gripper command (action channel) is normalized to [0, 1] (1 = open,
    # 0 = closed); thresholds are on that normalized signal.
    open_thr = float(rospy.get_param("~gripper_open_threshold", 0.6))
    close_thr = float(rospy.get_param("~gripper_close_threshold", 0.4))

    joint_names = ["%s_joint%d" % (arm_id, j) for j in range(1, 8)]

    entries = _list_episodes(data_root, max_list)
    if not entries:
        rospy.logerr("No LeRobot episodes found under %s; nothing to replay.", data_root)
        return
    entry = _choose_episode(entries, requested)
    dataset, episode = entry["dataset"], entry["index"]

    times, state, names, actions, action_names = _read_episode(dataset, episode)
    joints = state[:, _joint_indices(names)]
    rospy.loginfo("Replaying %s episode %d: %d frames, %.2f s, %d-DoF joints.",
                  os.path.basename(dataset.rstrip("/")), episode, len(times),
                  float(times[-1]), joints.shape[1])

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
        gripper_idx = action_names.index("gripper") if "gripper" in action_names else None
        if gripper_idx is None:
            rospy.logwarn("No 'gripper' in the action features; skipping gripper replay.")
        else:
            grip = GripperClients(
                open_width=float(rospy.get_param("~open_width", 0.08)),
                open_speed=float(rospy.get_param("~open_speed", 0.1)),
                close_width=float(rospy.get_param("~close_width", 0.0)),
                close_speed=float(rospy.get_param("~close_speed", 0.1)),
                close_force=float(rospy.get_param("~close_force", 30.0)),
                eps_inner=float(rospy.get_param("~close_epsilon_inner", 0.005)),
                eps_outer=float(rospy.get_param("~close_epsilon_outer", 0.08)))
            # Replay the recorded gripper COMMAND (action channel), not the
            # measured width -- which barely moves when grasping an object and
            # would never cross the thresholds.
            events = _gripper_events(times, actions[:, gripper_idx], open_thr, close_thr)
            for episode_t, cmd in events:
                exec_time = traj_start + rospy.Duration.from_sec(approach_time + episode_t)
                _sleep_until(exec_time)
                if rospy.is_shutdown():
                    return
                grip.open() if cmd == "open" else grip.close()

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
