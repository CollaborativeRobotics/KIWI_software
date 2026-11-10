"""Export a solved YAM rollout as a joint trajectory for the real robot and Isaac.

Reads ``replay_yam.npz`` (written by ``kiwi replay yam``) and writes a 50 Hz CSV
that openarm_ros2's ``replay_trajectory.py`` (real YAM followers or the Isaac
ROS station) and isaac_yam replay drivers consume:

- columns 1-14 in isaac_yam / openarm ``DESIRED_JOINT_ORDER``: left joint1-6,
  left gripper opening (m), right joint1-6, right gripper opening (m);
- columns 15-16 ``left_ok, right_ok``: 1 where that arm's solve is usable, by
  ``diagnose`` (tool-tip error <= 20 mm, no tip or joint jump) and within the
  joint limits to ``limit_tol`` rad. A 50 Hz step is ok only if the solved
  frames on both sides of it are ok.

The rollout's frames are resampled linearly to 50 Hz (the YAM control period).
Arm joints are clipped to ``JOINT_LIMITS``. The Chopstick SA angle (OPEN 0 to
CLOSED) maps linearly to the YAM gripper's total opening (FOLLOWER_GRIPPER_OPEN
to 0); replay_trajectory.py ignores the gripper columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiwi.replay.gripper import CLOSED
from kiwi.replay.yam import SIDES, IssueThresholds, diagnose

RATE_HZ = 50.0
FOLLOWER_GRIPPER_OPEN = 0.09385  # m, openarm_ros2 common.constants
# joint1-6 (lower, upper) rad: the tighter of the real YAM
# (bmm_robot_description joint_limits.yaml, as in kiwi.replay.yam) and
# isaac_yam's robot_description/yam/yam.urdf (joint1 upper 3.05433, joint4
# +-1.5708), so the sim and the robot share one range.
JOINT_LIMITS = np.array(
    [
        [-2.61799, 3.05433],
        [0.0, 3.65],
        [0.0, 3.13],
        [-1.5708, 1.5708],
        [-1.5708, 1.5708],
        [-2.0944, 2.0944],
    ]
)
HEADER = (
    "left_j1,left_j2,left_j3,left_j4,left_j5,left_j6,left_grip_m,"
    "right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_grip_m,"
    "left_ok,right_ok"
)


def _runs(mask: np.ndarray, dt: float) -> list[list[float]]:
    """[start, end] seconds of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [
        [round(float(a * dt), 2), round(float((b - 1) * dt), 2)]
        for a, b in zip(edges[::2], edges[1::2])
    ]


def export_joints(
    rollout: dict,
    *,
    rate_hz: float = RATE_HZ,
    limit_tol: float = 0.02,
    thr: IssueThresholds | None = None,
) -> tuple[np.ndarray, dict]:
    """(steps, 16) array as described in the module docstring, and a report."""
    fps = float(rollout["fps"])
    n = len(rollout["left_joints"])
    t = (
        np.asarray(rollout["times"], float)
        if "times" in rollout
        else np.arange(n) / fps
    )
    dt = 1.0 / rate_hz
    tq = np.arange(0.0, t[-1] + 1e-9, dt)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    diag = diagnose(rollout, fps, thr or IssueThresholds())

    def interp(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).reshape(len(t), -1)
        return np.stack([np.interp(tq, t, x[:, i]) for i in range(x.shape[1])], 1)

    cols, oks, report = [], [], {"frames": len(tq), "seconds": round(len(tq) * dt, 3)}
    for s in SIDES:
        q_raw = np.asarray(rollout[f"{s}_joints"], float)
        d = diag[s]
        bad = d["ik_fail"] | d["tcp_jump"] | d["joint_jump"].any(1)
        bad |= ((q_raw < lo - limit_tol) | (q_raw > hi + limit_tol)).any(1)
        ok = np.interp(tq, t, (~bad).astype(float)) > 0.999
        q = interp(q_raw)
        clipped = np.clip(q, lo, hi)
        grip = interp(rollout[f"{s}_gripper"])[:, 0]
        opening = np.clip(1.0 - grip / CLOSED, 0.0, 1.0) * FOLLOWER_GRIPPER_OPEN
        cols += [clipped, opening[:, None]]
        oks.append(ok)
        report[s] = {
            "unsolved_s": _runs(~ok, dt),
            "clipped_steps_per_joint": [
                int(c) for c in (np.abs(clipped - q) > 1e-9).sum(0)
            ],
            "max_clip_rad": round(float(np.abs(clipped - q).max()), 4),
        }
    out = np.concatenate(cols + [np.stack(oks, 1).astype(float)], 1)
    return out, report


def write_joints_csv(rollout_path: Path, out: Path, **kw) -> dict:
    """Export ``rollout_path`` (replay_yam.npz, or an episode dir holding it)
    to the CSV ``out``; writes ``<out stem>.json`` with the report beside it."""
    rollout_path = Path(rollout_path)
    if rollout_path.is_dir():
        rollout_path = rollout_path / "replay_yam.npz"
    with np.load(rollout_path, allow_pickle=True) as z:
        rollout = {k: z[k] for k in z.files}
    data, report = export_joints(rollout, **kw)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, data, delimiter=",", fmt="%.6f", header=HEADER)
    report["source"] = str(rollout_path)
    for k in ("separation", "base_yaw_deg", "left_tag", "right_tag"):
        if k in rollout:
            report[k] = np.asarray(rollout[k]).tolist()
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
