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

Task-space poses (``export_task``, ``task_50hz.csv`` beside the CSV): each arm's
demo tool pose - the Chopstick SA tip KIWI's IK tracked, ``SA_TCP`` in link_6 -
in that arm's own base frame (base_link: on the floor, X forward, Z up), as
x y z (m) and a quaternion qx qy qz qw, on the same 50 Hz rows. A replay can
apply the demo's motion relative to where the arm actually starts (delta
mode in openarm_ros2's replay_trajectory.py), so only the bases' yaw matters,
not where the tags put them.

Demo videos (``write_demo_videos``): each stream of the episode re-encoded on the
trajectory's clock (the episode's global clock), as ``demo_<label>.mp4`` beside
the CSV, so a frame at media time t shows the demo at trajectory time t. The
replay UI plays them in step with the replay.

The rollout's frames are resampled linearly to 50 Hz (the YAM control period).
Arm joints are clipped to ``JOINT_LIMITS``. The Chopstick SA angle (OPEN 0 to
CLOSED) maps linearly to the YAM gripper's total opening (FOLLOWER_GRIPPER_OPEN
to 0); replay_trajectory.py ignores the gripper columns.
"""

from __future__ import annotations

import json
import subprocess
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


SA_TCP = (
    0.0,
    -0.003,
    0.194,
)  # the Chopstick SA tip in link_6 (kiwi.replay.yam _SA_TCP, mount yaw 0)
TASK_HEADER = ",".join(
    f"{s}_{k}" for s in SIDES for k in ("x", "y", "z", "qx", "qy", "qz", "qw")
)


def base_poses(rollout: dict) -> dict[str, np.ndarray]:
    """Each arm's base_link pose (4x4) in the solve's world frame: ``yam_dual_urdf``
    puts left at +Y, right at -Y, ``separation`` apart, turned by ``base_yaw_deg``."""
    from scipy.spatial.transform import Rotation

    sep = float(rollout.get("separation", 1.0))
    yaws = np.asarray(rollout.get("base_yaw_deg", (0.0, 0.0)), float)
    out = {}
    for s, y, yaw in zip(SIDES, (sep / 2.0, -sep / 2.0), yaws):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        T[:3, 3] = (0.0, y, 0.0)
        out[s] = T
    return out


def export_task(rollout: dict, *, rate_hz: float = RATE_HZ) -> np.ndarray:
    """(steps, 14): per arm (left, right) the demo's tool pose (``<side>_target``)
    in that arm's base frame, x y z qx qy qz qw, resampled like ``export_joints``
    (positions linearly, rotations by slerp)."""
    from scipy.spatial.transform import Rotation, Slerp

    fps = float(rollout["fps"])
    n = len(rollout["left_joints"])
    t = (
        np.asarray(rollout["times"], float)
        if "times" in rollout
        else np.arange(n) / fps
    )
    tq = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate_hz)
    cols = []
    for s, Tb in base_poses(rollout).items():
        tgt = np.asarray(rollout[f"{s}_target"], float)
        R_w = Rotation.from_euler(
            "xyz", tgt[:, 3:6], degrees=True
        )  # as kiwi.replay.solve logs it
        Rb_inv = Rotation.from_matrix(Tb[:3, :3]).inv()
        pos = (tgt[:, :3] - Tb[:3, 3]) @ Tb[:3, :3]  # into the base frame
        rot = Slerp(t, Rb_inv * R_w)(tq)
        cols += [
            np.stack([np.interp(tq, t, pos[:, i]) for i in range(3)], 1),
            rot.as_quat(),
        ]
    return np.concatenate(cols, 1)


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
    task = out.with_name("task_50hz.csv")
    np.savetxt(
        task,
        export_task(rollout, **{k: v for k, v in kw.items() if k == "rate_hz"}),
        delimiter=",",
        fmt="%.7f",
        header=TASK_HEADER,
    )
    report["task"] = {
        "file": task.name,
        "frame": "each arm's base_link",
        "tool": "chopstick_sa_tcp",
        "tool_in_link6_m": list(SA_TCP),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def demo_video_argv(
    src: Path, dst: Path, offset_s: float, duration_s: float, size: int = 480
) -> list[str]:
    """ffmpeg argv: ``src`` (a stream at ``offset_s`` on the global clock) cut and
    shifted so media time t is global time t, scaled to fit ``size``, H.264 with
    a keyframe every 0.5 s so the browser seeks quickly. Pure."""
    skip = max(0.0, -offset_s)  # recorded offsets are <= 0: the stream started earlier
    pad = max(0.0, offset_s)  # a provisional positive offset: hold its first frame
    vf = (
        f"scale={size}:{size}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    if pad > 0:
        vf += f",tpad=start_duration={pad:.3f}:start_mode=clone"
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{skip:.3f}", "-i", str(src),
        "-map", "0:v:0", "-t", f"{duration_s:.3f}", "-an", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", "15",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]  # fmt: skip


def write_demo_videos(
    episode: Path, out_dir: Path, duration_s: float
) -> dict[str, str]:
    """``demo_<label>.mp4`` in ``out_dir`` for every synced stream of ``episode``
    with a playable video (building its H.264 proxy if needed). Returns label ->
    file name; streams without a video or a sync offset are skipped."""
    from kiwi.app.sources import viz_sources
    from kiwi.viz.proxies import playable_video

    src = viz_sources(Path(episode)).sources
    offsets = src.clock.offsets_s if src.clock is not None else {}
    out: dict[str, str] = {}
    for stream in src.streams:
        if stream.label not in offsets or not stream.video.is_file():
            continue
        video = playable_video(stream.video, dst=stream.proxy)
        if video is None:
            continue
        dst = Path(out_dir) / f"demo_{stream.label}.mp4"
        subprocess.run(
            demo_video_argv(video, dst, float(offsets[stream.label]), duration_s),
            check=True,
        )
        out[stream.label] = dst.name
    return out
