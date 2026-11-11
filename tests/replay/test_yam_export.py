"""kiwi replay yam-export: the 50 Hz joint CSV for openarm_ros2's replay_trajectory.py."""

import numpy as np

from kiwi.replay import yam_export
from kiwi.replay.gripper import CLOSED


def _rollout(n=31, fps=30.0):
    t = np.arange(n) / fps
    q = np.tile(np.array([0.1, 1.0, 1.0, 0.0, 0.0, 0.0]), (n, 1))
    q[:, 0] += 0.2 * t  # slow joint1 sweep
    tip = np.zeros((n, 6))
    tip[:, 0] = 0.3 + 0.01 * t
    r = {"fps": fps, "times": t}
    for s in ("left", "right"):
        r[f"{s}_joints"] = q.copy()
        r[f"{s}_gripper"] = np.full(n, CLOSED / 2)
        r[f"{s}_target"] = tip.copy()
        r[f"{s}_actual"] = tip.copy()
    return r


def test_columns_rate_and_gripper():
    out, report = yam_export.export_joints(_rollout())
    assert out.shape == (51, 16)  # 1 s at 50 Hz, inclusive
    assert np.allclose(out[:, 6], yam_export.FOLLOWER_GRIPPER_OPEN / 2)
    assert np.allclose(out[-1, 0], 0.3)
    assert out[:, 14:].all() and report["right"]["unsolved_s"] == []


def test_ik_miss_and_limits_mark_steps_unusable():
    r = _rollout()
    r["right_actual"][10:15, 0] += 0.05  # 50 mm tip error
    r["left_joints"][20:, 3] = 1.7  # past joint4's 1.5708
    out, report = yam_export.export_joints(r)
    assert not out[:, 15].all() and report["right"]["unsolved_s"][0][0] < 0.5
    assert not out[-1, 14] and np.isclose(out[:, 3].max(), 1.5708)  # clipped


def test_demo_video_argv_shifts_onto_the_global_clock():
    a = yam_export.demo_video_argv(
        "in.mp4", "out.mp4", offset_s=-0.192, duration_s=63.3
    )
    assert a[a.index("-ss") + 1] == "0.192" and a[a.index("-t") + 1] == "63.300"
    assert "tpad" not in a[a.index("-vf") + 1]
    b = yam_export.demo_video_argv("in.mp4", "out.mp4", offset_s=0.5, duration_s=10)
    assert (
        b[b.index("-ss") + 1] == "0.000"
        and "tpad=start_duration=0.500" in b[b.index("-vf") + 1]
    )


def test_task_poses_are_in_each_arms_base_frame():
    pytest = __import__("pytest")
    pytest.importorskip("scipy")
    from scipy.spatial.transform import Rotation

    r = _rollout(n=4)
    r["separation"], r["base_yaw_deg"] = 1.0, np.array([-90.0, 90.0])
    local = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True)
    for s, Tb in yam_export.base_poses(r).items():
        p_w = Tb[:3, :3] @ np.array([0.4, 0.05, 0.2]) + Tb[:3, 3]
        R_w = Rotation.from_matrix(Tb[:3, :3]) * local
        r[f"{s}_target"] = np.tile(
            np.r_[p_w, R_w.as_euler("xyz", degrees=True)], (4, 1)
        )
    task = yam_export.export_task(r)
    for c in (0, 7):
        assert np.allclose(task[:, c : c + 3], [0.4, 0.05, 0.2])
        q = Rotation.from_quat(task[0, c + 3 : c + 7])
        assert (q * local.inv()).magnitude() < 1e-9
