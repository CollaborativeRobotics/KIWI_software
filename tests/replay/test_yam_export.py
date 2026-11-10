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
