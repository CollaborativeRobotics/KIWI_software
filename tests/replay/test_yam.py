"""The dual-YAM rollout: generated URDF, kinematics, IK and the renderer."""

import numpy as np
import pytest

from kiwi.replay import yam


def test_checked_in_urdf_is_current():
    assert yam.URDF_PATH.read_text() == yam.yam_dual_urdf()


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("mujoco")
    return yam.build_model()


def test_bases_and_jaws(model):
    import mujoco

    m, home = model
    d = mujoco.MjData(m)
    d.qpos[:] = home
    mujoco.mj_kinematics(m, d)
    for side, y in (("left", 0.5), ("right", -0.5)):
        assert np.allclose(d.body(f"{side}_base_link").xpos, [0, y, 0])
        assert d.site(f"{side}_chopstick_sa_tcp").xpos[0] > 0.2  # reaches +X

    # Closed jaws meet at the TCP.
    from kiwi.replay.gripper import CLOSED, jaw_qpos

    for side in yam.SIDES:
        for j, v in zip(("rev_left", "rev_right"), jaw_qpos(CLOSED)):
            d.qpos[m.jnt_qposadr[m.joint(f"{side}_{j}").id]] = v
    mujoco.mj_kinematics(m, d)
    for side in yam.SIDES:
        tcp = d.site(f"{side}_chopstick_sa_tcp").xpos
        for j in ("rev_left", "rev_right"):
            b = d.body(f"{side}_chopstick_sa_{j}")
            g = m.body_geomadr[b.id] if m.body_geomnum[b.id] else None
            assert g is not None
            tip = (
                d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
            )
            assert np.linalg.norm(tip - tcp) < 0.003


def test_demo_tracks_and_renders(model, tmp_path):
    pytest.importorskip("mink")
    pytest.importorskip("PIL")
    m, home = model
    trajs = {s: q[::8] for s, q in yam.demo_trajectories().items()}
    ws = yam.centered_workspace(trajs, yam.DEFAULT_CENTER)
    ro = yam.solve_episode(m, home, trajs, ws, fps=30.0 / 8)
    assert ro.result.worst_pos < 0.002
    assert ro.result.qp_failures == 0
    assert ro.qpos.shape == (len(trajs["left"]), m.nq)

    npz = yam.save_rollout(ro, tmp_path / "r.npz")
    with np.load(npz) as d:
        assert d["left_joints"].shape == (len(trajs["left"]), 6)

    out = yam.render_rollout(m, ro, tmp_path / "r.gif", size=(160, 120))
    assert out.stat().st_size > 0


def test_viewer_rebuilds_saved_rollout(model, tmp_path):
    pytest.importorskip("mink")
    from kiwi.replay.yam_viewer import _qpos_frames, load_rollout

    m, home = model
    trajs = {s: q[::24] for s, q in yam.demo_trajectories().items()}
    ws = yam.centered_workspace(trajs, yam.DEFAULT_CENTER)
    ro = yam.solve_episode(m, home, trajs, ws, fps=1.25)
    yam.save_rollout(ro, tmp_path / "replay_yam.npz", separation=1.0, mount_yaw=0.0)

    loaded = load_rollout(tmp_path)  # an episode dir resolves to its npz
    assert loaded["separation"] == 1.0
    assert np.allclose(_qpos_frames(m, home, loaded), ro.qpos)


def _logs(n=30):
    """Two arms holding still: tool tips on target, joints constant."""
    logs = {}
    for s in yam.SIDES:
        logs[f"{s}_joints"] = np.zeros((n, 6))
        logs[f"{s}_target"] = np.zeros((n, 6))
        logs[f"{s}_actual"] = np.zeros((n, 6))
    return logs


def test_diagnose_flags_each_issue():
    logs = _logs()
    logs["left_actual"][10:14, 0] = 0.05  # 50 mm short of the target: IK miss
    logs["right_joints"][20:, 3] = 1.0  # joint4 flips by 1 rad in one frame
    logs["right_actual"][20:, 1] = 0.2  # ...and the tip leaps 200 mm
    logs["right_target"][20:, 1] = 0.2
    diag = yam.diagnose(logs, fps=30.0)

    assert np.flatnonzero(diag["left"]["ik_fail"]).tolist() == [10, 11, 12, 13]
    assert not diag["right"]["ik_fail"].any()  # it jumped onto the target
    assert np.flatnonzero(diag["right"]["joint_jump"][:, 3]).tolist() == [20]
    assert diag["right"]["tcp_jump"][20] and diag["right"]["source_jump"][20]

    events = yam.issue_events(diag, np.arange(30) / 30.0)
    kinds = {(e["side"], e["kind"], e["joint"]) for e in events}
    assert kinds == {
        ("left", "ik_fail", None),
        ("left", "tcp_jump", None),  # stepping off and back onto the path
        ("right", "joint_jump", 3),
        ("right", "tcp_jump", None),
    }
    assert "joint4" in yam.event_label(
        next(e for e in events if e["kind"] == "joint_jump")
    )


def test_issue_codes_mark_arm_or_link(model):
    m, _ = model
    logs = _logs()
    logs["left_actual"][5, 0] = 0.05
    logs["right_joints"][20:, 1] = 1.0
    geoms = yam._cylinder_geoms(m)
    codes = yam.issue_codes(m, geoms, yam.diagnose(logs, fps=30.0))
    side, joint = yam.geom_owners(m, geoms)
    miss = 1 + yam.ISSUES.index("ik_fail")
    flip = 1 + yam.ISSUES.index("joint_jump")
    assert (codes[5, side == 0] == miss).all()  # the whole left arm
    assert (codes[5, side == 1] == 0).all()
    assert (codes[20, (side == 1) & (joint == 1)] == flip).all()  # link_2 only
    assert (codes[20, (side == 1) & (joint == 3)] == 0).all()
