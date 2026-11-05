"""Bimanual YAM rollout: replay episode trajectories on two 6-dof YAM arms.

A self-contained sibling of the FR3 rig (``kiwi.replay.rig``) for the i2rt
YAM arm. The kinematics are copied from ``bmm_robot_description``'s
``yam_arm.xacro`` (joint origins, axes, limits); the STL meshes are replaced
by cylinders spanning each link, so the URDF needs no package resolution.
A simplified Chopstick SA (body + two hinged jaws) hangs off ``link_6`` with
the SA's CAD flange -> TCP offset, so the episode tool-tip trajectories map
onto the same ``chopstick_sa_tcp`` site the FR3 solver drives.

``yam_dual_urdf`` writes the dual-arm URDF (left at +Y, right at -Y);
``python -m kiwi.replay.yam --regenerate`` refreshes
``descriptions/yam/urdf/yam_dual_chopstick_sa.urdf``. ``kiwi replay yam``
solves an episode with the shared mink IK (``Fr3MinkIK`` is arm-agnostic)
and renders the rollout to an MP4 with a software rasterizer (Pillow +
ffmpeg), so no GPU or WebGL is needed.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import numpy as np

from kiwi.replay.gripper import (
    CLOSED,
    JOINTS as JAW_JOINTS,
    MULTIPLIERS,
    OPEN,
    TCP_SITE,
)

__all__ = [
    "ARM_JOINTS",
    "HOME_QPOS",
    "URDF_PATH",
    "build_model",
    "render_rollout",
    "solve_episode",
    "yam_dual_urdf",
]

URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "descriptions/yam/urdf/yam_dual_chopstick_sa.urdf"
)
DEFAULT_SEPARATION = 1.0

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

# (child link, joint origin xyz, joint origin rpy, lower, upper) per revolute
# joint, straight from yam_arm.xacro + config/arm/yam/joint_limits.yaml.
_CHAIN = (
    ("link_1", (0.0, 0.0, 0.0631), (0.0, 0.0, 1.5708), -2.61799, 3.13),
    ("link_2", (2.5e-05, -0.02, 0.0409), (1.5708, 0.0, 1.5708), 0.0, 3.65),
    ("link_3", (0.264, 0.0, 0.0), (-3.14159, 0.0, 0.0), 0.0, 3.13),
    ("link_4", (-0.245, -0.06, 0.0), (0.0, 0.0, 0.0), -1.65, 1.65),
    (
        "link_5",
        (-0.0739968, -0.0395003, 2.44738e-05),
        (-1.5708, 1.5708, 0.0),
        -1.5708,
        1.5708,
    ),
    ("link_6", (0.0, 0.0353, 0.0395), (-1.5708, 0.0, 0.0), -2.0944, 2.0944),
)
# The YAM's own gripper tip (unused by the solver; kept as a frame).
_HAND_TCP_Z = 0.1347

# Chopstick SA, from ChopstickSA.urdf re-expressed in its franka_flange
# frame (flange x/y/z = SA base y/z/x): hand_tcp shares the flange rotation
# at (0, -0.003, 0.194); the jaw hinges (rev_left/rev_right) sit at
# x = +/-0.015, z = 0.0733 and turn about flange +Y.
_SA_TCP = (0.0, -0.003, 0.194)
_SA_HINGE = (0.015, -0.0012, 0.0733)

# "Ready" pose: the TCP ~0.34 m in front of the base, 0.14 m up, the tool
# pitched 45 deg down. Straight-down tools are near joint4's limit on the
# YAM, so this is also the IK's posture reference.
HOME_QPOS = (0.0, 1.0, 0.9, -0.7, 0.0, 0.0)
# Where --center lands the mean tool-tip position (world, m): in front of and
# between the two bases, at the height of the home TCP.
DEFAULT_CENTER = (0.35, 0.0, 0.15)

_LINK_RADIUS = {"base_link": 0.045, "link_1": 0.035, "link_2": 0.03}
_DEFAULT_RADIUS = 0.025
_ORANGE = "1 0.6 0 1"
_DARK = "0.15 0.15 0.15 1"


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _segment_rpy(vec: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (yaw = 0) that turns a cylinder's local +Z onto ``vec``:
    Ry(p) . Rx(r) . ez = (cos r sin p, -sin r, cos r cos p)."""
    vx, vy, vz = vec / np.linalg.norm(vec)
    return (float(np.arcsin(-vy)), float(np.arctan2(vx, vz)), 0.0)


def _cylinder(xyz_a, xyz_b, radius: float, material: str) -> str:
    """A cylinder visual spanning two points of one link frame."""
    a, b = np.asarray(xyz_a, float), np.asarray(xyz_b, float)
    length = float(np.linalg.norm(b - a))
    return (
        f'    <visual>\n      <origin xyz="{_fmt((a + b) / 2)}" '
        f'rpy="{_fmt(_segment_rpy(b - a))}"/>\n'
        f'      <geometry><cylinder radius="{radius:g}" length="{length:.6g}"/>'
        f'</geometry>\n      <material name="{material}"/>\n    </visual>\n'
    )


def _hub(radius: float, length: float, material: str) -> str:
    """A joint housing: a short cylinder along the link frame's Z axis."""
    return _cylinder((0, 0, -length / 2), (0, 0, length / 2), radius, material)


# Nominal inertia for moving links: MuJoCo refuses massless moving bodies,
# and the replay is kinematic, so the values only need to be positive.
_INERTIAL = (
    '    <inertial>\n      <mass value="0.1"/>\n'
    '      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>\n'
    "    </inertial>\n"
)


def _link(name: str, body: str = "") -> str:
    if not body:
        return f'  <link name="{name}"/>\n'
    return f'  <link name="{name}">\n{_INERTIAL}{body}  </link>\n'


def _joint(name, kind, parent, child, xyz, rpy, axis=None, limit=None) -> str:
    out = (
        f'  <joint name="{name}" type="{kind}">\n'
        f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
        f'    <origin xyz="{_fmt(xyz)}" rpy="{_fmt(rpy)}"/>\n'
    )
    if axis is not None:
        out += f'    <axis xyz="{_fmt(axis)}"/>\n'
    if limit is not None:
        out += (
            f'    <limit lower="{limit[0]:.6g}" upper="{limit[1]:.6g}" '
            f'effort="40" velocity="10"/>\n'
        )
    return out + "  </joint>\n"


def _jaw_rest(hinge_x: float, mult: float) -> float:
    """Rest pitch of a jaw (about +Y) that puts its tip on the TCP at
    CLOSED; ``mult`` is the jaw's multiplier in the coupled pair."""
    closed = float(np.arctan2(_SA_TCP[0] - hinge_x, _SA_TCP[2] - _SA_HINGE[2]))
    return closed - mult * CLOSED


def _arm(prefix: str, mount_yaw: float) -> str:
    """One YAM + simplified Chopstick SA, every name under ``prefix``."""
    p = prefix
    out = _link(
        f"{p}base_link",
        _cylinder((0, 0, 0), _CHAIN[0][1], _LINK_RADIUS["base_link"], "yam_orange"),
    )
    parent = "base_link"
    for i, (child, xyz, rpy, lo, hi) in enumerate(_CHAIN):
        nxt = _CHAIN[i + 1][1] if i + 1 < len(_CHAIN) else (0, 0, 0.035)
        r = _LINK_RADIUS.get(child, _DEFAULT_RADIUS)
        body = _hub(r * 1.15, r * 2.4, "yam_dark")
        if np.linalg.norm(nxt) > 1e-6:
            body += _cylinder((0, 0, 0), nxt, r, "yam_orange")
        out += _link(f"{p}{child}", body)
        out += _joint(
            f"{p}joint{i + 1}",
            "revolute",
            f"{p}{parent}",
            f"{p}{child}",
            xyz,
            rpy,
            axis=(0, 0, 1),
            limit=(lo, hi),
        )
        parent = child

    out += _link(f"{p}hand_tcp")
    out += _joint(
        f"{p}hand_tcp_joint",
        "fixed",
        f"{p}link_6",
        f"{p}hand_tcp",
        (0, 0, _HAND_TCP_Z),
        (0, 0, 0),
    )

    # Simplified Chopstick SA: a body block, two hinged jaws, the TCP frame.
    sa = f"{p}chopstick_sa_base"
    out += _link(
        sa,
        _cylinder((0, 0, 0.0), (0, 0, _SA_HINGE[2] + 0.01), 0.028, "yam_dark"),
    )
    out += _joint(
        f"{p}chopstick_sa_mount",
        "fixed",
        f"{p}link_6",
        sa,
        (0, 0, 0),
        (0, 0, mount_yaw),
    )
    length = float(np.hypot(_SA_HINGE[0], _SA_TCP[2] - _SA_HINGE[2]))
    for jaw, mult in zip(JAW_JOINTS, MULTIPLIERS):
        x = mult * _SA_HINGE[0]
        lim = sorted((mult * CLOSED, mult * OPEN))
        link = f"{p}chopstick_sa_{jaw}"
        out += _link(link, _cylinder((0, 0, 0), (0, 0, length), 0.006, "yam_dark"))
        out += _joint(
            f"{p}{jaw}",
            "revolute",
            sa,
            link,
            (x, _SA_HINGE[1], _SA_HINGE[2]),
            (0, _jaw_rest(x, mult), 0),
            axis=(0, 1, 0),
            limit=lim,
        )
    out += _link(f"{p}chopstick_sa_tcp")
    out += _joint(
        f"{p}chopstick_sa_tcp_joint",
        "fixed",
        sa,
        f"{p}chopstick_sa_tcp",
        _SA_TCP,
        (0, 0, 0),
    )
    return out


def yam_dual_urdf(
    separation: float = DEFAULT_SEPARATION,
    mount_yaw_deg: float = 0.0,
    base_yaw_deg: tuple[float, float] = (0.0, 0.0),
) -> str:
    """Two YAM + Chopstick SA arms, bases ``separation`` apart along Y (left
    at +Y, right at -Y), facing +X on the floor turned by ``base_yaw_deg``
    (left, right) about Z."""
    half = separation / 2.0
    out = (
        '<?xml version="1.0"?>\n'
        "<!-- Generated by python -m kiwi.replay.yam --regenerate; "
        "kinematics from bmm_robot_description yam_arm.xacro. -->\n"
        '<robot name="yam_dual_chopstick_sa">\n'
        f'  <material name="yam_orange"><color rgba="{_ORANGE}"/></material>\n'
        f'  <material name="yam_dark"><color rgba="{_DARK}"/></material>\n'
        '  <link name="world"/>\n'
    )
    mount_yaw = float(np.deg2rad(mount_yaw_deg))
    for prefix, y, yaw in zip(("left_", "right_"), (half, -half), base_yaw_deg):
        out += _arm(prefix, mount_yaw)
        out += _joint(
            f"{prefix}base_joint",
            "fixed",
            "world",
            f"{prefix}base_link",
            (0, y, 0),
            (0, 0, float(np.deg2rad(yaw))),
        )
    return out + "</robot>\n"


# ---------------------------------------------------------------------------
# MuJoCo model + IK solve
# ---------------------------------------------------------------------------

SIDES = ("left", "right")
# Keep the fixed-joint frames (TCP, hand_tcp) as bodies and the cylinders as
# geoms: MuJoCo's URDF importer otherwise fuses the one and drops the other.
_MJ_EXT = '<mujoco><compiler fusestatic="false" discardvisual="false"/></mujoco>\n  '


def build_model(urdf: str | None = None):
    """Compile the dual URDF (default: the checked-in one) into a MuJoCo
    model with a ``<side>_chopstick_sa_tcp`` site per arm.

    Returns ``(model, home_qpos)``: arms at ``HOME_QPOS``, jaws open."""
    import mujoco

    text = URDF_PATH.read_text() if urdf is None else urdf
    text = text.replace('<link name="world"/>', _MJ_EXT + '<link name="world"/>', 1)
    spec = mujoco.MjSpec.from_string(text)
    for side in SIDES:
        spec.body(f"{side}_{TCP_SITE}").add_site(name=f"{side}_{TCP_SITE}")
    model = spec.compile()
    home = np.zeros(model.nq)
    for side in SIDES:
        for j, v in zip(ARM_JOINTS, HOME_QPOS):
            home[model.jnt_qposadr[model.joint(f"{side}_{j}").id]] = v
    return model, home


def centered_workspace(
    trajs: dict, center, *, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0
) -> dict:
    """The shared-frame workspace transform that lands the mean of both arms'
    tool-tip paths on ``center`` (world, m) after the ``yaw/pitch/roll``
    rotation (deg) - the episode's world origin is otherwise arbitrary."""
    from kiwi.replay.pose import _quat_rotate, _workspace_rot_quat

    mean = np.concatenate([trajs[s][:, :3] for s in SIDES]).mean(axis=0)
    anchor = np.asarray(center, float) - _quat_rotate(
        _workspace_rot_quat(yaw, pitch, roll), mean
    )
    return {
        "x": anchor[0],
        "y": anchor[1],
        "z": anchor[2],
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


@dataclasses.dataclass
class Rollout:
    """One solved episode: per-side joint, target and achieved-TCP logs."""

    qpos: np.ndarray  # (N, nq) full model qpos per frame (arms + jaws)
    joints: dict  # side -> (N, 6) arm joints (rad)
    target: dict  # side -> (N, 6) commanded TCP xyz (m) + rpy (deg)
    actual: dict  # side -> (N, 6) IK-achieved TCP xyz (m) + rpy (deg)
    grip: dict  # side -> (N,) jaw angle (rad)
    result: object  # kiwi.replay.solve.SolveResult
    fps: float


def solve_episode(
    model,
    home_qpos: np.ndarray,
    trajs: dict,
    ws: dict,
    *,
    fps: float,
    combined: bool = True,
    ik_iters: int = 120,
    ik_gain: float = 0.5,
    orientation_cost: float = 1.0,
    posture_cost: float = 1e-3,
) -> Rollout:
    """Track both arms' (N, 7) replay trajectories with the shared mink IK
    in the episode's shared frame (inter-hand geometry preserved)."""
    from kiwi.replay.ik import Fr3MinkIK
    from kiwi.replay.gripper import jaw_qpos
    from kiwi.replay.solve import _resolve_indices, solve_trajectory

    idx = {s: _resolve_indices(model, s, ARM_JOINTS) for s in SIDES}
    ik = Fr3MinkIK(
        model,
        [idx[s] for s in SIDES],
        home_qpos,
        combined=combined,
        frame_gain=ik_gain,
        orientation_cost=orientation_cost,
        posture_cost=posture_cost,
    )
    ik.reset()
    n = {s: len(trajs[s]) for s in SIDES}
    solved = {s: np.zeros((n[s], len(ARM_JOINTS))) for s in SIDES}
    target = {s: np.zeros((n[s], 6)) for s in SIDES}
    actual = {s: np.zeros((n[s], 6)) for s in SIDES}
    res = solve_trajectory(
        ik,
        idx,
        home_qpos,
        trajs,
        solved,
        target,
        actual,
        shared_frame=True,
        ws=ws,
        base_dist=0.0,  # shared frame: the bases are baked into the URDF
        ik_iters=ik_iters,
    )
    n_max = max(n.values())
    qpos = np.tile(home_qpos, (n_max, 1))
    grip = {}
    for s in SIDES:
        f = np.minimum(np.arange(n_max), n[s] - 1)
        qpos[:, idx[s]["arm_qadr"]] = solved[s][f]
        grip[s] = trajs[s][f, 6]
        qpos[:, idx[s]["gripper_qadr"]] = np.stack([jaw_qpos(g) for g in grip[s]])
    return Rollout(qpos, solved, target, actual, grip, res, fps)


def world_capsules(model, rollout: Rollout, ws: dict) -> dict:
    """Every cylinder of the rig, per frame, in the EPISODE's world frame.

    The solve places targets at ``p_rig = R_ws p_world + anchor`` (shared
    frame), so the rig sits in the world at the inverse transform. Returns
    ``ends`` (N, G, 2, 3) float32 axis endpoints, ``radius`` (G,) and
    ``color`` (G, 3) uint8 - a robot-agnostic capsule overlay that a scene
    viewer can draw without MuJoCo.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    from kiwi.replay.pose import _workspace_rot_quat

    w, x, y, z = _workspace_rot_quat(ws["yaw"], ws["pitch"], ws["roll"])
    r_ws = Rotation.from_quat([x, y, z, w]).as_matrix()
    anchor = np.array([ws["x"], ws["y"], ws["z"]], dtype=np.float64)
    geoms = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER)
    half = model.geom_size[geoms, 1]
    data = mujoco.MjData(model)
    ends = np.empty((len(rollout.qpos), len(geoms), 2, 3), dtype=np.float32)
    for f, q in enumerate(rollout.qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        axis = data.geom_xmat[geoms].reshape(-1, 3, 3)[:, :, 2] * half[:, None]
        p = data.geom_xpos[geoms]
        rig = np.stack([p - axis, p + axis], axis=1)
        ends[f] = (rig - anchor) @ r_ws  # R^T (p - a), row-vector form
    return {
        "ends": ends,
        "radius": model.geom_size[geoms, 0].astype(np.float32),
        "color": (model.geom_rgba[geoms, :3] * 255).astype(np.uint8),
    }


def save_rollout(rollout: Rollout, path: Path, **meta) -> Path:
    """Write the solved joints and TCP logs (plus ``meta`` scalars) to npz."""
    arrays = {"fps": rollout.fps, "arm_joints": np.asarray(ARM_JOINTS)}
    for s in SIDES:
        arrays[f"{s}_joints"] = rollout.joints[s]
        arrays[f"{s}_gripper"] = rollout.grip[s]
        arrays[f"{s}_target"] = rollout.target[s]
        arrays[f"{s}_actual"] = rollout.actual[s]
    for k, v in dataclasses.asdict(rollout.result).items():
        arrays[f"metric_{k}"] = v
    arrays.update({k: np.asarray(v) for k, v in meta.items()})
    np.savez(path, **arrays)
    return Path(path)


# ---------------------------------------------------------------------------
# Rollout diagnostics: IK misses, tool-tip jumps, joint jumps / flips
# ---------------------------------------------------------------------------

# Issue kinds in display priority (highest first): a joint flip usually
# happens inside an IK miss, so it paints over it. The colours are shared by
# the video, the YAM viewer and the splat overlay, so they mean the same
# thing everywhere.
ISSUES = ("joint_jump", "ik_fail", "tcp_jump")
ISSUE_RGB = {
    "ik_fail": (229, 57, 53),  # red
    "joint_jump": (216, 27, 196),  # magenta
    "tcp_jump": (251, 192, 45),  # yellow
}
ISSUE_LABEL = {"ik_fail": "IK miss", "joint_jump": "joint jump", "tcp_jump": "tip jump"}


@dataclasses.dataclass(frozen=True)
class IssueThresholds:
    """When a frame counts as a problem. Speeds are per second, so the
    thresholds do not depend on the rollout's frame rate."""

    ik_error_m: float = 0.020  # tool-tip position error
    tcp_speed_mps: float = 1.0  # achieved tool-tip speed
    joint_speed_rads: float = 10.0  # any single arm joint (the YAM velocity limit)


def diagnose(
    rollout: dict, fps: float, thr: IssueThresholds = IssueThresholds()
) -> dict:
    """Per-arm, per-frame issue flags from a rollout's logs.

    ``rollout`` maps ``<side>_joints`` (N, 6), ``<side>_target`` and
    ``<side>_actual`` (N, 6: xyz m + rpy deg) - a loaded ``replay_yam.npz``
    or ``rollout_logs(Rollout)``. Returns ``{side: {...}}`` with ``error``
    (N,) m, ``ik_fail`` (N,), ``tcp_jump`` (N,), ``source_jump`` (N,: the
    recorded target jumped too, so the jump is in the data, not the IK),
    ``joint_jump`` (N, 6) and ``joint_step`` (N, 6) rad. A step's flags land
    on the frame it arrives at.
    """
    out = {}
    for s in SIDES:
        q = np.asarray(rollout[f"{s}_joints"], float)
        tgt = np.asarray(rollout[f"{s}_target"], float)[:, :3]
        act = np.asarray(rollout[f"{s}_actual"], float)[:, :3]
        n = len(q)
        err = np.linalg.norm(tgt - act, axis=1)
        tip_step = np.r_[0.0, np.linalg.norm(np.diff(act, axis=0), axis=1)]
        tgt_step = np.r_[0.0, np.linalg.norm(np.diff(tgt, axis=0), axis=1)]
        joint_step = np.vstack([np.zeros((1, q.shape[1])), np.abs(np.diff(q, axis=0))])
        tip_limit = thr.tcp_speed_mps / fps
        out[s] = {
            "n": n,
            "error": err,
            "ik_fail": err > thr.ik_error_m,
            "tcp_jump": tip_step > tip_limit,
            "source_jump": tgt_step > tip_limit,
            "tip_step": tip_step,
            "joint_jump": joint_step > thr.joint_speed_rads / fps,
            "joint_step": joint_step,
        }
    return out


def rollout_logs(rollout: Rollout) -> dict:
    """A solved ``Rollout`` in the ``replay_yam.npz`` key layout."""
    logs = {}
    for s in SIDES:
        logs[f"{s}_joints"] = rollout.joints[s]
        logs[f"{s}_target"] = rollout.target[s]
        logs[f"{s}_actual"] = rollout.actual[s]
    return logs


def _segments(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of ``mask``, bridging gaps <= ``gap``."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def issue_events(diag: dict, times: np.ndarray, *, gap_s: float = 0.5) -> list[dict]:
    """Merge flagged frames into time-sorted events, one per kind, arm and
    (for joint jumps) joint. ``times`` (N,) seconds per frame."""
    times = np.asarray(times, float)
    fps = 1.0 / max(float(np.median(np.diff(times))) if len(times) > 1 else 1.0, 1e-9)
    gap = int(round(gap_s * fps))
    events = []
    for s in SIDES:
        d = diag[s]
        masks = [("ik_fail", None, d["ik_fail"]), ("tcp_jump", None, d["tcp_jump"])]
        masks += [
            ("joint_jump", j, d["joint_jump"][:, j])
            for j in range(d["joint_jump"].shape[1])
        ]
        for kind, joint, mask in masks:
            for a, b in _segments(mask, gap):
                ev = {"kind": kind, "side": s, "joint": joint, "start": a, "end": b}
                ev["t0"], ev["t1"] = float(times[a]), float(times[b])
                if kind == "ik_fail":
                    ev["peak"] = float(d["error"][a : b + 1].max())
                    ev["detail"] = f"up to {ev['peak'] * 1e3:.0f} mm off"
                elif kind == "tcp_jump":
                    ev["peak"] = float(d["tip_step"][a : b + 1].max())
                    src = d["source_jump"][a : b + 1].any()
                    ev["detail"] = f"{ev['peak'] * 1e3:.0f} mm step" + (
                        " (in the recording)" if src else " (IK)"
                    )
                else:
                    ev["peak"] = float(d["joint_step"][a : b + 1, joint].max())
                    ev["detail"] = (
                        f"{ARM_JOINTS[joint]} {np.degrees(ev['peak']):.0f} deg step"
                    )
                events.append(ev)
    events.sort(key=lambda e: (e["t0"], e["side"]))
    return events


def event_label(ev: dict) -> str:
    """One line for an event: ``12.30-13.10 s  left  joint jump: joint4 ...``."""
    return (
        f"{ev['t0']:.2f}-{ev['t1']:.2f} s  {ev['side']}  "
        f"{ISSUE_LABEL[ev['kind']]}: {ev['detail']}"
    )


def frame_issues(diag: dict, f: int) -> dict:
    """``{side: [(kind, joint | None), ...]}`` flagged at frame ``f``."""
    out = {}
    for s in SIDES:
        d = diag[s]
        g = min(f, d["n"] - 1)
        items = [("ik_fail", None)] if d["ik_fail"][g] else []
        items += [("joint_jump", int(j)) for j in np.flatnonzero(d["joint_jump"][g])]
        items += [("tcp_jump", None)] if d["tcp_jump"][g] else []
        out[s] = items
    return out


def _held(mask: np.ndarray, hold: int) -> np.ndarray:
    """Extend each flagged frame forward by ``hold`` frames, so a one-frame
    event stays visible long enough to see during playback."""
    if hold <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for k in range(1, hold + 1):
        out[k:] |= mask[:-k]
    return out


def geom_owners(model, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each geom: its arm (index into SIDES, -1 for none) and the arm
    joint (0-5) whose child link it belongs to (-1 for base and gripper)."""
    side = np.full(len(geoms), -1, int)
    joint = np.full(len(geoms), -1, int)
    for i, g in enumerate(geoms):
        name = model.body(model.geom_bodyid[g]).name
        for k, s in enumerate(SIDES):
            if name.startswith(f"{s}_"):
                side[i] = k
                rest = name[len(s) + 1 :]
                if rest.startswith("link_"):
                    joint[i] = int(rest[len("link_") :]) - 1
    return side, joint


def issue_codes(model, geoms: np.ndarray, diag: dict, *, hold: int = 0) -> np.ndarray:
    """(N, G) uint8 per-geom highlight: 0 = none, else 1 + index into
    ``ISSUES``. An IK miss or tip jump marks the whole arm; a joint jump
    marks the link that joint drives. Higher-priority issues win."""
    side, joint = geom_owners(model, geoms)
    n = max(diag[s]["n"] for s in SIDES)
    codes = np.zeros((n, len(geoms)), np.uint8)
    for k in reversed(range(len(ISSUES))):  # lowest priority first
        kind = ISSUES[k]
        for si, s in enumerate(SIDES):
            d = diag[s]
            pad = n - d["n"]
            if kind == "joint_jump":
                for j in range(d["joint_jump"].shape[1]):
                    m = np.r_[_held(d["joint_jump"][:, j], hold), np.zeros(pad, bool)]
                    codes[np.ix_(m, (side == si) & (joint == j))] = k + 1
            else:
                m = np.r_[_held(d[kind], hold), np.zeros(pad, bool)]
                codes[np.ix_(m, side == si)] = k + 1
    return codes


def tint(rgb, code: int, amount: float = 0.7) -> tuple[int, int, int]:
    """``rgb`` blended toward the issue colour for ``code`` (0 = unchanged)."""
    if not code:
        return tuple(int(c) for c in rgb)
    hi = ISSUE_RGB[ISSUES[code - 1]]
    return tuple(int(round((1 - amount) * c + amount * h)) for c, h in zip(rgb, hi))


def demo_trajectories(seconds: float = 8.0, fps: float = 30.0) -> dict:
    """A synthetic bimanual take in the replay (N, 7) layout: both tools
    pitched down as at ``HOME_QPOS``, the tips sweeping toward each other,
    meeting, and pulling apart while the jaws close and reopen. Centred on
    the world origin; for trying the rig without an episode."""
    from scipy.spatial.transform import Rotation

    from kiwi.replay.state import convert_ee_pose

    t = np.linspace(0.0, 1.0, int(seconds * fps))
    meet = np.sin(np.pi * t) ** 2  # 0 -> 1 -> 0
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        pos = np.stack(
            [
                0.05 * np.sin(2 * np.pi * t),
                sign * (0.25 - 0.19 * meet),
                0.08 * np.sin(np.pi * t) - 0.04 * np.cos(4 * np.pi * t),
            ],
            axis=1,
        )
        # The home tool frame (+Z pitched 45 deg down, +X along world +Y),
        # yawed in toward the other hand as they meet.
        yaw = Rotation.from_euler(
            "z", -sign * (15.0 + 25.0 * meet)[:, None], degrees=True
        )
        rot = yaw * Rotation.from_euler("XYZ", (180.0, 45.0, -90.0), degrees=True)
        grip01 = np.clip(1.0 - 1.6 * meet, 0.0, 1.0)
        out[side] = convert_ee_pose(
            pos, rot, grip01, rot_offset_deg=(0.0, 0.0, 0.0), traj_yaw_deg=0.0
        )
    return out


# ---------------------------------------------------------------------------
# Software renderer (no GPU / WebGL): capsule rasterization with Pillow
# ---------------------------------------------------------------------------

# Arm colours follow the real2simviz convention: left blue, right green.
_SIDE_RGB = {"left": (66, 133, 244), "right": (52, 168, 83)}
_BG = (246, 246, 243)
_GRID = (214, 214, 208)
_INK = (40, 40, 40)


@dataclasses.dataclass(frozen=True)
class View:
    """A pinhole camera orbiting ``target``: azimuth from +X toward +Y,
    elevation above the floor (deg), distance (m), vertical FOV (deg)."""

    title: str
    azimuth: float
    elevation: float
    distance: float = 2.2
    target: tuple[float, float, float] = (0.3, 0.0, 0.15)
    fov: float = 40.0

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """(eye, R) with R's rows the camera right/up/forward axes."""
        az, el = np.deg2rad(self.azimuth), np.deg2rad(self.elevation)
        tgt = np.asarray(self.target, float)
        fwd = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        eye = tgt - self.distance * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, np.stack([right, up, fwd])


class _Camera:
    def __init__(self, view: View, width: int, height: int) -> None:
        self.eye, self.R = view.basis()
        self.f = 0.5 * height / np.tan(np.deg2rad(view.fov) / 2)
        self.c = np.array([width / 2, height / 2])

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World (..., 3) -> pixel (..., 2) and depth (...)."""
        cam = (np.asarray(pts, float) - self.eye) @ self.R.T
        z = np.maximum(cam[..., 2], 1e-3)
        px = self.c + self.f * np.stack([cam[..., 0], -cam[..., 1]], -1) / z[..., None]
        return px, z


def _shade(rgb, k: float) -> tuple[int, int, int]:
    return tuple(
        int(np.clip(c * k + 255 * max(k - 1.0, 0.0) * 0.5, 0, 255)) for c in rgb
    )


def _cylinder_geoms(model) -> np.ndarray:
    import mujoco

    return np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER)


def _capsules(
    model, data, codes=None
) -> list[tuple[np.ndarray, np.ndarray, float, tuple]]:
    """Every cylinder geom as (end_a, end_b, radius, rgb) in world, tinted
    by ``codes`` (one ``issue_codes`` row) when given."""
    out = []
    for i, g in enumerate(_cylinder_geoms(model)):
        r, h = model.geom_size[g, 0], model.geom_size[g, 1]
        axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
        p = data.geom_xpos[g]
        rgb = tuple(int(255 * c) for c in model.geom_rgba[g, :3])
        if codes is not None:
            rgb = tint(rgb, int(codes[i]))
        out.append((p - h * axis, p + h * axis, float(r), rgb))
    return out


def issue_strip(diag: dict, width: int, height: int, *, bg=_BG, ink=_INK):
    """A timeline image: one row per arm, each column coloured by the
    highest-priority issue among the frames it covers (``bg`` where clean).
    Returns a PIL image without a playhead."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10, height // 4))
    label_w = 18
    row_h = (height - 6) // len(SIDES)
    for r, s in enumerate(SIDES):
        d = diag[s]
        y0 = 3 + r * row_h
        draw.text((3, y0 + row_h / 2 - 6), s[0].upper(), fill=ink, font=font)
        x0, x1 = label_w, width - 3
        draw.rectangle(
            [x0, y0 + 1, x1, y0 + row_h - 2], fill=tuple(int(0.9 * c) for c in bg)
        )
        n = d["n"]
        cols = np.minimum(
            ((np.arange(n) / max(n - 1, 1)) * (x1 - x0)).astype(int), x1 - x0
        )
        for kind in reversed(ISSUES):  # higher priority paints last
            mask = d[kind].any(axis=1) if d[kind].ndim == 2 else d[kind]
            for c in np.unique(cols[mask]):
                draw.line(
                    [x0 + c, y0 + 1, x0 + c, y0 + row_h - 2], fill=ISSUE_RGB[kind]
                )
    return img


def _draw_panel(
    img, cam: _Camera, caps, paths, frame: int, light: np.ndarray, ss: int, misses=()
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    # Floor grid (10 cm), drawn first.
    for v in np.arange(-0.8, 0.81, 0.1):
        for a, b in (((v, -0.9, 0), (v, 0.9, 0)), ((-0.3, v, 0), (0.9, v, 0))):
            if abs(v) > 0.9 and b[0] == 0.9:
                continue
            px, _ = cam.project(np.array([a, b]))
            draw.line([tuple(px[0]), tuple(px[1])], fill=_GRID, width=ss)

    # Tool-tip paths: the whole commanded path faint, the travelled part solid.
    for side, pts in paths.items():
        px, _ = cam.project(pts)
        faint = tuple(int(0.35 * c + 0.65 * b) for c, b in zip(_SIDE_RGB[side], _BG))
        draw.line([tuple(p) for p in px], fill=faint, width=2 * ss)
        if frame > 0:
            draw.line(
                [tuple(p) for p in px[: frame + 1]], fill=_SIDE_RGB[side], width=3 * ss
            )

    # IK misses: a red line from the achieved tool tip to its target.
    for act, tgt in misses:
        px, _ = cam.project(np.array([act, tgt]))
        draw.line([tuple(px[0]), tuple(px[1])], fill=ISSUE_RGB["ik_fail"], width=3 * ss)
        r = 5 * ss
        draw.ellipse(
            [px[1][0] - r, px[1][1] - r, px[1][0] + r, px[1][1] + r],
            outline=ISSUE_RGB["ik_fail"],
            width=2 * ss,
        )

    # Capsules back to front (painter's algorithm on the midpoint depth).
    items = []
    for a, b, r, rgb in caps:
        px, z = cam.project(np.array([a, b]))
        axis = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        lam = 0.55 + 0.45 * np.sqrt(max(0.0, 1.0 - float(axis @ light) ** 2))
        items.append((float(z.mean()), px, cam.f * r / z, rgb, lam))
    for _, px, rad, rgb, lam in sorted(items, key=lambda it: -it[0]):
        w = max(1, int(round(rad.mean() * 2)))
        base, edge, hi = (
            _shade(rgb, lam),
            _shade(rgb, lam * 0.6),
            _shade(rgb, lam * 1.25),
        )
        seg = [tuple(px[0]), tuple(px[1])]
        for p, rr in zip(px, rad):
            draw.ellipse(
                [p[0] - rr - ss, p[1] - rr - ss, p[0] + rr + ss, p[1] + rr + ss],
                fill=edge,
            )
        draw.line(seg, fill=edge, width=w + 2 * ss)
        for p, rr in zip(px, rad):
            draw.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=base)
        draw.line(seg, fill=base, width=w)
        # A specular stripe offset toward the top-left of the silhouette.
        d = px[1] - px[0]
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        n *= -np.sign(n[1] + 0.3 * n[0] + 1e-9) * rad.mean() * 0.4
        draw.line([tuple(px[0] + n), tuple(px[1] + n)], fill=hi, width=max(1, w // 4))


def render_rollout(
    model,
    rollout: Rollout,
    output: Path,
    *,
    views: tuple[View, ...] | None = None,
    size: tuple[int, int] = (720, 600),
    title: str = "",
    thresholds: IssueThresholds = IssueThresholds(),
) -> Path:
    """Render a solved rollout to ``output`` (``.mp4`` through ffmpeg, else an
    animated ``.gif``): one panel per view, side by side, with the commanded
    tool-tip paths and per-frame tracking error. Issues (``diagnose``) tint
    the affected arm or link, draw IK misses as target lines, and show on a
    timeline strip along the bottom."""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    views = views or (
        View("3/4 view", azimuth=-145.0, elevation=28.0),
        View("top", azimuth=180.0, elevation=89.0, distance=2.0),
    )
    ss = 2  # supersampling factor
    w, h = size
    cams = [_Camera(v, w * ss, h * ss) for v in views]
    light = np.array([0.3, -0.4, 0.87])
    light /= np.linalg.norm(light)
    paths = {s: rollout.target[s][:, :3] for s in SIDES}
    font = ImageFont.load_default(size=14)
    data = mujoco.MjData(model)
    n = len(rollout.qpos)
    strip_h = 44
    out_w, out_h = w * len(views), h + strip_h + 24
    diag = diagnose(rollout_logs(rollout), rollout.fps, thresholds)
    codes = issue_codes(
        model, _cylinder_geoms(model), diag, hold=int(round(0.2 * rollout.fps))
    )
    strip = issue_strip(diag, out_w - 24, strip_h)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") if output.suffix == ".mp4" else None
    proc = None
    gif = []
    if output.suffix == ".mp4" and ffmpeg is None:
        output = output.with_suffix(".gif")
    if ffmpeg:
        proc = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{out_w}x{out_h}",
                "-r",
                f"{rollout.fps:g}",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(output),
            ],
            stdin=subprocess.PIPE,
        )
    for f in range(n):
        data.qpos[:] = rollout.qpos[f]
        mujoco.mj_kinematics(model, data)
        caps = _capsules(model, data, codes[f])
        misses = [
            (rollout.actual[s][g, :3], rollout.target[s][g, :3])
            for s in SIDES
            if diag[s]["ik_fail"][g := min(f, diag[s]["n"] - 1)]
        ]
        frame = Image.new("RGB", (out_w, out_h), _BG)
        for i, (view, cam) in enumerate(zip(views, cams)):
            panel = Image.new("RGB", (w * ss, h * ss), _BG)
            _draw_panel(panel, cam, caps, paths, f, light, ss, misses)
            panel = panel.resize((w, h), Image.LANCZOS)
            ImageDraw.Draw(panel).text((12, 10), view.title, fill=_INK, font=font)
            frame.paste(panel, (i * w, 0))
        draw = ImageDraw.Draw(frame)
        errs = "   ".join(
            f"{s} err {1000 * np.linalg.norm(rollout.target[s][min(f, len(rollout.target[s]) - 1), :3] - rollout.actual[s][min(f, len(rollout.actual[s]) - 1), :3]):5.1f} mm"
            for s in SIDES
        )
        hud = f"{title}  t={f / rollout.fps:5.2f}s  frame {f + 1}/{n}   {errs}"
        draw.text((12, h - 22), hud.strip(), fill=_INK, font=font)
        # Current issues, then the timeline strip with a playhead.
        x = 12
        for s, items in frame_issues(diag, f).items():
            for kind, joint in items:
                text = f"{s}: " + (
                    ARM_JOINTS[joint] + " jump"
                    if joint is not None
                    else ISSUE_LABEL[kind]
                )
                draw.rectangle([x, h + 4, x + 10, h + 14], fill=ISSUE_RGB[kind])
                draw.text((x + 14, h + 1), text, fill=_INK, font=font)
                x += 24 + int(draw.textlength(text, font=font))
        frame.paste(strip, (12, h + 22))
        px = 12 + 18 + int(f / max(n - 1, 1) * (strip.width - 21))
        draw.line([px, h + 20, px, h + 22 + strip_h], fill=_INK, width=2)
        legend = [(f"{s} arm", _SIDE_RGB[s]) for s in SIDES]
        legend += [(ISSUE_LABEL[k], ISSUE_RGB[k]) for k in ISSUES]
        for k, (text, rgb) in enumerate(legend):
            x0, y0 = out_w - 150, 10 + 16 * k
            draw.rectangle([x0, y0 + 3, x0 + 10, y0 + 13], fill=rgb)
            draw.text((x0 + 16, y0 + 2), text, fill=_INK, font=font)
        if proc is not None:
            proc.stdin.write(frame.tobytes())
        else:
            gif.append(frame)
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {output}")
    else:
        gif[0].save(
            output,
            save_all=True,
            append_images=gif[1:],
            loop=0,
            duration=int(1000 / rollout.fps),
        )
    return output


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regenerate", action="store_true", help=f"rewrite {URDF_PATH}")
    ap.add_argument("--check", action="store_true", help="fail if the URDF is stale")
    args = ap.parse_args()
    text = yam_dual_urdf()
    if args.check:
        if not URDF_PATH.is_file() or URDF_PATH.read_text() != text:
            raise SystemExit(f"{URDF_PATH} is stale: run with --regenerate")
    elif args.regenerate:
        URDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        URDF_PATH.write_text(text)
        print(f"wrote {URDF_PATH}")
    else:
        ap.print_help()
