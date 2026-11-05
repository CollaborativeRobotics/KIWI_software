"""Interactive playback of a solved YAM rollout (``replay_yam.npz``) in viser.

Plays back what ``kiwi replay yam`` solved; nothing is re-solved here. The
rig is rebuilt from the separation, base yaws and gripper mount recorded in the npz,
posed from the stored joint and jaw trajectories, and drawn through the
vendored mjviser bridge. The controls follow ``kiwi real2simviz``: a time
slider, play / speed / loop, frame stepping, and path toggles, with the
commanded tool-tip paths and live per-arm tracking error. The Issues folder
flags IK misses, tool-tip jumps and joint jumps (``yam.diagnose``) with
live-tunable thresholds, a timeline strip, and jump-to / next / previous.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from kiwi.core.layout import REPLAY_YAM
from kiwi.replay import yam
from kiwi.replay.gripper import jaw_qpos

__all__ = ["launch", "load_rollout"]

_SIDE_RGB = {"left": (66, 133, 244), "right": (52, 168, 83)}


def load_rollout(path: Path) -> dict:
    """Read a ``replay_yam.npz`` (or the one in an episode dir)."""
    path = Path(path)
    if path.is_dir():
        path = path / REPLAY_YAM
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing: solve it first with `kiwi replay yam {path.parent}`"
        )
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    out["path"] = path
    out["fps"] = float(out["fps"])
    out["separation"] = float(out.get("separation", yam.DEFAULT_SEPARATION))
    out["mount_yaw"] = float(out.get("mount_yaw", 0.0))
    out["base_yaw_deg"] = tuple(float(v) for v in out.get("base_yaw_deg", (0.0, 0.0)))
    return out


def _qpos_frames(model, home: np.ndarray, ro: dict) -> np.ndarray:
    """(N, nq) model qpos from the stored per-side joints and jaw angles."""
    from kiwi.replay.solve import _resolve_indices

    n = max(len(ro[f"{s}_joints"]) for s in yam.SIDES)
    qpos = np.tile(home, (n, 1))
    for s in yam.SIDES:
        idx = _resolve_indices(model, s, yam.ARM_JOINTS)
        joints, grip = ro[f"{s}_joints"], ro[f"{s}_gripper"]
        f = np.minimum(np.arange(n), len(joints) - 1)
        qpos[:, idx["arm_qadr"]] = joints[f]
        qpos[:, idx["gripper_qadr"]] = np.stack([jaw_qpos(g) for g in grip[f]])
    return qpos


def launch(path: Path, *, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Serve the rollout at ``path`` until interrupted."""
    import mujoco
    import viser

    from kiwi.replay.mjviser import ViserMujocoScene

    ro = load_rollout(path)
    model, home = yam.build_model(
        yam.yam_dual_urdf(ro["separation"], ro["mount_yaw"], ro["base_yaw_deg"])
    )
    data = mujoco.MjData(model)
    qpos = _qpos_frames(model, home, ro)
    n, fps = len(qpos), ro["fps"]
    duration = (n - 1) / fps
    clock = np.arange(n) / fps  # the viewer's own time axis (frame / fps)
    target = {s: ro[f"{s}_target"][:, :3] for s in yam.SIDES}
    actual = {s: ro[f"{s}_actual"][:, :3] for s in yam.SIDES}
    err = {s: np.linalg.norm(target[s] - actual[s], axis=1) for s in yam.SIDES}
    saved = {
        k[len("issue_") :]: float(v) for k, v in ro.items() if k.startswith("issue_")
    }
    thr = yam.IssueThresholds(**saved) if saved else yam.IssueThresholds()

    server = viser.ViserServer(host=host, port=port, label="KIWI · YAM rollout")
    if server.get_port() != port:
        server.stop()
        raise RuntimeError(f"viewer port {port} is occupied")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(dark_mode=True, show_logo=False, show_share_button=False)
    server.initial_camera.position = (-0.9, -1.2, 1.0)
    server.initial_camera.look_at = (0.3, 0.0, 0.1)
    server.initial_camera.up_direction = (0.0, 0.0, 1.0)

    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    server.scene.add_grid(
        "/grid", width=2.0, height=2.0, cell_size=0.1, position=(0.3, 0, 0)
    )

    def segments(pts: np.ndarray) -> np.ndarray:
        return np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)

    paths = {
        s: server.scene.add_line_segments(
            f"/paths/{s}", segments(target[s]), colors=_SIDE_RGB[s], line_width=2.0
        )
        for s in yam.SIDES
    }
    markers = {
        s: server.scene.add_icosphere(
            f"/targets/{s}", radius=0.012, color=_SIDE_RGB[s], position=target[s][0]
        )
        for s in yam.SIDES
    }
    # Issue markers, shown only while flagged: a red target sphere and a red
    # line to it on an IK miss, a magenta sphere on the jumping joint, a
    # yellow sphere on the tool tip for a tip jump.
    miss_markers = {
        s: server.scene.add_icosphere(
            f"/issues/{s}/miss",
            radius=0.02,
            color=yam.ISSUE_RGB["ik_fail"],
            visible=False,
        )
        for s in yam.SIDES
    }
    joint_markers = {
        s: server.scene.add_icosphere(
            f"/issues/{s}/joint",
            radius=0.045,
            color=yam.ISSUE_RGB["joint_jump"],
            opacity=0.6,
            visible=False,
        )
        for s in yam.SIDES
    }
    tip_markers = {
        s: server.scene.add_icosphere(
            f"/issues/{s}/tip",
            radius=0.03,
            color=yam.ISSUE_RGB["tcp_jump"],
            opacity=0.6,
            visible=False,
        )
        for s in yam.SIDES
    }
    miss_lines: dict = {s: None for s in yam.SIDES}
    tcp_site = {s: model.site(f"{s}_{yam.TCP_SITE}").id for s in yam.SIDES}
    link_body = {
        s: [model.body(f"{s}_link_{j + 1}").id for j in range(len(yam.ARM_JOINTS))]
        for s in yam.SIDES
    }

    server.gui.add_markdown(
        f"**{ro['path'].parent.name} · YAM rollout**\n\n"
        f"Blue: left · Green: right. Spheres mark the commanded tool tips. "
        f"Drag to orbit; right-drag to pan; scroll to zoom."
    )
    timeline = server.gui.add_slider(
        "Time (s)", min=0.0, max=max(duration, 1e-3), step=1.0 / fps, initial_value=0.0
    )
    play = server.gui.add_checkbox("Play trajectory", initial_value=False)
    speed = server.gui.add_slider(
        "Playback speed", min=0.25, max=8.0, step=0.25, initial_value=1.0
    )
    loop = server.gui.add_checkbox("Loop", initial_value=True)
    back = server.gui.add_button("Step back")
    fwd = server.gui.add_button("Step forward")
    paths_ui = server.gui.add_checkbox("Show paths", initial_value=True)
    markers_ui = server.gui.add_checkbox("Show targets", initial_value=True)
    status = server.gui.add_markdown("")

    with server.gui.add_folder("Issues"):
        legend = server.gui.add_markdown("")
        strip_ui = server.gui.add_image(np.zeros((40, 300, 3), np.uint8))
        issue_ui = server.gui.add_dropdown("Section", options=("(none)",))
        go = server.gui.add_button("Go to section")
        prev_issue = server.gui.add_button("Previous issue")
        next_issue = server.gui.add_button("Next issue")
        ik_ui = server.gui.add_slider(
            "IK miss above (mm)",
            min=2,
            max=100,
            step=1,
            initial_value=thr.ik_error_m * 1e3,
        )
        tip_ui = server.gui.add_slider(
            "Tip jump above (m/s)",
            min=0.2,
            max=5.0,
            step=0.1,
            initial_value=thr.tcp_speed_mps,
        )
        joint_ui = server.gui.add_slider(
            "Joint jump above (rad/s)",
            min=0.5,
            max=20.0,
            step=0.5,
            initial_value=thr.joint_speed_rads,
        )
    m = {
        k[len("metric_") :]: float(v) for k, v in ro.items() if k.startswith("metric_")
    }
    if m:
        server.gui.add_markdown(
            f"Episode tracking: mean {m.get('mean_pos', 0) * 1e3:.1f} mm, "
            f"worst {m.get('worst_pos', 0) * 1e3:.1f} mm. "
            f"Bases {ro['separation']:g} m apart; gripper mount {ro['mount_yaw']:g} deg."
        )

    lock = threading.RLock()
    shown = [-1]
    issues: dict = {}

    def swatch(kind: str) -> str:
        r, g, b = yam.ISSUE_RGB[kind]
        return f'<span style="color:rgb({r},{g},{b})">&#9632;</span>'

    def diagnose() -> None:
        """(Re)compute the flags, events, strip and section list."""
        with lock:
            t = yam.IssueThresholds(
                ik_ui.value / 1e3, float(tip_ui.value), float(joint_ui.value)
            )
            diag = yam.diagnose(ro, fps, t)
            events = yam.issue_events(diag, clock)
            strip = np.asarray(
                yam.issue_strip(diag, 600, 48, bg=(40, 40, 46), ink=(220, 220, 220))
            )
            issues.update(diag=diag, events=events, strip=strip, strip_x=-1)
            counts = {k: sum(e["kind"] == k for e in events) for k in yam.ISSUES}
            legend.content = " · ".join(
                f"{swatch(k)} {yam.ISSUE_LABEL[k]}: {counts[k]}" for k in yam.ISSUES
            )
            labels = [f"{i + 1}. {yam.event_label(e)}" for i, e in enumerate(events)]
            issues["labels"] = labels
            issue_ui.options = labels or ["(none)"]
            shown[0] = -1  # redraw markers under the new thresholds
        show(frame_of(timeline.value))

    def frame_of(t: float) -> int:
        return int(np.clip(round(t * fps), 0, n - 1))

    def draw_strip(f: int) -> None:
        strip = issues["strip"]
        x = 18 + int(f / max(n - 1, 1) * (strip.shape[1] - 21))
        if abs(x - issues["strip_x"]) < 2:
            return
        img = strip.copy()
        img[:, max(x - 1, 0) : x + 1] = 255
        strip_ui.image = img
        issues["strip_x"] = x

    def show(f: int) -> None:
        with lock:
            if f == shown[0]:
                return
            shown[0] = f
            data.qpos[:] = qpos[f]
            mujoco.mj_kinematics(model, data)
            current = yam.frame_issues(issues["diag"], f)
            with server.atomic():
                scene.update_from_mjdata(data)
                lines = []
                for s in yam.SIDES:
                    g = min(f, len(target[s]) - 1)
                    markers[s].position = target[s][g]
                    kinds = {k for k, _ in current[s]}
                    miss = "ik_fail" in kinds
                    miss_markers[s].visible = miss
                    if miss_lines[s] is not None:
                        miss_lines[s].remove()
                        miss_lines[s] = None
                    if miss:
                        miss_markers[s].position = target[s][g]
                        miss_lines[s] = server.scene.add_line_segments(
                            f"/issues/{s}/miss_line",
                            np.array([[actual[s][g], target[s][g]]], np.float32),
                            colors=yam.ISSUE_RGB["ik_fail"],
                            line_width=4.0,
                        )
                    joints = [j for k, j in current[s] if k == "joint_jump"]
                    joint_markers[s].visible = bool(joints)
                    if joints:
                        worst = max(
                            joints, key=lambda j: issues["diag"][s]["joint_step"][g, j]
                        )
                        joint_markers[s].position = data.xpos[link_body[s][worst]]
                    tip_markers[s].visible = "tcp_jump" in kinds
                    tip_markers[s].position = data.site_xpos[tcp_site[s]]
                    text = f"{s} err {err[s][g] * 1e3:.1f} mm"
                    for kind, j in current[s]:
                        name = (
                            f"{yam.ARM_JOINTS[j]} jump"
                            if j is not None
                            else yam.ISSUE_LABEL[kind]
                        )
                        text += f" {swatch(kind)} {name}"
                    lines.append(text)
                status.content = (
                    f"**t = {f / fps:.2f} s** · frame {f + 1}/{n}<br>"
                    + "<br>".join(lines)
                )
                draw_strip(f)

    @timeline.on_update
    def _(_) -> None:
        show(frame_of(timeline.value))

    def step(delta: int) -> None:
        play.value = False
        timeline.value = frame_of(timeline.value + delta / fps) / fps

    back.on_click(lambda _: step(-1))
    fwd.on_click(lambda _: step(1))

    def goto_event(ev: dict) -> None:
        play.value = False
        # Land a little before the section so its onset is visible.
        timeline.value = max(0.0, ev["t0"] - 0.5)

    @go.on_click
    def _(_) -> None:
        if issue_ui.value in issues.get("labels", []):
            goto_event(issues["events"][issues["labels"].index(issue_ui.value)])

    def neighbour(direction: int) -> None:
        events = issues.get("events", [])
        t = timeline.value + 0.5  # goto_event parks 0.5 s early
        later = [e for e in events if e["t0"] > t + 1e-6]
        earlier = [e for e in events if e["t0"] < t - 1.0 / fps]
        pick = (
            (later[0] if later else None)
            if direction > 0
            else (earlier[-1] if earlier else None)
        )
        if pick is not None:
            issue_ui.value = issues["labels"][events.index(pick)]
            goto_event(pick)

    prev_issue.on_click(lambda _: neighbour(-1))
    next_issue.on_click(lambda _: neighbour(1))
    for control in (ik_ui, tip_ui, joint_ui):
        control.on_update(lambda _: diagnose())

    @paths_ui.on_update
    def _(_) -> None:
        for h in paths.values():
            h.visible = paths_ui.value

    @markers_ui.on_update
    def _(_) -> None:
        for h in markers.values():
            h.visible = markers_ui.value

    diagnose()
    print(f"READY http://localhost:{port}", flush=True)
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt, last = min(now - last, 0.2), now
            if play.value:
                t = timeline.value + dt * speed.value
                if t > duration:
                    if loop.value:
                        t = t % max(duration, 1e-3)
                    else:
                        t, play.value = duration, False
                timeline.value = t
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
