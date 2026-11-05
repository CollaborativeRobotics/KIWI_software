"""Locate the robot base tags (table ArUco markers) from the wrist cameras.

The wrist cameras are the only cameras posed in the episode's world, so the
tags are found in their four lens videos (front and rear of each arm). Each
detection gives the tag's pose in that lens (fisheye-undistorted corners,
``SOLVEPNP_IPPE_SQUARE``), and the lens's world pose at that frame comes from
the arm's ``scene/state.tum`` (the front lens; the rear goes through the
unit's ``T_rear_front``). A tag's world centre is the median over its
detections and its normal the mean, after dropping centres more than
``outlier_m`` from the median. :mod:`kiwi.replay.base_tags` places the YAM
bases on the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kiwi.core import layout

__all__ = ["DICTIONARY", "find_base_tags"]

DICTIONARY = "DICT_4X4_100"
LENS_VIDEO = {"front": layout.LENS0, "rear": layout.LENS1}


def _detections(
    video: Path, kb4: np.ndarray, ids: set[int], stride: int, min_px: float
):
    """``(frame index, tag id, (4, 2) undistorted normalized corners)`` per hit."""
    import cv2

    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)),
        cv2.aruco.DetectorParameters(),
    )
    fx, fy, cx, cy = kb4[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.asarray(kb4[4:8], np.float64)
    cap = cv2.VideoCapture(str(video))
    out, i = [], -1
    while True:
        i += 1
        if i % stride:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        corners, found, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if found is None:
            continue
        for c, k in zip(corners, found.ravel()):
            c = c.reshape(4, 2)
            if int(k) not in ids or np.linalg.norm(c[0] - c[2]) < min_px * np.sqrt(2):
                continue
            n = cv2.fisheye.undistortPoints(c.reshape(-1, 1, 2), K, D).reshape(4, 2)
            out.append((i, int(k), n))
    cap.release()
    return out


def _pnp(norm: np.ndarray, size: float) -> np.ndarray | None:
    """``T_cam_tag`` from normalized corners (ArUco order, tag z out of the face)."""
    import cv2

    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj, norm.astype(np.float64), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or tvec[2, 0] <= 0:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def find_base_tags(
    episode: Path,
    ids: tuple[int, ...],
    size: float,
    *,
    stride: int = 4,
    min_px: float = 15.0,
    outlier_m: float = 0.05,
) -> dict:
    """World poses of the tags ``ids`` (``size`` m edge) seen by the wrist cameras."""
    import cv2
    from scipy.spatial.transform import Rotation

    from kiwi.core.trajectory import load_tum, se3_stack
    from kiwi.episode.manifest import MANIFEST_NAME, EpisodeConfig
    from kiwi.episode.scene_build import load_unit
    from kiwi.state.geometry import interpolate_poses, se3_inv
    from kiwi.track.frames import rescale_kb4
    from kiwi.track.sequence import frame_times_ns

    episode = Path(episode)
    cfg = EpisodeConfig.load(episode / MANIFEST_NAME)
    hits = {k: [] for k in ids}  # id -> [(T_world_tag, lens)]
    for arm in ("left", "right"):
        unit = load_unit(cfg.arms[arm])
        offset = float(cfg.sync_offsets[arm])
        t_s, p, q, _ = load_tum(layout.scene_dir(episode, arm) / layout.STATE_TUM)
        t_s = np.asarray(t_s, np.float64)
        T_s = se3_stack(p, q)
        for lens, name in LENS_VIDEO.items():
            video = layout.stream_dir(episode, arm) / name
            pts = np.asarray(frame_times_ns(video), np.int64)
            cap = cv2.VideoCapture(str(video))
            wh = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            cap.release()
            kb4 = rescale_kb4(unit.kb4[lens], unit.resolution[lens], wh)
            T_lens_front = np.eye(4) if lens == "front" else unit.T_rear_front()
            for i, k, norm in _detections(video, kb4, set(ids), stride, min_px):
                t = pts[i] / 1e9 + unit.timeshift_cam_imu[lens] + offset
                if not t_s[0] <= t <= t_s[-1]:
                    continue
                T_cam_tag = _pnp(norm, size)
                if T_cam_tag is None:
                    continue
                T_wc = interpolate_poses(t_s, T_s, np.array([t]))[0] @ se3_inv(
                    T_lens_front
                )
                hits[k].append((T_wc @ T_cam_tag, f"{arm}_{lens}"))

    tags = {}
    for k, obs in hits.items():
        if not obs:
            tags[k] = {"id": k, "observations": 0}
            continue
        T = np.stack([o[0] for o in obs])
        centre = np.median(T[:, :3, 3], axis=0)
        keep = np.linalg.norm(T[:, :3, 3] - centre, axis=1) <= outlier_m
        if keep.sum() >= 3:
            centre = np.median(T[keep, :3, 3], axis=0)
        T = T[keep] if keep.any() else T
        kept = [o for o, m in zip(obs, keep) if m] or obs
        normal = T[:, :3, 2].mean(axis=0)
        normal /= np.linalg.norm(normal)
        x = T[:, :3, 0].mean(axis=0)
        x -= normal * (x @ normal)
        x /= np.linalg.norm(x)
        R = np.c_[x, np.cross(normal, x), normal]
        lenses: dict[str, int] = {}
        for o in kept:
            lenses[o[1]] = lenses.get(o[1], 0) + 1
        tags[k] = {
            "id": k,
            "observations": len(obs),
            "inliers": len(kept),
            "lenses": lenses,
            "position": centre.tolist(),
            "normal": normal.tolist(),
            "quaternion_xyzw": Rotation.from_matrix(R).as_quat().tolist(),
            "spread_m": float(np.median(np.linalg.norm(T[:, :3, 3] - centre, axis=1))),
        }
    return {
        "schema": "kiwi_base_tags_v1",
        "dictionary": DICTIONARY,
        "size": size,
        "stride": stride,
        "tags": [tags[k] for k in ids],
    }
