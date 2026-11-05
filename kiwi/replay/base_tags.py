"""Robot base tags: place the YAM bases on two located table tags.

:func:`base_placement` turns two tags into the YAM rig placement: each base
on its tag's centre, on the plane through the tags, the arms on the side of
the line between them where the recorded work is, each yawed to face its own
tool-tip workspace. Up is the world +Z (the scene frame is gravity aligned),
not the tag normals, which are the noisiest part of a small-tag pose.

``kiwi replay yam --base-tags 52 53`` locates the tags
(:func:`kiwi.app.base_tags.find_base_tags`), caches them in
``<episode>/base_tags.json`` and places the arms on them; ``kiwi viz play``
draws the tags it finds there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kiwi.core import layout

__all__ = ["base_placement", "load_base_tags", "save_base_tags"]


def save_base_tags(episode: Path, found: dict) -> Path:
    path = Path(episode) / layout.BASE_TAGS
    path.write_text(json.dumps(found, indent=2) + "\n")
    return path


def load_base_tags(episode: Path) -> dict | None:
    path = Path(episode) / layout.BASE_TAGS
    return json.loads(path.read_text()) if path.is_file() else None


def base_placement(found: dict, trajs: dict) -> dict:
    """The YAM rig placement that puts the bases on the two tags in ``found``.

    ``trajs`` maps ``left``/``right`` to (N, >=3) tool-tip paths in the world
    frame. Returns the rig's workspace transform ``ws`` (``p_rig = R_ws
    p_world + anchor``), the base ``separation``, each base's yaw about its
    own axis (``base_yaw_deg``, left then right) and which tag is which arm.
    """
    from scipy.spatial.transform import Rotation

    tags = [t for t in found["tags"] if t.get("position") is not None]
    if len(tags) != 2:
        seen = [t["id"] for t in tags]
        raise ValueError(f"need two located base tags, found {seen or 'none'}")
    up = np.array([0.0, 0.0, 1.0])
    a, b = (np.asarray(t["position"], np.float64) for t in tags)
    work = {
        s: np.asarray(trajs[s], np.float64)[:, :3].mean(axis=0)
        for s in ("left", "right")
    }
    mid = (a + b) / 2
    y = (a - b) - up * ((a - b) @ up)
    separation = float(np.linalg.norm(y))
    y /= separation
    x = np.cross(y, up)
    if x @ (np.mean(list(work.values()), axis=0) - mid) < 0:
        # Face the arms toward the work: the left base is then tag b.
        tags, x, y = tags[::-1], -x, -y
    R = np.stack([x, y, up])  # rows: rig axes in world
    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    anchor = -R @ mid
    yaws = []
    for side, base_y in (("left", separation / 2), ("right", -separation / 2)):
        d = R @ (work[side] - mid) - np.array([0.0, base_y, 0.0])
        yaws.append(float(np.degrees(np.arctan2(d[1], d[0]))))
    near = {
        s: min(tags, key=lambda t: np.linalg.norm(np.asarray(t["position"]) - work[s]))[
            "id"
        ]
        for s in ("left", "right")
    }
    return {
        "ws": {
            "x": float(anchor[0]),
            "y": float(anchor[1]),
            "z": float(anchor[2]),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
        },
        "separation": separation,
        "base_yaw_deg": tuple(yaws),
        "left_tag": tags[0]["id"],
        "right_tag": tags[1]["id"],
        # Each arm's work is nearer its own base unless the hands crossed.
        "crossed": near["left"] != tags[0]["id"] or near["right"] != tags[1]["id"],
    }
