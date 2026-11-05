import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kiwi.replay.base_tags import base_placement


def _found(*positions):
    return {
        "size": 0.05,
        "tags": [
            {"id": 50 + i, "position": list(p), "quaternion_xyzw": [0, 0, 0, 1]}
            for i, p in enumerate(positions)
        ],
    }


def _to_rig(ws, p):
    R = Rotation.from_euler("ZYX", [ws["yaw"], ws["pitch"], ws["roll"]], degrees=True)
    return R.as_matrix() @ np.asarray(p) + [ws["x"], ws["y"], ws["z"]]


def test_the_bases_land_on_the_tags_with_the_work_in_front():
    # Tags 1 m apart along world X at table height 0.3; the hands work at +Y.
    found = _found((1.0, 0.0, 0.3), (0.0, 0.0, 0.3))
    trajs = {
        "left": np.array([[0.2, 0.4, 0.5]]),
        "right": np.array([[0.8, 0.4, 0.5]]),
    }
    place = base_placement(found, trajs)
    ws, s = place["ws"], place["separation"]
    assert s == pytest.approx(1.0)
    left = next(t for t in found["tags"] if t["id"] == place["left_tag"])
    right = next(t for t in found["tags"] if t["id"] == place["right_tag"])
    np.testing.assert_allclose(_to_rig(ws, left["position"]), [0, s / 2, 0], atol=1e-9)
    np.testing.assert_allclose(
        _to_rig(ws, right["position"]), [0, -s / 2, 0], atol=1e-9
    )
    # The work is in front of the bases (+X in the rig) and above them.
    work = _to_rig(ws, [0.5, 0.4, 0.5])
    assert work[0] > 0 and work[2] == pytest.approx(0.2)
    assert not place["crossed"]


def test_each_base_turns_toward_its_own_hand():
    found = _found((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Each hand works straight in front of its own base: no turn needed.
    place = base_placement(
        found,
        {"left": np.array([[0.0, 0.5, 0.2]]), "right": np.array([[1.0, 0.5, 0.2]])},
    )
    np.testing.assert_allclose(place["base_yaw_deg"], [0.0, 0.0], atol=1e-9)
    # Both hands work midway between the bases: the arms turn inward.
    place = base_placement(
        found,
        {"left": np.array([[0.5, 0.5, 0.2]]), "right": np.array([[0.5, 0.5, 0.2]])},
    )
    left, right = place["base_yaw_deg"]
    assert left == pytest.approx(-45.0) and right == pytest.approx(45.0)


def test_a_missing_tag_is_an_error():
    found = _found((0.0, 0.0, 0.0))
    found["tags"].append({"id": 99, "observations": 0})
    with pytest.raises(ValueError, match="two located base tags"):
        base_placement(found, {"left": np.zeros((1, 3)), "right": np.zeros((1, 3))})
