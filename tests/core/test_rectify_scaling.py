"""The rectifier rescales a calibration to the frames it is applied to."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from kiwi.core.rectify import IntrinsicsBundle, load_rectifier_auto

CAL = (480, 480)
DS = np.array([-0.004, 0.58, 129.0, 129.0, 239.6, 240.1])  # [xi alpha fx fy cx cy]
K = np.array([[130.0, 0.0, 239.6], [0.0, 130.0, 240.1], [0.0, 0.0, 1.0]])
D = np.array([0.04, -0.0003, -0.0003, -0.0001])


def _bundle(kind: str) -> IntrinsicsBundle:
    if kind == "ds":
        return IntrinsicsBundle(None, None, None, None, CAL, ds_front=DS)
    return IntrinsicsBundle(K, D, None, None, CAL)


def _scene(size: tuple[int, int]) -> np.ndarray:
    """A smooth pattern, so resizing barely changes it."""
    w, h = size
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img = 127 + 60 * np.sin(6 * np.pi * u) + 60 * np.cos(4 * np.pi * v)
    return img.astype(np.uint8)


@pytest.mark.parametrize("kind", ["ds", "equi"])
def test_downscaled_source_rectifies_like_full_size(kind):
    bundle = _bundle(kind)
    out = (200, 200)
    full = _scene(CAL)
    half = cv2.resize(full, (240, 240), interpolation=cv2.INTER_AREA)

    ref = bundle.rectifier_for("front", out, 100.0).apply(full)
    got = bundle.rectifier_for("front", out, 100.0, src_size=(240, 240)).apply(half)
    unscaled = bundle.rectifier_for("front", out, 100.0).apply(half)

    err = np.abs(got.astype(float) - ref).mean()
    assert err < 3.0
    # without the rescale the half-size frame samples the wrong pixels
    assert np.abs(unscaled.astype(float) - ref).mean() > 5 * err


def test_same_size_is_unchanged():
    bundle = _bundle("ds")
    assert bundle.scaled_to(CAL) is bundle


def test_pinhole_part_scales_angles_kept():
    s = _bundle("ds").scaled_to((240, 240))
    np.testing.assert_allclose(s.ds_front[:2], DS[:2])
    np.testing.assert_allclose(s.ds_front[2:], DS[2:] / 2)
    e = _bundle("equi").scaled_to((960, 960))
    np.testing.assert_allclose(e.K_front[:2, :], K[:2, :] * 2)
    assert e.image_size == (960, 960)


def test_aspect_change_is_refused():
    with pytest.raises(ValueError, match="different crop"):
        _bundle("ds").scaled_to((480, 270))


def test_load_rectifier_auto_uses_src_size(tmp_path):
    npz = tmp_path / "intrinsics.npz"
    np.savez(npz, image_size=np.array(CAL, dtype=np.int32), ds_front=DS)
    full = _scene(CAL)
    half = cv2.resize(full, (240, 240), interpolation=cv2.INTER_AREA)
    ref = load_rectifier_auto(npz, CAL, out_size=(200, 200), fov_deg=100.0)
    got = load_rectifier_auto(npz, (240, 240), out_size=(200, 200), fov_deg=100.0)
    assert np.abs(got.apply(half).astype(float) - ref.apply(full)).mean() < 3.0
