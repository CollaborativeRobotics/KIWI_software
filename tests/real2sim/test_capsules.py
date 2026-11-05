"""Robot capsules composite into a splat render, hidden behind nearer scene."""

import math

import numpy as np

from kiwi.real2sim.overlay import composite_capsules

# OpenCV camera at the origin looking down +Z; one capsule 1 m ahead.
POSE = np.eye(4)
ENDS = np.array([[[-0.2, 0.0, 1.0], [0.2, 0.0, 1.0]]])
RADIUS = np.array([0.03])
COLOR = np.array([[255, 150, 0]], dtype=np.uint8)


def _draw(depth):
    rgb = np.zeros((120, 160, 3), np.uint8)
    return composite_capsules(rgb, depth, POSE, math.radians(60), ENDS, RADIUS, COLOR)


def test_capsule_draws_over_empty_scene():
    out = _draw(None)
    assert out[60, 80].sum() > 0  # the axis projects to the image centre
    assert out[5, 5].sum() == 0


def test_nearer_scene_hides_capsule():
    assert _draw(np.full((120, 160), 0.5, np.float32))[60, 80].sum() == 0
    assert _draw(np.full((120, 160), 2.0, np.float32))[60, 80].sum() > 0
