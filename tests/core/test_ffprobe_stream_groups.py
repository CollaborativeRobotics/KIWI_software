"""ffprobe 7+ repeats stream-level entries under each stream group.

Insta360 X5 ``.insv`` and Ace Pro 2 ``.mp4`` both carry a stream group, so
``-show_entries stream=...`` prints every value twice. A single-lens original
must still count as one video stream, and the tmcd base rate must still parse.
ffprobe is monkeypatched out - the parsing is what is under test.

Run: pixi run test
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kiwi.core.camera import convert as convert_mod
from kiwi.sync import tmcd


def _fake_run(stdout):
    def run(cmd, **kwargs):
        out = stdout if kwargs.get("text") else stdout.encode()
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr=b"")

    return run


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("0\n\n0\n", 1),  # Ace Pro 2 .mp4 on ffprobe 9: the stream, then its group
        ("0\n1\n\n0\n1\n", 2),  # X5 .insv on ffprobe 9
        ("0\n", 1),  # older ffprobe, no stream groups
    ],
)
def test_video_streams_are_counted_once_each(monkeypatch, stdout, expected):
    monkeypatch.setattr(convert_mod.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(convert_mod.subprocess, "run", _fake_run(stdout))
    assert convert_mod.count_video_streams(Path("clip.mp4")) == expected


def test_a_repeated_tmcd_base_rate_still_parses(monkeypatch):
    outputs = iter(["12:56:06:29\n", "30000/1001\n30000/1001\n"])
    monkeypatch.setattr(
        tmcd.subprocess, "run", lambda cmd, **kw: _fake_run(next(outputs))(cmd, **kw)
    )
    tc, base = tmcd.read_tmcd(Path("clip.mp4"))
    assert tc == "12:56:06:29"
    assert base == pytest.approx(30000 / 1001)


def test_disagreeing_repeats_are_refused(monkeypatch):
    monkeypatch.setattr(tmcd.subprocess, "run", _fake_run("25/1\n30000/1001\n"))
    with pytest.raises(RuntimeError, match="ambiguous"):
        tmcd._ffprobe(
            ["-show_entries", "stream=avg_frame_rate"], Path("c.mp4"), "ffprobe"
        )
