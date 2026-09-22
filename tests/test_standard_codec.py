from types import SimpleNamespace

import numpy as np
import subprocess

from src.codecs.standard import StandardCodec


def test_ffmpeg_never_consumes_parent_shell_stdin(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "pipe:0" in command:
            # Materialise the elementary stream expected by the codec.
            with open(command[-1], "wb") as stream:
                stream.write(b"encoded")
            return SimpleNamespace(stdout=b"", stderr=b"")
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout="I\n", stderr="")
        return SimpleNamespace(stdout=bytes(2 * 2 * 3), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clip = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    StandardCodec("h264", 35)._encode_decode_clip(clip)

    assert len(calls) == 3
    ffmpeg_calls = [call for call in calls if call[0][0] == "ffmpeg"]
    assert all("-nostdin" in command for command, _ in ffmpeg_calls)
    assert calls[1][1]["stdin"] is subprocess.DEVNULL  # ffprobe
    assert calls[2][1]["stdin"] is subprocess.DEVNULL  # decoder


def test_picture_type_metadata_is_returned(monkeypatch):
    def fake_run(command, **kwargs):
        if "pipe:0" in command:
            with open(command[-1], "wb") as stream:
                stream.write(b"encoded")
            return SimpleNamespace(stdout=b"", stderr=b"")
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout="I\nB\nP\n", stderr="")
        return SimpleNamespace(stdout=bytes(3 * 2 * 2 * 3), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clip = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    _, _, picture_types = StandardCodec("h264", 35)._encode_decode_clip_with_metadata(clip)
    assert picture_types == [0, 2, 1]
