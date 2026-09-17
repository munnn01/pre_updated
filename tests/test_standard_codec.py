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
        return SimpleNamespace(stdout=bytes(2 * 2 * 3), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clip = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    StandardCodec("h264", 35)._encode_decode_clip(clip)

    assert len(calls) == 2
    assert all("-nostdin" in command for command, _ in calls)
    assert calls[1][1]["stdin"] is subprocess.DEVNULL
