"""ensure_utf8_stdio: emoji prints must not crash when stdout is cp1252 (Windows redirect)."""
import io
import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.console import ensure_utf8_stdio  # noqa: E402


def _cp1252_stream():
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", write_through=True)


def test_emoji_print_crashes_on_cp1252_without_fix():
    _, stream = _cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        print("\U0001f680 API starting", file=stream)


def test_reconfigures_cp1252_stdout_and_stderr_to_utf8():
    out_raw, out = _cp1252_stream()
    err_raw, err = _cp1252_stream()
    with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
        ensure_utf8_stdio()
        print("\U0001f680 API starting")
        print("✅ ready", file=sys.stderr)
    assert out_raw.getvalue().decode("utf-8") == "\U0001f680 API starting\n".replace("\n", os.linesep)
    assert "✅ ready" in err_raw.getvalue().decode("utf-8")


def test_leaves_utf8_streams_alone():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", write_through=True)
    with patch.object(sys, "stdout", stream), patch.object(sys, "stderr", stream):
        ensure_utf8_stdio()
    assert stream.errors == "strict"


def test_ignores_streams_without_reconfigure():
    plain = io.StringIO()  # no .reconfigure on StringIO
    with patch.object(sys, "stdout", plain), patch.object(sys, "stderr", plain):
        ensure_utf8_stdio()  # must not raise
