"""Console encoding helpers."""
import sys


def ensure_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 so emoji in prints and logs can't crash the process.

    On Windows, output redirected to a file or pipe uses the ANSI code page (cp1252),
    which can't encode emoji: a print("🚀 ...") during startup raised UnicodeEncodeError
    and uvicorn exited. Streams that are already UTF-8 (Linux, Docker) are left alone;
    characters that still can't be written are replaced instead of raising.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # detached or non-reconfigurable stream: keep the default
