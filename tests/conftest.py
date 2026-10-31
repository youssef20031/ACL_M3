"""Shared test setup: no test reaches FPL's live feed (services/live_points.py)."""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(autouse=True)
def _no_live_fpl(monkeypatch):
    from services import live_points

    def offline(path):
        raise RuntimeError(f"tests don't call FPL ({path})")

    monkeypatch.setattr(live_points, "_http_get", offline)
    live_points.clear_cache()
    yield
    live_points.clear_cache()
