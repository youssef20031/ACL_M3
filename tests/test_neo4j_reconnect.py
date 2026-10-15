"""The backend must recover when Neo4j comes up after it (e.g. both restarted after a VM reboot)."""
import asyncio
import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api_main  # noqa: E402


async def _no_sleep(_delay):
    return None


def _run_reconnect(attempts, state=None):
    """Run the reconnect loop with `_try_connect_neo4j` returning `attempts` in turn."""
    state = {**api_main.app_state, "neo4j_conn": None, "ml_integration": None, **(state or {})}
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    with patch.dict(api_main.app_state, state, clear=True), \
            patch.object(api_main, "_try_connect_neo4j", side_effect=attempts) as try_connect, \
            patch.object(api_main, "get_player_index", return_value=[1, 2, 3]) as build_index, \
            patch.object(api_main, "reset_graph_caches") as reset, \
            patch.object(api_main.asyncio, "sleep", record_sleep):
        asyncio.run(api_main._reconnect_neo4j_until_connected())
        return dict(api_main.app_state), sleeps, try_connect, build_index, reset


def test_retries_with_backoff_until_neo4j_answers():
    conn = Mock()
    ml = Mock()
    final, sleeps, try_connect, build_index, reset = _run_reconnect(
        [None, None, None, conn], state={"ml_integration": ml, "entity_extractor": Mock()})

    assert final["neo4j_conn"] is conn
    assert try_connect.call_count == 4
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    build_index.assert_called_once_with(conn)
    reset.assert_called_once()
    assert ml.neo4j_conn is conn  # ML was built with the startup value (None)
    assert ml.fixture_service.conn is conn
    assert final["entity_extractor"] is None  # rebuilt lazily with player names


def test_backoff_is_capped():
    conn = Mock()
    _, sleeps, *_ = _run_reconnect([None] * 6 + [conn])
    assert max(sleeps) == api_main.NEO4J_RETRY_MAX_DELAY
    assert sleeps[-2:] == [30.0, 30.0]


def test_keeps_a_connection_made_meanwhile():
    """If the admin connect endpoint attached a connection during the wait, don't replace it."""
    existing = Mock()
    late = Mock()

    async def connect_meanwhile(_delay):
        api_main.app_state["neo4j_conn"] = existing

    state = {**api_main.app_state, "neo4j_conn": None, "ml_integration": None}
    with patch.dict(api_main.app_state, state, clear=True), \
            patch.object(api_main, "_try_connect_neo4j", return_value=late) as try_connect, \
            patch.object(api_main, "get_player_index") as build_index, \
            patch.object(api_main.asyncio, "sleep", connect_meanwhile):
        asyncio.run(api_main._reconnect_neo4j_until_connected())
        assert api_main.app_state["neo4j_conn"] is existing
    try_connect.assert_not_called()  # the loop exits before trying
    build_index.assert_not_called()
    late.close.assert_not_called()


def test_discards_its_own_connection_if_another_won_the_race():
    existing = Mock()
    late = Mock()

    def connect_and_lose_race():
        api_main.app_state["neo4j_conn"] = existing
        return late

    state = {**api_main.app_state, "neo4j_conn": None, "ml_integration": None}
    with patch.dict(api_main.app_state, state, clear=True), \
            patch.object(api_main, "_try_connect_neo4j", side_effect=connect_and_lose_race), \
            patch.object(api_main, "get_player_index") as build_index, \
            patch.object(api_main.asyncio, "sleep", _no_sleep):
        asyncio.run(api_main._reconnect_neo4j_until_connected())
        assert api_main.app_state["neo4j_conn"] is existing
    late.close.assert_called_once()
    build_index.assert_not_called()


def test_try_connect_returns_none_when_neo4j_is_down():
    with patch.object(api_main, "Neo4jConnection", side_effect=OSError("connection refused")):
        assert api_main._try_connect_neo4j() is None

    unhealthy = Mock()
    unhealthy.test_connection.return_value = False
    with patch.object(api_main, "Neo4jConnection", return_value=unhealthy):
        assert api_main._try_connect_neo4j() is None
    unhealthy.close.assert_called_once()


def test_attach_survives_a_failed_player_index_build():
    conn = Mock()
    state = {**api_main.app_state, "neo4j_conn": None, "ml_integration": None}
    with patch.dict(api_main.app_state, state, clear=True), \
            patch.object(api_main, "get_player_index", side_effect=RuntimeError("boom")), \
            patch.object(api_main, "reset_graph_caches"):
        asyncio.run(api_main._attach_neo4j(conn))
        assert api_main.app_state["neo4j_conn"] is conn
