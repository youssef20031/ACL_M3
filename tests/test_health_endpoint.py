"""/health must answer 200 in every startup state: the frontend reads Neo4j status from it."""
import asyncio
import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api_main  # noqa: E402


def _state(**overrides):
    state = dict(api_main.app_state)
    state.update(
        neo4j_conn=None,
        llm_manager=None,
        embedding_manager=None,
        embeddings_built=False,
        embedding_building=False,
        embedding_build_error=None,
        ml_integration=None,
    )
    state.update(overrides)
    return state


def _health(state):
    with patch.dict(api_main.app_state, state, clear=True):
        return asyncio.run(api_main.health_check())


def test_health_without_ml_or_neo4j():
    body = _health(_state())
    assert body["status"] == "healthy"
    assert body["neo4j"] == "disconnected"
    assert body["ml_available"] is False


def test_health_reports_ml_and_model_label():
    ml = Mock(predictor_loaded=True, predictors={"FWD": object()})
    with patch.object(api_main, "get_predictions_df", return_value=None):
        body = _health(_state(ml_integration=ml))
    assert body["ml_available"] is True
    assert body["ml_model_type"] == api_main.ML_FALLBACK_LABEL == "Backup model"


def test_health_prefers_v3_predictions_file():
    with patch.object(api_main, "get_predictions_df", return_value=[{"name": "x"}]):
        body = _health(_state())
    assert body["ml_model_type"] == api_main.ML_MODEL_LABEL
    assert "v3" not in body["ml_model_type"].lower()
