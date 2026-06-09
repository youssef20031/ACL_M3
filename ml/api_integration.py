"""
API Integration for ML Predictions
Extends the FastAPI backend with ML prediction endpoints
"""
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from ml.predictor import FPLPredictor, PlayerPrediction

logger = logging.getLogger(__name__)


# Pydantic models for API
class PlayerPredictionRequest(BaseModel):
    """Request for single player prediction."""
    player_name: str
    player_data: Dict[str, Any] = {}  # Optional; fetched from Neo4j if empty


class TopPerformersRequest(BaseModel):
    """Request for top performers prediction."""
    position: Optional[str] = None
    top_k: int = 10
    season: Optional[str] = None


class BestValueRequest(BaseModel):
    """Request for best value players."""
    position: Optional[str] = None
    max_price: Optional[float] = None
    top_k: int = 10


class PredictionResponse(BaseModel):
    """Response for predictions."""
    player_name: str
    predicted_points: float
    confidence_interval: Optional[List[float]] = None
    features_used: Optional[Dict[str, Any]] = None
    model_version: str = "v1"


class TopPerformersResponse(BaseModel):
    """Response for top performers."""
    predictions: List[PredictionResponse]
    metadata: Dict[str, Any]


class MLAPIIntegration:
    """
    Integration layer for ML predictions in the API.
    Manages predictor lifecycle and query execution.
    """
    
    def __init__(self, neo4j_conn, query_executor):
        """
        Initialize ML API integration.
        
        Args:
            neo4j_conn: Neo4j connection instance
            query_executor: Query executor for fetching player data
        """
        self.neo4j_conn = neo4j_conn
        self.query_executor = query_executor
        self.predictor = None
        self.predictor_loaded = False
        
    def load_predictor(self, model_path: str = "ml/models/linear_regression_v1.pkl"):
        """
        Load ML predictor on API startup.
        
        Args:
            model_path: Path to trained model
        """
        try:
            self.predictor = FPLPredictor(model_path=model_path, model_type="linear")
            self.predictor_loaded = True
            logger.info("✅ ML Predictor loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ML predictor: {e}")
            logger.warning("ML prediction endpoints will return errors")
    
    def _check_predictor(self):
        """Ensure predictor is loaded."""
        if not self.predictor_loaded or not self.predictor:
            raise HTTPException(
                status_code=503,
                detail="ML predictor not loaded. Train a model first."
            )
    
    async def get_player_data_for_prediction(
        self,
        player_name: str,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch player data from Neo4j for prediction.

        Stats are stored as properties on the :PLAYED_IN relationship.
        Position is stored on the :PLAYS_POSITION -> Position node.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[:PLAYS_POSITION]->(pos:Position)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
        WHERE ($season IS NULL OR s.id = $season)
          AND r.minutes > 0
        WITH p, pos, r, f
        ORDER BY f.kickoff_time DESC
        LIMIT 4
        RETURN
            p.name                  AS name,
            pos.code                AS position,
            AVG(r.minutes)          AS minutes,
            AVG(r.goals_scored)     AS goals_scored,
            AVG(r.assists)          AS assists,
            AVG(r.total_points)     AS form,
            AVG(r.bps)              AS bps,
            AVG(r.ict_index)        AS ict_index,
            AVG(r.influence)        AS influence,
            AVG(r.creativity)       AS creativity,
            AVG(r.threat)           AS threat,
            SUM(r.clean_sheets)     AS clean_sheets,
            SUM(r.bonus)            AS bonus,
            SUM(r.goals_conceded)   AS goals_conceded,
            SUM(r.saves)            AS saves,
            SUM(r.yellow_cards)     AS yellow_cards,
            SUM(r.red_cards)        AS red_cards,
            AVG(r.value)            AS value
        """

        results = self.neo4j_conn.execute_query(
            query,
            {"player_name": player_name, "season": season}
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Player '{player_name}' not found or no recent data available"
            )

        return results[0]
    
    @staticmethod
    def _prepare_for_inference(player_data: dict) -> dict:
        """
        Fill in columns that the FeatureEngineer pipeline expects but that
        are not returned by the aggregated Neo4j query.

        The feature engineer was designed for per-gameweek rows; during
        inference we feed a single aggregated row, so we just set sentinel
        values so that sort/groupby operations don't crash.
        """
        defaults = {
            "kickoff_time": "2000-01-01 00:00:00",  # arbitrary past date for sorting
            "was_home": 0,
            "GW": 19,                                # mid-season default
            "total_points": player_data.get("form", 0),  # form already is rolling avg
        }
        return {**defaults, **player_data}

    async def predict_player_next_gameweek(
        self,
        request: PlayerPredictionRequest
    ) -> PredictionResponse:
        """
        Predict points for a single player's next gameweek.

        Args:
            request: Prediction request

        Returns:
            Prediction response
        """
        self._check_predictor()

        try:
            # Get player data if not provided
            if not request.player_data:
                player_data = await self.get_player_data_for_prediction(request.player_name)
            else:
                player_data = dict(request.player_data)

            # Fill sentinel columns required by the feature engineering pipeline
            player_data = self._prepare_for_inference(player_data)

            # Predict
            prediction = self.predictor.predict_next_gameweek(player_data)

            return PredictionResponse(
                player_name=prediction.player_name,
                predicted_points=prediction.predicted_points,
                features_used=prediction.features_used,
                model_version=prediction.model_version
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def predict_top_performers(
        self,
        request: TopPerformersRequest
    ) -> TopPerformersResponse:
        """
        Predict top K performers for next gameweek.

        Args:
            request: Top performers request

        Returns:
            List of predictions
        """
        self._check_predictor()

        try:
            # Build optional filters — use parameterised query to avoid injection
            position_clause = "AND pos.code = $position" if request.position else ""
            season_clause   = "AND s.id = $season"      if request.season    else ""

            query = f"""
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            WHERE r.minutes > 0
              {position_clause}
              {season_clause}
            WITH p, pos, r, f
            ORDER BY f.kickoff_time DESC
            WITH p, pos, COLLECT(r)[..4] AS recent_perfs
            WHERE SIZE(recent_perfs) >= 2
            UNWIND recent_perfs AS perf
            RETURN
                p.name                                                   AS name,
                pos.code                                                 AS position,
                AVG(perf.minutes)                                        AS minutes,
                AVG(perf.goals_scored)                                   AS goals_scored,
                AVG(perf.assists)                                        AS assists,
                AVG(perf.total_points)                                   AS form,
                AVG(perf.bps)                                            AS bps,
                AVG(perf.ict_index)                                      AS ict_index,
                AVG(perf.influence)                                      AS influence,
                AVG(perf.creativity)                                     AS creativity,
                AVG(perf.threat)                                         AS threat,
                SUM(perf.clean_sheets)                                   AS clean_sheets,
                SUM(perf.bonus)                                          AS bonus,
                AVG(perf.value)                                          AS value
            LIMIT 200
            """

            params = {
                "position": request.position,
                "season":   request.season,
            }
            players_data = self.neo4j_conn.execute_query(query, params)

            if not players_data:
                raise HTTPException(
                    status_code=404,
                    detail="No player data found for prediction"
                )

            # Prepare for inference (adds total_points, etc.)
            players_data = [self._prepare_for_inference(p) for p in players_data]

            # Predict
            predictions = self.predictor.predict_top_performers(
                players_data,
                position=request.position,
                top_k=request.top_k
            )

            # Convert to response format
            response_predictions = [
                PredictionResponse(
                    player_name=pred.player_name,
                    predicted_points=pred.predicted_points,
                    features_used=pred.features_used,
                    model_version=pred.model_version
                )
                for pred in predictions
            ]

            return TopPerformersResponse(
                predictions=response_predictions,
                metadata={
                    "position_filter": request.position,
                    "total_players_analyzed": len(players_data),
                    "top_k": request.top_k
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Top performers prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def predict_best_value(
        self,
        request: BestValueRequest
    ) -> List[Dict[str, Any]]:
        """
        Predict best value players (points per million).

        Args:
            request: Best value request

        Returns:
            List of best value players
        """
        self._check_predictor()

        try:
            position_clause = "AND pos.code = $position" if request.position else ""
            # value in graph is stored as integer * 10 (e.g. 85 = £8.5m)
            # We filter AFTER UNWINDING to correctly calculate averages
            price_filter = (
                f"WITH name, position, minutes, goals_scored, assists, form, bps, ict_index, value WHERE value <= {request.max_price * 10}"
                if request.max_price else ""
            )

            query = f"""
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.minutes > 0
              {position_clause}
            WITH p, pos, r, f
            ORDER BY f.kickoff_time DESC
            WITH p, pos, COLLECT(r)[..4] AS recent_perfs
            WHERE SIZE(recent_perfs) >= 2
            UNWIND recent_perfs AS perf
            WITH 
                p.name AS name, 
                pos.code AS position, 
                AVG(perf.minutes) AS minutes,
                AVG(perf.goals_scored) AS goals_scored,
                AVG(perf.assists) AS assists,
                AVG(perf.total_points) AS form,
                AVG(perf.bps) AS bps,
                AVG(perf.ict_index) AS ict_index,
                AVG(perf.value) AS value
            {price_filter}
            RETURN name, position, minutes, goals_scored, assists, form, bps, ict_index, value
            LIMIT 200
            """

            params = {"position": request.position}
            players_data = self.neo4j_conn.execute_query(query, params)

            if not players_data:
                return []

            # Prepare for inference
            players_data = [self._prepare_for_inference(p) for p in players_data]

            # Predict best value
            results = self.predictor.predict_best_value(
                players_data,
                position=request.position,
                top_k=request.top_k
            )

            return results

        except Exception as e:
            logger.error(f"Best value prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# FastAPI route registration helper
def register_ml_routes(app, ml_integration: MLAPIIntegration):
    """
    Register ML prediction routes to FastAPI app.
    
    Args:
        app: FastAPI application
        ml_integration: ML API integration instance
    """
    
    @app.post("/api/ml/predict/player", response_model=PredictionResponse)
    async def predict_player(request: PlayerPredictionRequest):
        """Predict next gameweek points for a specific player."""
        return await ml_integration.predict_player_next_gameweek(request)
    
    @app.post("/api/ml/predict/top-performers", response_model=TopPerformersResponse)
    async def predict_top_performers(request: TopPerformersRequest):
        """Predict top K performers for next gameweek."""
        return await ml_integration.predict_top_performers(request)
    
    @app.post("/api/ml/predict/best-value")
    async def predict_best_value(request: BestValueRequest):
        """Predict best value players (points per million)."""
        return await ml_integration.predict_best_value(request)
    
    @app.get("/api/ml/status")
    async def ml_status():
        """Check ML predictor status."""
        return {
            "predictor_loaded": ml_integration.predictor_loaded,
            "model_type": ml_integration.predictor.model_type if ml_integration.predictor else None,
            "endpoints": [
                "/api/ml/predict/player",
                "/api/ml/predict/top-performers",
                "/api/ml/predict/best-value"
            ]
        }
    
    logger.info("✅ ML prediction routes registered")
