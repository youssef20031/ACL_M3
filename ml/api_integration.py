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
    player_data: Dict[str, Any]


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
        
        Args:
            player_name: Player name
            season: Optional season filter
            
        Returns:
            Player data dictionary with recent stats
        """
        # Query for recent player stats (last 4 games for form)
        query = """
        MATCH (p:Player {name: $player_name})-[:PLAYED_IN]->(perf:Performance)
        MATCH (perf)-[:IN_FIXTURE]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
        WHERE CASE WHEN $season IS NOT NULL THEN s.id = $season ELSE true END
        WITH p, perf, f, gw, s
        ORDER BY f.kickoff_time DESC
        LIMIT 4
        RETURN 
            p.name as name,
            AVG(perf.minutes) as minutes,
            AVG(perf.goals_scored) as goals_scored,
            AVG(perf.assists) as assists,
            AVG(perf.total_points) as form,
            AVG(perf.bps) as bps,
            AVG(perf.ict_index) as ict_index,
            AVG(perf.influence) as influence,
            AVG(perf.creativity) as creativity,
            AVG(perf.threat) as threat,
            SUM(perf.clean_sheets) as clean_sheets,
            SUM(perf.bonus) as bonus,
            SUM(perf.goals_conceded) as goals_conceded,
            SUM(perf.saves) as saves,
            SUM(perf.yellow_cards) as yellow_cards,
            SUM(perf.red_cards) as red_cards,
            AVG(perf.value) as value,
            p.position as position
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
                player_data = request.player_data
            
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
            # Query for all players with recent stats
            position_filter = f"WHERE p.position = '{request.position}'" if request.position else ""
            season_filter = f"AND s.id = '{request.season}'" if request.season else ""
            
            query = f"""
            MATCH (p:Player)-[:PLAYED_IN]->(perf:Performance)
            MATCH (perf)-[:IN_FIXTURE]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            {position_filter}
            WHERE perf.minutes > 0 {season_filter}
            WITH p, perf, f
            ORDER BY f.kickoff_time DESC
            WITH p, COLLECT(perf)[..4] as recent_perfs
            WHERE SIZE(recent_perfs) >= 2
            RETURN 
                p.name as name,
                p.position as position,
                AVG([perf IN recent_perfs | perf.minutes]) as minutes,
                AVG([perf IN recent_perfs | perf.goals_scored]) as goals_scored,
                AVG([perf IN recent_perfs | perf.assists]) as assists,
                AVG([perf IN recent_perfs | perf.total_points]) as form,
                AVG([perf IN recent_perfs | perf.bps]) as bps,
                AVG([perf IN recent_perfs | perf.ict_index]) as ict_index,
                AVG([perf IN recent_perfs | perf.influence]) as influence,
                AVG([perf IN recent_perfs | perf.creativity]) as creativity,
                AVG([perf IN recent_perfs | perf.threat]) as threat,
                SUM([perf IN recent_perfs | perf.clean_sheets]) as clean_sheets,
                SUM([perf IN recent_perfs | perf.bonus]) as bonus,
                AVG([perf IN recent_perfs | perf.value]) as value
            LIMIT 200
            """
            
            players_data = self.neo4j_conn.execute_query(query)
            
            if not players_data:
                raise HTTPException(
                    status_code=404,
                    detail="No player data found for prediction"
                )
            
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
            # Query for players within price range
            price_filter = f"AND AVG([perf IN recent_perfs | perf.value]) <= {request.max_price * 10}" if request.max_price else ""
            position_filter = f"WHERE p.position = '{request.position}'" if request.position else ""
            
            query = f"""
            MATCH (p:Player)-[:PLAYED_IN]->(perf:Performance)
            MATCH (perf)-[:IN_FIXTURE]->(f:Fixture)
            {position_filter}
            WHERE perf.minutes > 0
            WITH p, perf, f
            ORDER BY f.kickoff_time DESC
            WITH p, COLLECT(perf)[..4] as recent_perfs
            WHERE SIZE(recent_perfs) >= 2
            RETURN 
                p.name as name,
                p.position as position,
                AVG([perf IN recent_perfs | perf.minutes]) as minutes,
                AVG([perf IN recent_perfs | perf.goals_scored]) as goals_scored,
                AVG([perf IN recent_perfs | perf.assists]) as assists,
                AVG([perf IN recent_perfs | perf.total_points]) as form,
                AVG([perf IN recent_perfs | perf.bps]) as bps,
                AVG([perf IN recent_perfs | perf.ict_index]) as ict_index,
                AVG([perf IN recent_perfs | perf.value]) as value
            {price_filter}
            LIMIT 200
            """
            
            players_data = self.neo4j_conn.execute_query(query)
            
            if not players_data:
                return []
            
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
