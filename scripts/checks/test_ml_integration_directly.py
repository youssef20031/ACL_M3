"""Test ML Integration directly without going through HTTP"""
import sys
import asyncio
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from ml.api_integration import MLAPIIntegration, TopPerformersRequest
from query_executor import QueryExecutor

async def test_predict_top_performers():
    """Test the predict_top_performers method directly"""
    
    print("Testing ML Integration directly...")
    print("=" * 80)
    
    # Setup
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    query_executor = QueryExecutor(conn)
    
    # Create ML integration
    ml_integration = MLAPIIntegration(conn, query_executor)
    
    # Load predictors (same as in api_main.py)
    model_paths = {
        "GK": "ml/models/xgboost_GK_v1.pkl",
        "DEF": "ml/models/xgboost_DEF_v1.pkl",
        "MID": "ml/models/xgboost_MID_v1.pkl",
        "FWD": "ml/models/xgboost_FWD_v1.pkl"
    }
    
    print("Loading predictors...")
    ml_integration.load_predictors_by_position(model_paths)
    
    if not ml_integration.predictor_loaded:
        print("❌ Failed to load predictors")
        return
    
    print("✅ Predictors loaded\n")
    
    # Create request
    request = TopPerformersRequest(position=None, top_k=5)
    
    print(f"Calling predict_top_performers(position={request.position}, top_k={request.top_k})...")
    
    try:
        response = await ml_integration.predict_top_performers(request)
        
        print(f"\n✅ SUCCESS! Got {len(response.predictions)} predictions\n")
        
        for i, pred in enumerate(response.predictions, 1):
            print(f"{i}. {pred.player_name}: {pred.predicted_points:.1f} pts")
            if pred.fixture_context:
                print(f"   Fixture: vs {pred.fixture_context['opponent_team']} ({pred.fixture_context['home_away']})")
            else:
                print(f"   Fixture: None")
                
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(test_predict_top_performers())
