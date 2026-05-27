"""
FPL FantasyTrivia - FastAPI Backend
Main API server replacing Streamlit
"""
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'          # skip TF/Keras import entirely
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import inspect
from contextlib import asynccontextmanager

# Import project modules
from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    HUGGINGFACE_API_TOKEN, DATA_PATH
)
from graph.connection import Neo4jConnection
from graph.queries import CypherQueries, QueryExecutor
from graph.data_loader import FPLDataLoader
from preprocessing.intent_classifier import IntentClassifier
from preprocessing.entity_extractor import EntityExtractor
from embeddings.embedding_manager import EmbeddingManager
from trivia.trivia_generator import TriviaGenerator
from llm.llm_manager import LLMManager, PromptBuilder
from llm.prompts import PromptTemplates

# Global state
app_state = {
    "neo4j_conn": None,
    "intent_classifier": None,
    "entity_extractor": None,
    "llm_manager": None,
    "embedding_manager": None,
    "embeddings_built": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    # Startup
    print("🚀 Initializing FPL FantasyTrivia API...")
    
    # Initialize LLM manager first (lightweight)
    if HUGGINGFACE_API_TOKEN:
        app_state["llm_manager"] = LLMManager(api_token=HUGGINGFACE_API_TOKEN)
    
    # Connect to Neo4j (fast)
    try:
        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if conn.test_connection():
            app_state["neo4j_conn"] = conn
            print("✅ Neo4j connected")
    except Exception as e:
        print(f"⚠️  Neo4j connection failed: {e}")
    
    print("✅ API Ready!")
    
    # Defer heavy ML model initialization to first use
    # This makes startup much faster
    
    yield
    
    # Shutdown
    if app_state["neo4j_conn"]:
        app_state["neo4j_conn"].close()
    print("👋 API Shutdown")


app = FastAPI(
    title="FPL FantasyTrivia API",
    description="Graph-RAG API for Fantasy Premier League Q&A and Trivia",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://aclm3-production.up.railway.app",
]

# Add Vercel domains if deployed
if os.getenv("VERCEL_URL"):
    allowed_origins.append(f"https://{os.getenv('VERCEL_URL')}")
    
# Allow all Vercel preview deployments
allowed_origins.append("https://*.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class ConnectionRequest(BaseModel):
    uri: str
    username: str
    password: str


class ConnectionResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict[str, int]] = None


class QueryRequest(BaseModel):
    question: str
    model: str = "qwen-2.5-coder"
    retrieval_method: str = "Hybrid"  # "Baseline", "Embeddings", "Hybrid"
    embedding_model: str = "minilm"  # "minilm" or "mpnet"


class QueryResponse(BaseModel):
    answer: str
    intent: str
    entities: Dict[str, Any]
    cypher_query: str
    kg_context: str
    embedding_context: Optional[str] = None
    embedding_used: bool = False
    results: List[Dict[str, Any]]
    graph_data: Optional[Dict[str, Any]] = None


class TriviaQuestion(BaseModel):
    question: str
    options: List[str]
    category: str
    difficulty: str
    question_id: str


class TriviaAnswerRequest(BaseModel):
    question_id: str
    answer: str


class TriviaAnswerResponse(BaseModel):
    correct: bool
    feedback: str
    correct_answer: Optional[str] = None


class EmbeddingBuildRequest(BaseModel):
    model: str = "minilm"


class EmbeddingBuildResponse(BaseModel):
    success: bool
    count: int
    message: str


class PlayerSearchRequest(BaseModel):
    query: str


class PlayerComparisonRequest(BaseModel):
    player1: str
    player2: str
    season: Optional[str] = None


class DataLoadRequest(BaseModel):
    clear_existing: bool = True


# ============================================================================
# Helper Functions
# ============================================================================

def get_intent_classifier():
    """Lazy load intent classifier."""
    if not app_state["intent_classifier"]:
        print("🔄 Loading intent classifier...")
        app_state["intent_classifier"] = IntentClassifier()
    return app_state["intent_classifier"]


def get_entity_extractor():
    """Lazy load entity extractor."""
    if not app_state["entity_extractor"]:
        print("🔄 Loading entity extractor...")
        app_state["entity_extractor"] = EntityExtractor()
        
        # Load known players if Neo4j is connected
        if app_state["neo4j_conn"]:
            try:
                query, _ = CypherQueries.get_all_player_names()
                results = app_state["neo4j_conn"].execute_query(query)
                if results:
                    players = {r['player_name'] for r in results}
                    app_state["entity_extractor"].set_known_players(players)
            except Exception as e:
                print(f"Warning: Failed to load player names: {e}")
    return app_state["entity_extractor"]


def get_neo4j_conn():
    """Dependency to get Neo4j connection."""
    if not app_state["neo4j_conn"]:
        raise HTTPException(status_code=503, detail="Neo4j not connected")
    return app_state["neo4j_conn"]


def build_graph_data(results: List[Dict], limit: int = 20) -> Dict[str, Any]:
    """Build graph visualization data from query results."""
    nodes = []
    edges = []
    node_ids = set()
    
    for i, result in enumerate(results[:limit]):
        # Player nodes
        if "player_name" in result:
            player = result["player_name"]
            if player not in node_ids:
                nodes.append({
                    "id": player,
                    "label": player,
                    "type": "player",
                    "data": {
                        "points": result.get("total_points", result.get("points", ""))
                    }
                })
                node_ids.add(player)
            
            # Position relationship
            if "position" in result:
                pos = result["position"]
                if pos not in node_ids:
                    nodes.append({"id": pos, "label": pos, "type": "position"})
                    node_ids.add(pos)
                edges.append({"from": player, "to": pos, "label": "PLAYS"})
            
            # Season relationship
            if "season" in result:
                season = result["season"]
                if season not in node_ids:
                    nodes.append({"id": season, "label": season, "type": "season"})
                    node_ids.add(season)
                edges.append({"from": player, "to": season, "label": "IN"})
        
        # Fixture nodes
        if "home_team" in result and "away_team" in result:
            home = result["home_team"]
            away = result["away_team"]
            
            if home not in node_ids:
                nodes.append({"id": home, "label": home, "type": "team"})
                node_ids.add(home)
            if away not in node_ids:
                nodes.append({"id": away, "label": away, "type": "team"})
                node_ids.add(away)
            
            fixture_id = f"fixture_{i}"
            nodes.append({
                "id": fixture_id,
                "label": f"{result.get('home_score', 0)}-{result.get('away_score', 0)}",
                "type": "fixture"
            })
            edges.append({"from": home, "to": fixture_id, "label": "HOME"})
            edges.append({"from": away, "to": fixture_id, "label": "AWAY"})
    
    return {"nodes": nodes, "edges": edges}


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint - lightweight for Railway."""
    return {
        "message": "FPL FantasyTrivia API",
        "status": "running"
    }


@app.get("/debug/env")
async def debug_env():
    """Temporary: show what Neo4j config Railway is using (no password)."""
    return {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USER": NEO4J_USER,
        "NEO4J_PASSWORD_LENGTH": len(NEO4J_PASSWORD),
        "NEO4J_PASSWORD_FIRST4": NEO4J_PASSWORD[:4] if NEO4J_PASSWORD else "EMPTY",
        "neo4j_connected": app_state["neo4j_conn"] is not None,
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    neo4j_status = "connected" if app_state["neo4j_conn"] else "disconnected"
    
    stats = None
    if app_state["neo4j_conn"]:
        try:
            stats = app_state["neo4j_conn"].get_database_stats()
        except:
            neo4j_status = "error"
    
    return {
        "status": "healthy",
        "neo4j": neo4j_status,
        "neo4j_stats": stats,
        "llm_available": app_state["llm_manager"] is not None,
        "embeddings_built": app_state["embeddings_built"],
        "embedding_count": len(app_state["embedding_manager"].player_embeddings) if app_state["embedding_manager"] else 0
    }


@app.post("/api/connection/connect", response_model=ConnectionResponse)
async def connect_neo4j(request: ConnectionRequest):
    """Connect to Neo4j database."""
    try:
        conn = Neo4jConnection(request.uri, request.username, request.password)
        if conn.test_connection():
            # Close old connection
            if app_state["neo4j_conn"]:
                app_state["neo4j_conn"].close()
            
            app_state["neo4j_conn"] = conn
            stats = conn.get_database_stats()
            
            # Load player names
            try:
                query, _ = CypherQueries.get_all_player_names()
                results = conn.execute_query(query)
                if results:
                    players = {r['player_name'] for r in results}
                    app_state["entity_extractor"].set_known_players(players)
            except:
                pass
            
            # Only return top-level int fields for Pydantic validation
            simple_stats = {
                "total_nodes": stats["total_nodes"],
                "total_relationships": stats["total_relationships"]
            }
            
            return ConnectionResponse(
                success=True,
                message="Connected successfully",
                stats=simple_stats
            )
        else:
            return ConnectionResponse(success=False, message="Connection test failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/connection/disconnect")
async def disconnect_neo4j():
    """Disconnect from Neo4j."""
    if app_state["neo4j_conn"]:
        app_state["neo4j_conn"].close()
        app_state["neo4j_conn"] = None
    return {"success": True, "message": "Disconnected"}


@app.post("/api/query", response_model=QueryResponse)
async def query_fpl(request: QueryRequest, conn=Depends(get_neo4j_conn)):
    """Process FPL query and return answer."""
    try:
        # Step 1: Intent Classification (lazy load)
        intent_classifier = get_intent_classifier()
        intent_result = intent_classifier.classify(request.question)
        
        # Step 2: Entity Extraction (lazy load)
        entity_extractor = get_entity_extractor()
        entities = entity_extractor.extract(request.question)
        
        # Step 3: Get query parameters
        params = entity_extractor.get_query_parameters(entities)
        
        # Step 4: Execute Cypher query
        query_executor = QueryExecutor(conn)
        cypher_context = ""
        executed_query = ""
        results = []
        
        query_method = intent_classifier.get_query_type_for_intent(intent_result.intent)
        
        if query_method:
            # Ensure required parameters
            if "limit" not in params:
                params["limit"] = 10
            if "limit_per_position" not in params:
                params["limit_per_position"] = 5
            
            # Special handling for different query types
            if query_method == "get_player_season_stats" and "gameweek" in params:
                query_method = "get_player_gameweek_performance"
                if "season" not in params:
                    params["season"] = "2022-23"
            elif query_method == "get_player_season_stats" and "season" not in params:
                query_method = "get_player_all_seasons_stats"
            
            if query_method == "get_most_transferred_players":
                if "season" not in params:
                    params["season"] = "2022-23"
                if "gameweek" not in params:
                    params["gameweek"] = 1
                if "out" in request.question.lower():
                    params["direction"] = "out"
                else:
                    params["direction"] = "in"
            
            if query_method == "get_most_selected_players":
                if "season" not in params:
                    params["season"] = "2022-23"
                if "gameweek" not in params:
                    params["gameweek"] = 1
            
            # Execute query
            method = getattr(CypherQueries, query_method)
            sig = inspect.signature(method)
            
            required_params = [
                p.name for p in sig.parameters.values() 
                if p.default == inspect.Parameter.empty
            ]
            missing_params = [p for p in required_params if p not in params]
            
            if missing_params:
                query, query_params = CypherQueries.get_top_players_all_positions(
                    season=params.get("season"),
                    gameweek=params.get("gameweek"),
                    limit_per_position=5
                )
            else:
                valid_params = {k: v for k, v in params.items() if k in sig.parameters}
                query, query_params = method(**valid_params)
            
            results = conn.execute_query(query, query_params)
            cypher_context = PromptBuilder.format_kg_context(results)
            executed_query = query
        
        if not results:
            query, query_params = CypherQueries.get_top_players_all_positions(
                season=params.get("season"),
                gameweek=params.get("gameweek"),
                limit_per_position=5
            )
            results = conn.execute_query(query, query_params)
            cypher_context = PromptBuilder.format_kg_context(results)
            executed_query = query
        
        # Step 5: Embedding search
        embedding_context = ""
        embedding_used = False
        
        if request.retrieval_method in ["Embeddings", "Hybrid"]:
            if not app_state["embedding_manager"]:
                app_state["embedding_manager"] = EmbeddingManager(model_key=request.embedding_model)
            elif app_state["embedding_manager"].model_key != request.embedding_model:
                app_state["embedding_manager"].switch_model(request.embedding_model)
            
            if app_state["embeddings_built"]:
                try:
                    key_player = None
                    if entities.players:
                        key_player = entities.players[0]
                    elif results and 'player_name' in results[0]:
                        key_player = results[0]['player_name']
                    
                    if key_player:
                        season = entities.seasons[0] if entities.seasons else "2022-23"
                        similar_players = app_state["embedding_manager"].find_similar_to_player(
                            key_player, season=season, top_k=10, exclude_self=False
                        )
                    else:
                        similar_players = app_state["embedding_manager"].find_similar_players(
                            request.question, top_k=5
                        )
                    embedding_context = PromptBuilder.format_embedding_context(similar_players)
                    embedding_used = True
                except Exception as e:
                    print(f"Embedding search failed: {e}")
        
        # Step 6: Generate LLM response
        answer = ""
        if app_state["llm_manager"] and app_state["llm_manager"].client:
            data_scope = f"the {params['season']} season" if "season" in params else "all seasons"
            full_prompt = PromptTemplates.qa_template(
                question=request.question,
                kg_context=cypher_context,
                embedding_context=embedding_context if embedding_context else None,
                data_scope=data_scope
            )
            
            response = app_state["llm_manager"].generate(
                full_prompt,
                model_key=request.model,
                timeout=60
            )
            
            if response.success:
                answer = response.text
            else:
                answer = f"Error generating response: {response.error}"
        else:
            answer = f"**Knowledge Graph data:**\n\n{cypher_context}"
        
        # Build graph visualization data
        graph_data = build_graph_data(results)
        
        return QueryResponse(
            answer=answer,
            intent=intent_result.intent.value,
            entities=entities.to_dict(),
            cypher_query=executed_query,
            kg_context=cypher_context,
            embedding_context=embedding_context,
            embedding_used=embedding_used,
            results=results[:50],  # Limit results sent to frontend
            graph_data=graph_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/embeddings/build", response_model=EmbeddingBuildResponse)
async def build_embeddings(request: EmbeddingBuildRequest, conn=Depends(get_neo4j_conn)):
    """Build player embeddings from Neo4j data."""
    try:
        # Initialize embedding manager
        if not app_state["embedding_manager"]:
            app_state["embedding_manager"] = EmbeddingManager(model_key=request.model)
        elif app_state["embedding_manager"].model_key != request.model:
            app_state["embedding_manager"].switch_model(request.model)
        
        # Get player data
        query, query_params = CypherQueries.get_player_embeddings_data()
        results = conn.execute_query(query, query_params)
        
        if not results:
            return EmbeddingBuildResponse(
                success=False,
                count=0,
                message="No player data found. Load FPL data first."
            )
        
        # Build embeddings
        app_state["embedding_manager"].build_player_embeddings(results)
        app_state["embeddings_built"] = True
        count = len(app_state["embedding_manager"].player_embeddings)
        
        return EmbeddingBuildResponse(
            success=True,
            count=count,
            message=f"Successfully built {count} embeddings"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trivia/new", response_model=TriviaQuestion)
async def get_trivia_question(conn=Depends(get_neo4j_conn)):
    """Generate a new trivia question."""
    try:
        trivia_gen = TriviaGenerator(conn)
        question = trivia_gen.generate_question()
        
        if not question:
            raise HTTPException(status_code=500, detail="Failed to generate question")
        
        return TriviaQuestion(
            question=question.question,
            options=question.options,
            category=question.category.value,
            difficulty=question.difficulty.value,
            question_id=question.question_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trivia/answer", response_model=TriviaAnswerResponse)
async def check_trivia_answer(request: TriviaAnswerRequest, conn=Depends(get_neo4j_conn)):
    """Check trivia answer (simplified - store questions in session/cache in production)."""
    try:
        # Note: In production, you'd store the question in Redis/cache with question_id
        # For now, just return a generic response
        return TriviaAnswerResponse(
            correct=False,
            feedback="Answer checking requires session management. Implement with Redis/cache.",
            correct_answer=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/players/search")
async def search_players(request: PlayerSearchRequest, conn=Depends(get_neo4j_conn)):
    """Search for players by name, with accent-insensitive matching."""
    import unicodedata

    def normalize(text: str) -> str:
        """Strip accents: é -> e, ü -> u, etc."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    try:
        raw_query = request.query
        normalized_query = normalize(raw_query)

        # First try the standard query (exact accent match)
        query, params = CypherQueries.search_players_by_name(raw_query)
        results = conn.execute_query(query, params)

        # If no results, fall back to accent-stripped search across all players
        if not results:
            all_query = "MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position) RETURN p.name AS player_name, pos.code AS position"
            all_players = conn.execute_query(all_query, {})
            results = [
                r for r in all_players
                if normalized_query in normalize(r.get("player_name", ""))
            ][:10]

        return {"players": results[:10]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/players/compare")
async def compare_players(request: PlayerComparisonRequest, conn=Depends(get_neo4j_conn)):
    """Compare two players, resolving accented names automatically."""
    import unicodedata

    def normalize(text: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    def resolve_name(raw: str, conn) -> str:
        """Return the exact DB name that matches the input after accent stripping."""
        norm = normalize(raw)
        # Try exact match first
        exact_query = "MATCH (p:Player) WHERE toLower(p.name) = toLower($name) RETURN p.name AS player_name LIMIT 1"
        result = conn.execute_query(exact_query, {"name": raw})
        if result:
            return result[0]["player_name"]
        # Fall back to accent-stripped scan
        all_query = "MATCH (p:Player) RETURN p.name AS player_name"
        all_players = conn.execute_query(all_query, {})
        for r in all_players:
            if normalize(r["player_name"]) == norm:
                return r["player_name"]
        return raw  # return original if nothing found

    try:
        p1 = resolve_name(request.player1, conn)
        p2 = resolve_name(request.player2, conn)
        query, params = CypherQueries.compare_players(p1, p2, request.season)
        results = conn.execute_query(query, params)
        return {"comparison": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/load")
async def load_fpl_data(request: DataLoadRequest, conn=Depends(get_neo4j_conn)):
    """Load FPL data into Neo4j."""
    try:
        loader = FPLDataLoader(conn)
        stats = loader.load_all(DATA_PATH, clear_existing=request.clear_existing)
        
        # Reload player names for entity extraction (if already loaded)
        if app_state["entity_extractor"]:
            query, _ = CypherQueries.get_all_player_names()
            results = conn.execute_query(query)
            if results:
                players = {r['player_name'] for r in results}
                app_state["entity_extractor"].set_known_players(players)
        
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/database")
async def get_database_stats(conn=Depends(get_neo4j_conn)):
    """Get Neo4j database statistics."""
    try:
        stats = conn.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket for Real-time Updates (Optional)
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time query streaming."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process query and stream response
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)