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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import asdict
from typing import List, Optional, Dict, Any
import json
import inspect
import time
import hashlib
import re
import unicodedata
import logging
import io
from threading import Lock, Thread
from contextlib import asynccontextmanager
from urllib.parse import quote_plus

# Import project modules
from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    HUGGINGFACE_API_TOKEN, DATA_PATH,
    GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_CUSTOM_SEARCH_CX
)
from graph.connection import Neo4jConnection
from typing import Optional


from graph.queries import CypherQueries, QueryExecutor
from graph.data_loader import FPLDataLoader
from preprocessing.intent_classifier import IntentClassifier, Intent
from preprocessing.entity_extractor import EntityExtractor
from embeddings.embedding_manager import EmbeddingManager
try:
    import redis  # type: ignore
except Exception:
    redis = None

from trivia.trivia_generator import TriviaGenerator, TriviaQuestion as GeneratedTriviaQuestion, TriviaCategory, Difficulty
from llm.llm_manager import LLMManager, PromptBuilder
from llm.prompts import PromptTemplates
from ml.api_integration import MLAPIIntegration, register_ml_routes

import requests

logger = logging.getLogger(__name__)

# Global state
app_state = {
    "neo4j_conn": None,
    "intent_classifier": None,
    "entity_extractor": None,
    "llm_manager": None,
    "embedding_manager": None,
    "embeddings_built": False,
    "embedding_building": False,
    "embedding_build_error": None,
    "trivia_cache": None,
    "player_search_cache": None,
    "ml_integration": None,
}

embedding_build_lock = Lock()


def get_wikipedia_player_image_url(player_name: str) -> Optional[str]:
    """Fallback to the Wikimedia thumbnail for the player's Wikipedia page."""
    normalized_name = " ".join(player_name.split()).strip().lower()
    if not normalized_name:
        return None

    if normalized_name in PLAYER_AVATAR_CACHE:
        return PLAYER_AVATAR_CACHE[normalized_name]

    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "pithumbsize": 250,
                "redirects": 1,
                "titles": player_name,
            },
            headers={"User-Agent": "FPL FantasyTrivia/1.0 (local testing)"},
            timeout=15,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumbnail = page.get("thumbnail", {})
            image_url = thumbnail.get("source")
            if image_url:
                PLAYER_AVATAR_CACHE[normalized_name] = image_url
                return image_url
    except Exception as exc:
        logger.warning("Wikipedia avatar lookup failed for %s: %s", player_name, exc)

    return None


def build_image_search_variants(query: str) -> List[str]:
    """Return a small set of progressively simpler search strings for image lookup."""
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return []

    variants: List[str] = []
    seen = set()

    def add_variant(value: str) -> None:
        value = " ".join(value.split()).strip()
        if value and value not in seen:
            seen.add(value)
            variants.append(value)

    add_variant(normalized)

    accent_free = unicodedata.normalize("NFKD", normalized)
    accent_free = "".join(char for char in accent_free if not unicodedata.combining(char))
    add_variant(accent_free)

    parts = normalized.split()
    if len(parts) > 2:
        add_variant(" ".join(parts[:2]))
        add_variant(" ".join(accent_free.split()[:2]))
    if len(parts) > 1:
        add_variant(parts[-1])
        add_variant(accent_free.split()[-1])
    if len(parts) > 3:
        add_variant(" ".join(parts[-2:]))
        add_variant(" ".join(accent_free.split()[-2:]))

    return variants


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8/Latin-1 mojibake such as NÃºÃ±ez -> Núñez."""
    if not text:
        return text
    if "Ã" not in text and "Â" not in text:
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text
PLAYER_AVATAR_CACHE: Dict[str, Optional[str]] = {}
FPL_PLAYER_PHOTO_CACHE: Dict[str, Optional[str]] = {}
FPL_PLAYER_PHOTO_CACHE_READY = False


class TriviaQuestionCache:
    """Cache generated trivia questions for answer validation."""

    def __init__(self, ttl_seconds: int = 1800, repeat_window_seconds: int = 1800):
        self.ttl_seconds = ttl_seconds
        self.repeat_window_seconds = repeat_window_seconds
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._recent_questions: Dict[str, float] = {}
        self._redis = None

        redis_url = os.getenv("REDIS_URL")
        if redis_url and redis is not None:
            try:
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                print("✅ Trivia cache using Redis")
            except Exception as exc:
                print(f"⚠️ Redis unavailable for trivia cache, falling back to memory: {exc}")
                self._redis = None

    def _serialize(self, question: GeneratedTriviaQuestion) -> Dict[str, Any]:
        payload = asdict(question)
        payload["category"] = question.category.value
        payload["difficulty"] = question.difficulty.value
        return payload

    def _deserialize(self, payload: Dict[str, Any]) -> Optional[GeneratedTriviaQuestion]:
        try:
            return GeneratedTriviaQuestion(
                question_id=payload["question_id"],
                question=payload["question"],
                correct_answer=payload["correct_answer"],
                options=list(payload.get("options", [])),
                category=TriviaCategory(payload["category"]),
                difficulty=Difficulty(payload["difficulty"]),
                explanation=payload["explanation"],
                source_query=payload["source_query"],
                metadata=payload.get("metadata", {}),
            )
        except Exception:
            return None

    def store(self, question: GeneratedTriviaQuestion) -> None:
        payload = self._serialize(question)
        if self._redis is not None:
            self._redis.setex(question.question_id, self.ttl_seconds, json.dumps(payload))
            return

        self._memory[question.question_id] = {
            "expires_at": time.time() + self.ttl_seconds,
            "payload": payload,
        }

    def get(self, question_id: str) -> Optional[GeneratedTriviaQuestion]:
        if self._redis is not None:
            raw = self._redis.get(question_id)
            if not raw:
                return None
            try:
                return self._deserialize(json.loads(raw))
            except Exception:
                return None

        record = self._memory.get(question_id)
        if not record:
            return None
        if record["expires_at"] < time.time():
            self._memory.pop(question_id, None)
            return None
        return self._deserialize(record["payload"])

    def _question_key(self, question_text: str) -> str:
        normalized = " ".join(question_text.lower().split())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return f"trivia:recent:{digest}"

    def is_recent(self, question_text: str) -> bool:
        key = self._question_key(question_text)

        if self._redis is not None:
            try:
                return bool(self._redis.exists(key))
            except Exception:
                return False

        now = time.time()
        expired = [k for k, exp in self._recent_questions.items() if exp < now]
        for k in expired:
            self._recent_questions.pop(k, None)
        return key in self._recent_questions

    def mark_recent(self, question_text: str) -> None:
        key = self._question_key(question_text)

        if self._redis is not None:
            try:
                self._redis.setex(key, self.repeat_window_seconds, "1")
                return
            except Exception:
                pass

        self._recent_questions[key] = time.time() + self.repeat_window_seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    # Startup
    print("🚀 Initializing FPL FantasyTrivia API...")
    
    # Initialize LLM manager first (lightweight)
    if HUGGINGFACE_API_TOKEN:
        app_state["llm_manager"] = LLMManager(api_token=HUGGINGFACE_API_TOKEN)

    app_state["trivia_cache"] = TriviaQuestionCache()
    
    # Connect to Neo4j (fast)
    try:
        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if conn.test_connection():
            app_state["neo4j_conn"] = conn
            print("✅ Neo4j connected")
            try:
                get_player_search_cache(conn)
                print("✅ Player search cache warmed")
            except Exception as cache_error:
                print(f"⚠️ Player search cache warm-up failed: {cache_error}")
    except Exception as e:
        print(f"⚠️  Neo4j connection failed: {e}")
    
    # Try to load prebuilt embeddings on startup (Railway optimization)
    print("🔮 Checking for prebuilt embeddings...")
    try:
        # Try MPNet first, then MiniLM
        for model_key in ["mpnet", "minilm"]:
            prebuilt_path = f"embeddings/prebuilt/{model_key}_embeddings.pkl"
            if os.path.exists(prebuilt_path):
                print(f"   Found {model_key} embeddings, loading...")
                # Load embeddings WITHOUT initializing the sentence-transformer model
                # The model is only needed for search queries, not for loading prebuilt vectors
                manager = EmbeddingManager.__new__(EmbeddingManager)
                manager.model_key = model_key
                manager.model_info = EmbeddingManager.MODELS[model_key]
                manager.model = None  # Defer model loading until first search
                manager.player_embeddings = {}
                manager.player_metadata = {}
                if manager.load_embeddings(prebuilt_path):
                    app_state["embedding_manager"] = manager
                    app_state["embeddings_built"] = True
                    print(f"✅ Loaded {len(manager.player_embeddings)} prebuilt {model_key} embeddings (model deferred)")
                    break
        else:
            print("   ℹ️  No prebuilt embeddings found (will build on request)")
    except Exception as emb_error:
        print(f"⚠️  Failed to load prebuilt embeddings: {emb_error}")

    # Initialize ML prediction integration
    print("🤖 Initializing ML prediction engine...")
    try:
        ml_integration = MLAPIIntegration(
            neo4j_conn=app_state["neo4j_conn"],
            query_executor=None  # Not needed; queries run directly via neo4j_conn
        )
        # Try XGBoost models first (best performance), fall back to linear regression
        model_loaded = False
        for model_path in [
            "ml/models/linear_regression_v1.pkl",
        ]:
            if os.path.exists(model_path):
                ml_integration.load_predictor(model_path)
                model_loaded = True
                print(f"✅ ML model loaded: {model_path}")
                break
        if not model_loaded:
            print("⚠️  No trained ML model found — prediction endpoints will be unavailable")
        app_state["ml_integration"] = ml_integration
    except Exception as ml_error:
        print(f"⚠️  ML integration failed to initialize: {ml_error}")

    print("✅ API Ready!")

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
    "http://localhost:3001",
    "https://aclm3-production.up.railway.app",
    "https://acl-m3.vercel.app",
]

# Add Vercel domains if deployed
if os.getenv("VERCEL_URL"):
    allowed_origins.append(f"https://{os.getenv('VERCEL_URL')}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ============================================================================
# ML Prediction Routes (registered at module load; integration resolved lazily)
# ============================================================================

class _LazyMLIntegration:
    """Thin proxy that forwards calls to the MLAPIIntegration stored in app_state.
    This lets us register routes at import time while the real instance is created
    during the lifespan startup."""

    @property
    def _real(self):
        return app_state.get("ml_integration")

    @property
    def predictor_loaded(self):
        real = self._real
        return real.predictor_loaded if real else False

    @property
    def predictor(self):
        real = self._real
        return real.predictor if real else None

    async def predict_player_next_gameweek(self, request):
        if not self._real:
            raise HTTPException(status_code=503, detail="ML integration not initialized")
        return await self._real.predict_player_next_gameweek(request)

    async def predict_top_performers(self, request):
        if not self._real:
            raise HTTPException(status_code=503, detail="ML integration not initialized")
        return await self._real.predict_top_performers(request)

    async def predict_best_value(self, request):
        if not self._real:
            raise HTTPException(status_code=503, detail="ML integration not initialized")
        return await self._real.predict_best_value(request)


_lazy_ml = _LazyMLIntegration()
register_ml_routes(app, _lazy_ml)


# ============================================================================
# Pydantic Models
# ============================================================================

class ConnectionRequest(BaseModel):
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class ConnectionResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict[str, int]] = None


class QueryRequest(BaseModel):
    question: str
    model: str = "qwen-2.5-coder"
    retrieval_method: str = "Hybrid"  # "Baseline", "Embeddings", "Hybrid"
    embedding_model: str = "minilm"  # "minilm" or "mpnet"
    is_first_message: bool = False
    chat_history: List[Dict[str, str]] = []  # List of {"role": "user"/"assistant", "content": "..."}


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


class ImageSearchResponse(BaseModel):
    query: str
    image_url: Optional[str] = None
    source: Optional[str] = None


def looks_like_small_talk(question: str) -> bool:
    """Return True when the message is a normal chat response rather than an FPL query."""
    text = question.strip().lower()
    if not text:
        return True

    small_talk_patterns = [
        r"^(hi|hello|hey|yo|thanks|thank you|thx|nice|cool|wow|great|awesome|amazing)([!.?\s].*)?$",
        r"^(that's|that is)\s+(great|awesome|amazing|cool|nice)([!.?\s].*)?$",
        r"^(ok|okay|sure|got it|understood|perfect)([!.?\s].*)?$",
        r"^(lol|lmao|haha|haha+)([!.?\s].*)?$",
    ]

    return any(re.match(pattern, text) for pattern in small_talk_patterns)


def build_proxy_image_url(source_url: str) -> str:
    return f"/api/images/proxy?url={quote_plus(source_url)}"


def _looks_like_generic_wikimedia_image(url: str) -> bool:
    """Return True for known generic/placeholder Wikimedia images we should ignore."""
    if not url:
        return False
    lowered = url.lower()
    generic_indicators = [
        'image_created_with_a_mobile_phone',
        'no_image_available',
        'no_photo',
        'default',
        'placeholder',
    ]
    return any(ind in lowered for ind in generic_indicators)


def normalize_text(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def _tokenize_normalized_text(text: str) -> List[str]:
    return [token for token in normalize_text(text).split() if token]


def _tokens_appear_in_order(query_tokens: List[str], candidate_tokens: List[str]) -> bool:
    if not query_tokens:
        return False

    candidate_index = 0
    for query_token in query_tokens:
        found = False
        while candidate_index < len(candidate_tokens):
            candidate_token = candidate_tokens[candidate_index]
            candidate_index += 1
            if candidate_token == query_token or candidate_token.startswith(query_token) or query_token in candidate_token:
                found = True
                break
        if not found:
            return False
    return True


def score_player_search_match(query: str, candidate_name: str) -> Optional[tuple]:
    normalized_query = normalize_text(query)
    normalized_candidate = normalize_text(candidate_name)

    if not normalized_query or not normalized_candidate:
        return None

    if normalized_query == normalized_candidate:
        return (4, 1, 1, 1, 0, 0)

    query_tokens = _tokenize_normalized_text(normalized_query)
    candidate_tokens = _tokenize_normalized_text(normalized_candidate)
    if not query_tokens or not candidate_tokens:
        return None

    matched_tokens = 0
    for query_token in query_tokens:
        if any(candidate_token == query_token or candidate_token.startswith(query_token) or query_token in candidate_token for candidate_token in candidate_tokens):
            matched_tokens += 1

    if matched_tokens == 0:
        return None

    return (
        matched_tokens,
        1 if _tokens_appear_in_order(query_tokens, candidate_tokens) else 0,
        1 if normalized_candidate.startswith(normalized_query) else 0,
        1 if normalized_query in normalized_candidate else 0,
        -len(candidate_tokens),
        -len(normalized_candidate),
    )


def rank_player_search_results(rows: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    ranked_rows = [
        (score_player_search_match(query, str(row.get("player_name", ""))), row)
        for row in rows
    ]
    ranked_rows = [item for item in ranked_rows if item[0] is not None]
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in ranked_rows[:limit]]


def get_player_search_cache(conn) -> List[Dict[str, Any]]:
    """Load and cache searchable player suggestions in memory."""
    if app_state["player_search_cache"] is not None:
        return app_state["player_search_cache"]

    query = """
    MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
    RETURN p.name AS player_name, pos.code AS position, p.element_id AS element_id
    ORDER BY p.name
    """
    results = conn.execute_query(query, {})
    app_state["player_search_cache"] = results
    return results


def get_player_avatar_url(player_name: str, team_name: Optional[str] = None) -> Optional[str]:
    """Return the best available avatar URL for a player."""
    player_name = repair_mojibake(player_name)
    normalized_name = " ".join(player_name.split()).strip().lower()
    if not normalized_name:
        return None

    if normalized_name in PLAYER_AVATAR_CACHE:
        cached_url = PLAYER_AVATAR_CACHE[normalized_name]
        return build_proxy_image_url(cached_url) if cached_url else None

    # Try to find team name if not provided
    if not team_name and app_state["neo4j_conn"]:
        try:
            team_query = "MATCH (p:Player {name: $name})-[:PLAYS_FOR]->(t:Team) RETURN t.name AS team_name LIMIT 1"
            team_res = app_state["neo4j_conn"].execute_query(team_query, {"name": player_name})
            if team_res:
                team_name = team_res[0]["team_name"]
        except Exception:
            pass

    image_url = None
    search_queries: List[str] = []
    if team_name:
        search_queries.extend(build_image_search_variants(f"{player_name} {team_name}"))
        search_queries.extend(build_image_search_variants(f"{player_name} {team_name} official picture"))
    search_queries.extend(build_image_search_variants(player_name))
    search_queries.extend(build_image_search_variants(f"{player_name} official picture"))

    deduped_queries: List[str] = []
    seen_queries = set()
    for search_query in search_queries:
        if search_query not in seen_queries:
            seen_queries.add(search_query)
            deduped_queries.append(search_query)

    # First prefer official FPL player photos when available (more reliable).
    # If the FPL photo URL is blocked (returns non-200), ignore it and fall back.
    fpl_photo = get_fpl_player_photo_url(player_name)
    if fpl_photo:
        try:
            # quick HEAD check to avoid choosing inaccessible URLs
            head_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "image/*,*/*;q=0.8", "Referer": "https://fantasy.premierleague.com/"}
            head_resp = requests.head(fpl_photo, headers=head_headers, timeout=6, allow_redirects=True)
            if head_resp.status_code == 200:
                image_url = fpl_photo
            else:
                logger.debug("FPL photo unreachable (%s): %s", head_resp.status_code, fpl_photo)
        except Exception as exc:
            logger.debug("FPL photo HEAD check failed for %s: %s", fpl_photo, exc)

    if not image_url and GOOGLE_CUSTOM_SEARCH_API_KEY and GOOGLE_CUSTOM_SEARCH_CX:
        for search_query in deduped_queries:
            try:
                response = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": GOOGLE_CUSTOM_SEARCH_API_KEY,
                        "cx": GOOGLE_CUSTOM_SEARCH_CX,
                        "q": search_query,
                        "searchType": "image",
                        "num": 1,
                        "safe": "active",
                    },
                    timeout=10,
                )
                response.raise_for_status()
                items = response.json().get("items", [])
                if items:
                    image_info = items[0].get("image", {})
                    image_url = image_info.get("thumbnailLink") or items[0].get("link")
                    if image_url:
                        # ignore clearly generic Wikimedia images
                        if _looks_like_generic_wikimedia_image(image_url):
                            image_url = None
                            continue
                        break
            except Exception as exc:
                logger.warning("Google avatar lookup failed for %s with query %s: %s", player_name, search_query, exc)

    if not image_url:
        for search_query in deduped_queries:
            image_url = get_wikipedia_player_image_url(search_query)
            if image_url:
                # discard generic Wikimedia placeholders
                if _looks_like_generic_wikimedia_image(image_url):
                    image_url = None
                    continue
                break

    PLAYER_AVATAR_CACHE[normalized_name] = image_url
    return build_proxy_image_url(image_url) if image_url else None


def get_fpl_player_photo_url(player_name: str) -> Optional[str]:
    """Fallback to the official FPL player photo when no Google image is available."""
    global FPL_PLAYER_PHOTO_CACHE_READY

    normalized_name = " ".join(player_name.split()).strip().lower()
    if not normalized_name:
        return None

    if normalized_name in FPL_PLAYER_PHOTO_CACHE:
        return FPL_PLAYER_PHOTO_CACHE[normalized_name]

    if not FPL_PLAYER_PHOTO_CACHE_READY:
        try:
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15)
            response.raise_for_status()
            for element in response.json().get("elements", []):
                photo = element.get("photo")
                if not photo:
                    continue

                photo_url = f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{photo}"
                aliases = {
                    f"{element.get('first_name', '')} {element.get('second_name', '')}".strip(),
                    f"{element.get('second_name', '')} {element.get('first_name', '')}".strip(),
                    element.get("web_name", ""),
                }
                for alias in aliases:
                    alias_key = " ".join(alias.split()).strip().lower()
                    if alias_key:
                        FPL_PLAYER_PHOTO_CACHE.setdefault(alias_key, photo_url)
            FPL_PLAYER_PHOTO_CACHE_READY = True
        except Exception as exc:
            logger.warning("FPL photo cache could not be loaded: %s", exc)
            FPL_PLAYER_PHOTO_CACHE_READY = True

    return FPL_PLAYER_PHOTO_CACHE.get(normalized_name)


def enrich_rows_with_avatars(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach avatar URLs to player rows when possible."""
    enriched_rows: List[Dict[str, Any]] = []
    for row in rows:
        enriched_row = dict(row)
        player_name = enriched_row.get("player_name")
        team_name = enriched_row.get("team_name")
        if player_name and not enriched_row.get("avatar"):
            enriched_row["avatar"] = get_player_avatar_url(str(player_name), team_name=team_name)
        enriched_rows.append(enriched_row)
    return enriched_rows


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
    started: bool = False
    building: bool = False
    model: Optional[str] = None


class PlayerSearchRequest(BaseModel):
    query: str
    limit: int = 20
    include_avatars: bool = True


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


def run_embedding_build(model_key: str, conn: Neo4jConnection) -> None:
    """Build embeddings asynchronously or load prebuilt ones."""
    try:
        logger.info("Starting background embedding build for model %s", model_key)
        app_state["embedding_build_error"] = None

        if not app_state["embedding_manager"]:
            logger.info("Initializing embedding manager...")
            app_state["embedding_manager"] = EmbeddingManager(model_key=model_key)
            
            # Try to load prebuilt embeddings first (for Railway)
            prebuilt_path = f"embeddings/prebuilt/{model_key}_embeddings.pkl"
            if app_state["embedding_manager"].load_embeddings(prebuilt_path):
                logger.info("Successfully loaded prebuilt embeddings from %s", prebuilt_path)
                app_state["embeddings_built"] = True
                app_state["embedding_building"] = False
                return
            else:
                logger.info("No prebuilt embeddings found, building from scratch...")
        elif app_state["embedding_manager"].model_key != model_key:
            logger.info("Switching embedding model...")
            app_state["embedding_manager"].switch_model(model_key)
            
            # Try to load prebuilt for new model
            prebuilt_path = f"embeddings/prebuilt/{model_key}_embeddings.pkl"
            if app_state["embedding_manager"].load_embeddings(prebuilt_path):
                logger.info("Successfully loaded prebuilt embeddings for %s", model_key)
                app_state["embeddings_built"] = True
                app_state["embedding_building"] = False
                return

        logger.info("Fetching player data from Neo4j...")
        query, query_params = CypherQueries.get_player_embeddings_data()
        results = conn.execute_query(query, query_params)

        if not results:
            app_state["embedding_build_error"] = "No player data found. Load FPL data first."
            app_state["embeddings_built"] = False
            logger.error("No player data found for embedding build")
            return

        logger.info(f"Building embeddings for {len(results)} players...")
        app_state["embedding_manager"].build_player_embeddings(results, batch_size=16)
        app_state["embeddings_built"] = True
        logger.info(
            "✅ Embedding build finished for model %s with %d vectors",
            model_key,
            len(app_state["embedding_manager"].player_embeddings),
        )
    except MemoryError as exc:
        app_state["embeddings_built"] = False
        app_state["embedding_build_error"] = "Out of memory. Try restarting the server or upgrading Railway plan."
        logger.exception("Embedding build failed due to memory error for model %s", model_key)
    except Exception as exc:
        app_state["embeddings_built"] = False
        app_state["embedding_build_error"] = str(exc)
        logger.exception("Embedding build failed for model %s: %s", model_key, str(exc))
    finally:
        app_state["embedding_building"] = False
        logger.info("Embedding build process completed (building flag reset)")


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


def add_tie_note_if_needed(
    intent: Intent,
    query_method: str,
    params: Dict[str, Any],
    results: List[Dict[str, Any]],
    context: str,
) -> str:
    """Annotate ranked results when the top value is tied so the LLM can mention all co-leaders."""
    if not results:
        return context

    is_goal_leader_query = intent == Intent.TOP_SCORERS or (
        query_method == "get_top_points_by_position" and params.get("sort_by") == "goals"
    )

    if not is_goal_leader_query:
        return context

    metric_key = next((candidate for candidate in ("total_goals", "goals", "answer") if candidate in results[0]), None)
    if not metric_key:
        return context

    top_value = results[0].get(metric_key)
    if top_value is None:
        return context

    name_key = "player_name" if "player_name" in results[0] else "answer"
    tied_names = []
    for row in results:
        if row.get(metric_key) == top_value and row.get(name_key):
            tied_names.append(str(row[name_key]))

    tied_names = list(dict.fromkeys(tied_names))
    if len(tied_names) < 2:
        return context

    if len(tied_names) == 2:
        tie_text = f"{tied_names[0]} and {tied_names[1]}"
    else:
        tie_text = f"{', '.join(tied_names[:-1])}, and {tied_names[-1]}"

    summary = f"**Top Scorers Summary**: {tie_text} share the top spot with {top_value} goals each. Mention all co-leaders."
    note = f"\n**Tie Note**: The top value is shared by {tie_text}. Mention all of them as co-leaders.\n"
    return summary + note + context


def enforce_tied_top_scorers_in_answer(
    answer: str,
    intent: Intent,
    query_method: str,
    params: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> str:
    """Guarantee tied top-scorer answers mention every co-leader."""
    if not answer or not results:
        return answer

    is_goal_leader_query = intent == Intent.TOP_SCORERS or (
        query_method == "get_top_points_by_position" and params.get("sort_by") == "goals"
    )
    if not is_goal_leader_query:
        return answer

    metric_key = next((candidate for candidate in ("total_goals", "goals", "answer") if candidate in results[0]), None)
    if not metric_key:
        return answer

    top_value = results[0].get(metric_key)
    if top_value is None:
        return answer

    name_key = "player_name" if "player_name" in results[0] else "answer"
    tied_names = []
    for row in results:
        if row.get(metric_key) == top_value and row.get(name_key):
            tied_names.append(str(row[name_key]))

    tied_names = list(dict.fromkeys(tied_names))
    if len(tied_names) < 2:
        return answer

    missing_names = [name for name in tied_names if name.lower() not in answer.lower()]
    if not missing_names:
        return answer

    if len(tied_names) == 2:
        tie_text = f"{tied_names[0]} and {tied_names[1]}"
    else:
        tie_text = f"{', '.join(tied_names[:-1])}, and {tied_names[-1]}"

    tie_sentence = f"Top scorers are {tie_text}, with {top_value} goals each."
    stripped_answer = answer.strip()
    if stripped_answer:
        return f"{tie_sentence} {stripped_answer}"
    return tie_sentence


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


@app.get("/debug/embeddings")
async def debug_embeddings():
    """Debug: show embedding state on Railway."""
    import os
    manager = app_state["embedding_manager"]
    files = {}
    for key in ["mpnet", "minilm"]:
        path = f"embeddings/prebuilt/{key}_embeddings.pkl"
        files[path] = {"exists": os.path.exists(path), "size_mb": round(os.path.getsize(path) / 1_000_000, 1) if os.path.exists(path) else 0}
    return {
        "embedding_manager_exists": manager is not None,
        "model": getattr(manager, "model_key", None) if manager else None,
        "model_loaded": getattr(manager, "model", None) is not None if manager else False,
        "player_embeddings_count": len(getattr(manager, "player_embeddings", {})) if manager else 0,
        "embeddings_built": app_state["embeddings_built"],
        "prebuilt_files": files,
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
    
    # Get embedding count safely
    embedding_count = 0
    if app_state["embedding_manager"]:
        try:
            embedding_count = len(app_state["embedding_manager"].player_embeddings)
        except Exception as e:
            logger.warning(f"Failed to get embedding count: {e}")
            embedding_count = 0
    
    return {
        "status": "healthy",
        "neo4j": neo4j_status,
        "neo4j_stats": stats,
        "llm_available": app_state["llm_manager"] is not None,
        "embeddings_built": app_state["embeddings_built"],
        "embedding_count": embedding_count,
        "embeddings_building": app_state["embedding_building"],
        "embedding_build_error": app_state["embedding_build_error"],
    }


@app.post("/api/connection/connect", response_model=ConnectionResponse)
async def connect_neo4j(request: ConnectionRequest):
    """Connect to Neo4j database."""
    try:
        uri = (request.uri or NEO4J_URI).strip()
        username = (request.username or NEO4J_USER).strip()
        password = request.password if request.password is not None else NEO4J_PASSWORD

        if not password:
            password = NEO4J_PASSWORD

        conn = Neo4jConnection(uri, username, password)
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
        # Only return the initial greeting when this is the user's first message in the session
        if request.is_first_message and looks_like_small_talk(request.question):
            return QueryResponse(
                answer="Hi! 👋 I'm your FPL assistant — ask about players, teams, stats, or say 'compare' to pit two players against each other. What would you like to know?",
                intent=Intent.GENERAL_QUESTION.value,
                entities={"players": [], "teams": [], "seasons": [], "stats": [], "positions": []},
                cypher_query="",
                kg_context="",
                embedding_context=None,
                embedding_used=False,
                results=[],
                graph_data=None,
            )

        # Step 1: Intent Classification (lazy load)
        intent_classifier = get_intent_classifier()
        intent_result = intent_classifier.classify(request.question)

        # If the user's text explicitly mentions clean sheets, prefer the CLEAN_SHEETS intent
        q_lower = request.question.lower()
        if 'clean sheet' in q_lower or 'clean sheets' in q_lower:
            intent_result.intent = Intent.CLEAN_SHEETS
        
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
            cypher_context = add_tie_note_if_needed(intent_result.intent, query_method, params, results, cypher_context)
            executed_query = query
        
        if not results:
            query, query_params = CypherQueries.get_top_players_all_positions(
                season=params.get("season"),
                gameweek=params.get("gameweek"),
                limit_per_position=5
            )
            results = conn.execute_query(query, query_params)
            cypher_context = PromptBuilder.format_kg_context(results)
            cypher_context = add_tie_note_if_needed(intent_result.intent, query_method, params, results, cypher_context)
            executed_query = query
        
        # Step 5: Embedding search
        embedding_context = ""
        embedding_used = False
        
        if request.retrieval_method in ["Embeddings", "Hybrid"]:
            current_manager = app_state["embedding_manager"]
            if not current_manager:
                # No manager yet — create a lazy one with prebuilt
                manager = EmbeddingManager.__new__(EmbeddingManager)
                manager.model_key = request.embedding_model
                manager.model_info = EmbeddingManager.MODELS[request.embedding_model]
                manager.model = None
                manager.player_embeddings = {}
                manager.player_metadata = {}
                prebuilt_path = f"embeddings/prebuilt/{request.embedding_model}_embeddings.pkl"
                manager.load_embeddings(prebuilt_path)
                app_state["embedding_manager"] = manager
                if manager.player_embeddings:
                    app_state["embeddings_built"] = True
            elif current_manager.model_key != request.embedding_model:
                # Switch to different prebuilt model without loading transformer
                manager = EmbeddingManager.__new__(EmbeddingManager)
                manager.model_key = request.embedding_model
                manager.model_info = EmbeddingManager.MODELS[request.embedding_model]
                manager.model = None
                manager.player_embeddings = {}
                manager.player_metadata = {}
                prebuilt_path = f"embeddings/prebuilt/{request.embedding_model}_embeddings.pkl"
                manager.load_embeddings(prebuilt_path)
                app_state["embedding_manager"] = manager
                if manager.player_embeddings:
                    app_state["embeddings_built"] = True
            
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
            data_scope = f"the {params['season']} season" if "season" in params else "all seasons (2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26)"
            full_prompt = PromptTemplates.qa_template(
                question=request.question,
                kg_context=cypher_context,
                embedding_context=embedding_context if embedding_context else None,
                data_scope=data_scope,
                is_first_message=request.is_first_message,
                chat_history=request.chat_history,
                model_key=request.model  # Pass model key for model-specific prompting
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

            answer = enforce_tied_top_scorers_in_answer(
                answer=answer,
                intent=intent_result.intent,
                query_method=query_method,
                params=params,
                results=results,
            )
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
        with embedding_build_lock:
            if app_state["embedding_building"]:
                raise HTTPException(status_code=409, detail="Embedding build already in progress")

            app_state["embedding_building"] = True
            app_state["embedding_build_error"] = None

        # Initialize embedding manager if not already present
        if not app_state["embedding_manager"]:
            app_state["embedding_manager"] = EmbeddingManager(model_key=request.model)

        build_thread = Thread(
            target=run_embedding_build,
            args=(request.model, conn),
            daemon=True,
            name=f"embedding-build-{request.model}",
        )
        build_thread.start()

        return EmbeddingBuildResponse(
            success=True,
            count=len(app_state["embedding_manager"].player_embeddings) if app_state["embedding_manager"] else 0,
            message=f"Embedding build started in the background for {request.model}. Refresh status to see progress.",
            started=True,
            building=True,
            model=request.model,
        )
        
    except Exception as e:
        app_state["embedding_building"] = False
        logger.error(f"Failed to start embedding build: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start embedding build: {str(e)}")


@app.get("/api/trivia/new", response_model=TriviaQuestion)
async def get_trivia_question(difficulty: Optional[str] = None, conn=Depends(get_neo4j_conn)):
    """Generate a new trivia question, optionally filtered by difficulty."""
    try:
        trivia_gen = TriviaGenerator(conn)
        trivia_cache: TriviaQuestionCache = app_state["trivia_cache"]
        if not trivia_cache:
            raise HTTPException(status_code=503, detail="Trivia cache not initialized")

        # Map difficulty string to enum
        difficulty_enum = None
        if difficulty:
            try:
                difficulty_enum = Difficulty(difficulty.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}. Use easy, medium, or hard.")

        # Avoid serving very recent duplicate questions.
        question = None
        max_attempts = 12
        for _ in range(max_attempts):
            candidate = trivia_gen.generate_question(difficulty=difficulty_enum)
            if not candidate:
                continue
            if not trivia_cache.is_recent(candidate.question):
                question = candidate
                break
            question = candidate  # fallback
        
        if not question:
            raise HTTPException(status_code=500, detail="Failed to generate question")

        trivia_cache.mark_recent(question.question)
        trivia_cache.store(question)
        
        return TriviaQuestion(
            question=question.question,
            options=question.options,
            category=question.category.value,
            difficulty=question.difficulty.value,
            question_id=question.question_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trivia/answer", response_model=TriviaAnswerResponse)
async def check_trivia_answer(request: TriviaAnswerRequest, conn=Depends(get_neo4j_conn)):
    """Check trivia answer (simplified - store questions in session/cache in production)."""
    try:
        trivia_cache: TriviaQuestionCache = app_state["trivia_cache"]
        if not trivia_cache:
            raise HTTPException(status_code=503, detail="Trivia cache not initialized")

        stored_question = trivia_cache.get(request.question_id)
        if not stored_question:
            raise HTTPException(status_code=404, detail="Trivia question not found or expired")

        trivia_gen = TriviaGenerator(conn)
        correct, feedback = trivia_gen.check_answer(stored_question, request.answer)
        return TriviaAnswerResponse(
            correct=correct,
            feedback=feedback,
            correct_answer=stored_question.correct_answer if not correct else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PlayerStatsRequest(BaseModel):
    player_name: str
    season: Optional[str] = None


@app.post("/api/players/stats")
async def get_player_stats(request: PlayerStatsRequest, conn=Depends(get_neo4j_conn)):
    """Fetch stats for a specific player, either for a single season or combined career."""
    try:
        player_name = request.player_name
        season = request.season

        if season:
            query, params = CypherQueries.get_player_season_stats(player_name, season)
            results = conn.execute_query(query, params)
        else:
            # Combined career stats
            query = """
            MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
            WITH p, pos, 
                 SUM(r.total_points) AS total_points,
                 SUM(r.goals_scored) AS goals,
                 SUM(r.assists) AS assists,
                 SUM(r.clean_sheets) AS clean_sheets,
                 SUM(r.bonus) AS bonus,
                 SUM(r.minutes) AS minutes,
                 AVG(r.ict_index) AS avg_ict,
                 MAX(r.value) AS max_value,
                 MAX(r.selected) AS max_selected,
                 COUNT(f) AS games
            RETURN p.name AS player_name, pos.code AS position, 'All seasons' AS season,
                   total_points, goals, assists, clean_sheets, bonus,
                   minutes, round(avg_ict, 2) AS avg_ict_index, 
                   round(max_value / 10.0, 2) AS avg_value_millions, 
                   max_selected, games
            """
            results = conn.execute_query(query, {"player_name": player_name})

        if not results:
            return {"stats": None}

        # Format ICT index to 2 decimal places if it's not already
        stats = dict(results[0])
        if "avg_ict" in stats and stats["avg_ict"] is not None:
            stats["avg_ict_index"] = round(float(stats["avg_ict"]), 2)
        
        # Add avatar
        stats["avatar"] = get_player_avatar_url(player_name)
        
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/players/search")
async def search_players(request: PlayerSearchRequest, conn=Depends(get_neo4j_conn)):
    """Search for players by name, with accent-insensitive matching."""
    try:
        raw_query = request.query

        if not request.include_avatars:
            all_players = get_player_search_cache(conn)
            results = rank_player_search_results(all_players, raw_query, request.limit)
            return {"players": results}

        # First try the standard query (exact accent match)
        query, params = CypherQueries.search_players_by_name(raw_query, limit=request.limit)
        results = conn.execute_query(query, params)

        # If no results, fall back to accent-stripped search across all players
        if not results:
            all_query = """
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
            OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
            RETURN p.name AS player_name, pos.code AS position, t.name AS team_name
            """
            all_players = conn.execute_query(all_query, {})
            results = rank_player_search_results(all_players, raw_query, request.limit)
        else:
            # Enrich results with team names if missing
            enriched_results = []
            for r in results:
                row = dict(r)
                if not row.get("team_name"):
                    team_q = "MATCH (p:Player {name: $name})-[:PLAYS_FOR]->(t:Team) RETURN t.name AS team_name LIMIT 1"
                    team_res = conn.execute_query(team_q, {"name": row["player_name"]})
                    if team_res:
                        row["team_name"] = team_res[0]["team_name"]
                enriched_results.append(row)
            results = rank_player_search_results(enriched_results, raw_query, request.limit)

        if request.include_avatars:
            results = enrich_rows_with_avatars(results[: request.limit])
        else:
            results = results[: request.limit]

        return {"players": results[: request.limit]}
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
        return {"comparison": enrich_rows_with_avatars(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/search", response_model=ImageSearchResponse)
async def search_image(query: str, conn=Depends(get_neo4j_conn)):
    """Return the first Google image result for a search term, if configured."""

    def normalize(text: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    resolved_query = repair_mojibake(query)
    try:
        normalized_query = normalize(query)

        exact_query = "MATCH (p:Player) WHERE toLower(p.name) = toLower($name) RETURN p.name AS player_name LIMIT 1"
        exact_result = conn.execute_query(exact_query, {"name": query})
        if exact_result:
            resolved_query = repair_mojibake(exact_result[0]["player_name"])
        else:
            all_query = "MATCH (p:Player) RETURN p.name AS player_name"
            all_players = conn.execute_query(all_query, {})
            matching_names = [
                repair_mojibake(row["player_name"])
                for row in all_players
                if normalize(row["player_name"]) in normalized_query
            ]
            if not matching_names:
                matching_names = [
                    repair_mojibake(row["player_name"])
                    for row in all_players
                    if normalized_query in normalize(row["player_name"])
                ]
            if matching_names:
                resolved_query = max(matching_names, key=len)
    except Exception:
        resolved_query = query

    image_url = get_player_avatar_url(resolved_query)
    if not image_url:
        source = None
    elif "resources.premierleague.com" in image_url or "fantasy.premierleague.com" in image_url:
        source = "fpl_official"
    else:
        source = "google_custom_search"
    return ImageSearchResponse(query=resolved_query, image_url=image_url, source=source)


@app.get("/api/images/proxy")
async def proxy_image(url: str):
    """
    Fetch an external image and stream it from the backend origin.
    Handles rate limits gracefully by returning a fallback placeholder.
    """
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid image URL")

    try:
        # Some domains (like Wikipedia) block requests without a proper User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/*,*/*;q=0.8",
        }
        # Some CDNs (e.g., resources.premierleague.com) deny requests without a Referer.
        # Add a sensible Referer when requesting known image hosts to avoid Access Denied.
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or '').lower()
            if 'premierleague.com' in hostname or 'fantasy.premierleague.com' in hostname:
                headers['Referer'] = 'https://fantasy.premierleague.com/'
        except Exception:
            pass
        # Allow redirects and stream the response
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg")
        # Stream the content back to the client
        return StreamingResponse(io.BytesIO(response.content), media_type=content_type)
    except requests.exceptions.HTTPError as e:
        # Handle rate limits (429) or other HTTP errors gracefully
        status_code = e.response.status_code if e.response else 502
        if status_code == 429:
            # Return a 503 (Service Unavailable) with retry hint instead of crashing
            raise HTTPException(
                status_code=503,
                detail="Image source is rate-limited. Try again in a few moments.",
                headers={"Retry-After": "60"}
            )
        elif status_code == 404:
            raise HTTPException(status_code=404, detail="Image not found at source")
        elif status_code == 403:
            raise HTTPException(status_code=403, detail="Access forbidden by image source")
        else:
            raise HTTPException(status_code=502, detail=f"Image proxy failed: HTTP {status_code}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Image source timed out")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image proxy failed: {exc}")


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

        app_state["player_search_cache"] = None
        get_player_search_cache(conn)
        
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