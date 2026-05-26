"""
FPL FantasyTrivia - Graph-RAG Streamlit Application
Main entry point for the application
"""
import os
os.environ['TRANSFORMERS_BACKEND'] = 'pytorch'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import json
import time
import inspect

# Import project modules
from config.settings import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    HUGGINGFACE_API_TOKEN, APP_TITLE, APP_ICON, APP_DESCRIPTION,
    DATA_PATH, SEASONS, POSITIONS
)
from graph.connection import Neo4jConnection
from graph.queries import CypherQueries, QueryExecutor
from graph.data_loader import FPLDataLoader
from preprocessing.intent_classifier import IntentClassifier, Intent
from preprocessing.entity_extractor import EntityExtractor
from embeddings.embedding_manager import EmbeddingManager
from trivia.trivia_generator import TriviaGenerator, TriviaCategory, Difficulty
from llm.llm_manager import LLMManager, PromptBuilder
from llm.prompts import PromptTemplates
from utils.helpers import fuzzy_match_player

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .context-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .query-box {
        background-color: #e8f4ea;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
    }
    .trivia-question {
        font-size: 1.2em;
        font-weight: bold;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 10px 0;
    }
    .score-display {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
@st.cache_resource
def get_intent_classifier():
    """Cache the IntentClassifier across all sessions."""
    return IntentClassifier()


@st.cache_resource
def get_entity_extractor():
    """Cache the EntityExtractor across all sessions."""
    return EntityExtractor()


@st.cache_resource
def get_llm_manager(api_token: str):
    """Cache the LLMManager across all sessions."""
    if api_token:
        return LLMManager(api_token=api_token)
    return None


def init_session_state():
    """Initialize session state variables."""
    if "neo4j_connected" not in st.session_state:
        st.session_state.neo4j_connected = False
    if "graph_conn" not in st.session_state:
        st.session_state.graph_conn = None
        
    # Use cached resources
    st.session_state.intent_classifier = get_intent_classifier()
    st.session_state.entity_extractor = get_entity_extractor()
    
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = get_llm_manager(HUGGINGFACE_API_TOKEN)
        
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = None
    if "embeddings_built" not in st.session_state:
        st.session_state.embeddings_built = False
    if "embedding_count" not in st.session_state:
        st.session_state.embedding_count = 0
    if "trivia_score" not in st.session_state:
        st.session_state.trivia_score = 0
    if "trivia_total" not in st.session_state:
        st.session_state.trivia_total = 0
    if "current_trivia" not in st.session_state:
        st.session_state.current_trivia = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_query_info" not in st.session_state:
        st.session_state.last_query_info = None


def connect_neo4j(uri: str, user: str, password: str) -> bool:
    """Attempt to connect to Neo4j database."""
    try:
        conn = Neo4jConnection(uri, user, password)
        if conn.test_connection():
            st.session_state.graph_conn = conn
            st.session_state.neo4j_connected = True
            
            # Load known players for entity extraction to improve accuracy
            if "entity_extractor" in st.session_state:
                try:
                    query, _ = CypherQueries.get_all_player_names()
                    results = conn.execute_query(query)
                    if results:
                        players = {r['player_name'] for r in results}
                        st.session_state.entity_extractor.set_known_players(players)
                except Exception as e:
                    print(f"Warning: Failed to load player names: {e}")
            
            return True
    except Exception as e:
        st.error(f"Connection failed: {e}")
    return False


def init_llm_manager(api_token: str):
    """Initialize LLM manager with API token."""
    st.session_state.llm_manager = LLMManager(api_token=api_token)


def init_embedding_manager(model_key: str = "minilm"):
    """Initialize embedding manager."""
    st.session_state.embedding_manager = EmbeddingManager(model_key=model_key)


def create_graph_visualization(query_results: List[Dict], entities: Dict = None):
    """
    Create an interactive graph visualization from Cypher query results.
    
    Args:
        query_results: Results from a Cypher query
        entities: Extracted entities from the query
        
    Returns:
        Plotly figure with graph visualization
    """
    import networkx as nx
    import plotly.graph_objects as go
    
    if not query_results:
        return None
    
    G = nx.Graph()
    
    # Color scheme for different node types
    colors = {
        "player": "#1f77b4",      # Blue
        "team": "#ff7f0e",        # Orange
        "season": "#2ca02c",      # Green
        "position": "#d62728",    # Red
        "gameweek": "#9467bd",    # Purple
        "stat": "#8c564b",        # Brown
    }
    
    # Build nodes and edges based on result structure
    for i, result in enumerate(query_results[:20]):  # Limit to 20 for clarity
        # Player node
        if "player_name" in result:
            player = result["player_name"]
            points = result.get("total_points", result.get("points", ""))
            G.add_node(player, 
                      node_type="player", 
                      color=colors["player"],
                      label=f"{player}\n({points} pts)" if points else player)
            
            # Position relationship
            if "position" in result:
                pos = result["position"]
                G.add_node(pos, node_type="position", color=colors["position"], label=pos)
                G.add_edge(player, pos, relationship="PLAYS_POSITION")
            
            # Season relationship
            if "season" in result:
                season = result["season"]
                G.add_node(season, node_type="season", color=colors["season"], label=season)
                G.add_edge(player, season, relationship="PLAYED_IN")
        
        # Team nodes for fixture results
        if "home_team" in result and "away_team" in result:
            home = result["home_team"]
            away = result["away_team"]
            home_score = result.get("home_score", 0)
            away_score = result.get("away_score", 0)
            
            G.add_node(home, node_type="team", color=colors["team"], label=home)
            G.add_node(away, node_type="team", color=colors["team"], label=away)
            
            # Create fixture node
            gw = result.get("gameweek", i)
            fixture_id = f"GW{gw}: {home} vs {away}"
            G.add_node(fixture_id, node_type="gameweek", color=colors["gameweek"], 
                      label=f"{home_score}-{away_score}")
            G.add_edge(home, fixture_id, relationship="HOME")
            G.add_edge(away, fixture_id, relationship="AWAY")
    
    if len(G.nodes()) == 0:
        return None
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_colors = [G.nodes[node].get('color', '#1f77b4') for node in G.nodes()]
    node_labels = [G.nodes[node].get('label', str(node)) for node in G.nodes()]
    node_types = [G.nodes[node].get('node_type', 'unknown') for node in G.nodes()]
    
    # Hover text
    hover_text = [f"<b>{label}</b><br>Type: {ntype}" 
                  for label, ntype in zip(node_labels, node_types)]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_text,
        text=[str(node)[:15] + "..." if len(str(node)) > 15 else str(node) for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=node_colors,
            size=25,
            line=dict(width=2, color='white'),
            symbol='circle'
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="📊 Knowledge Graph Subgraph",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(
                    text="🔵 Player  🟠 Team  🟢 Season  🔴 Position  🟣 Gameweek",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    font=dict(size=10)
                )
            ]
        )
    )
    
    return fig


# Sidebar
def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title(f"{APP_ICON} {APP_TITLE}")
    st.sidebar.markdown(APP_DESCRIPTION)
    st.sidebar.divider()
    
    # Neo4j Connection
    st.sidebar.subheader("🗄️ Database Connection")
    
    with st.sidebar.expander("Neo4j Settings", expanded=not st.session_state.neo4j_connected):
        neo4j_uri = st.text_input("URI", value=NEO4J_URI, key="neo4j_uri")
        neo4j_user = st.text_input("Username", value=NEO4J_USER, key="neo4j_user")
        neo4j_password = st.text_input("Password", type="password", key="neo4j_password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", type="primary"):
                with st.spinner("Connecting..."):
                    if connect_neo4j(neo4j_uri, neo4j_user, neo4j_password):
                        st.success("Connected!")
                        st.rerun()
        with col2:
            if st.session_state.neo4j_connected:
                if st.button("Disconnect"):
                    if st.session_state.graph_conn:
                        st.session_state.graph_conn.close()
                    st.session_state.neo4j_connected = False
                    st.session_state.graph_conn = None
                    st.rerun()
    
    # Connection status
    if st.session_state.neo4j_connected:
        st.sidebar.success("✅ Neo4j Connected")
        if st.session_state.graph_conn:
            try:
                stats = st.session_state.graph_conn.get_database_stats()
                st.sidebar.caption(f"Nodes: {stats['total_nodes']:,} | Relationships: {stats['total_relationships']:,}")
            except:
                pass
    else:
        st.sidebar.warning("⚠️ Not Connected")
    
    st.sidebar.divider()
    
    # LLM Configuration
    st.sidebar.subheader("🤖 LLM Settings")
    
    with st.sidebar.expander("HuggingFace API", expanded=False):
        hf_token = st.text_input(
            "API Token", 
            type="password", 
            value=HUGGINGFACE_API_TOKEN,
            key="hf_token",
            help="Get your token from huggingface.co/settings/tokens"
        )
        if st.button("Set Token"):
            init_llm_manager(hf_token)
            st.success("Token set!")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        options=["gemma-2-2b", "mistral-7b", "llama-3-8b", "zephyr-7b", "qwen-2.5-72b"],
        index=0,
        key="selected_model"
    )
    
    st.sidebar.divider()
    
    # Retrieval Method
    st.sidebar.subheader("🔍 Retrieval Settings")
    retrieval_method = st.sidebar.radio(
        "Method",
        options=["Baseline (Cypher)", "Embeddings", "Hybrid"],
        index=2,
        key="retrieval_method"
    )
    
    # Embedding model selection (if using embeddings)
    if retrieval_method in ["Embeddings", "Hybrid"]:
        st.sidebar.info("🔮 **Embedding Mode Active**")
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            options=["minilm", "mpnet"],
            format_func=lambda x: "MiniLM (Fast)" if x == "minilm" else "MPNet (Quality)",
            key="embedding_model"
        )
        
        # Embedding status and build button
        if st.session_state.embeddings_built:
            st.sidebar.success(f"✅ {st.session_state.embedding_count:,} embeddings loaded")
        else:
            st.sidebar.warning("⚠️ Embeddings not built yet")
        
        if st.sidebar.button("🔮 Build Embeddings", help="Generate embeddings from Neo4j data"):
            if st.session_state.neo4j_connected:
                with st.spinner(f"Building embeddings using {embedding_model.upper()}..."):
                    try:
                        # Initialize embedding manager if needed
                        if not st.session_state.embedding_manager:
                            init_embedding_manager(embedding_model)
                        elif st.session_state.embedding_manager.model_key != embedding_model:
                            st.session_state.embedding_manager.switch_model(embedding_model)
                        
                        # Fetch player data from Neo4j
                        # First check if there's any data
                        count_query = "MATCH (p:Player) RETURN count(p) as count"
                        count_result = st.session_state.graph_conn.execute_query(count_query)
                        player_count = count_result[0]['count'] if count_result else 0
                        
                        # Get aggregated stats per player per season
                        query, query_params = CypherQueries.get_player_embeddings_data()
                        results = st.session_state.graph_conn.execute_query(query, query_params)
                        st.sidebar.write(f"Debug: Found {player_count} players, query returned {len(results) if results else 0} results")
                        
                        if not results:
                            st.sidebar.warning("⚠️ No player data found in Neo4j. Please load FPL data first using '📥 Load FPL Data' button below.")
                        else:
                            # Build embeddings
                            st.session_state.embedding_manager.build_player_embeddings(results)
                            st.session_state.embeddings_built = True
                            st.session_state.embedding_count = len(st.session_state.embedding_manager.player_embeddings)
                            
                            st.sidebar.success(f"✅ Built {st.session_state.embedding_count:,} embeddings!")
                            st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error building embeddings: {e}")
                        import traceback
                        st.sidebar.text(traceback.format_exc())
            else:
                st.sidebar.error("Connect to Neo4j first!")
    else:
        st.sidebar.info("📊 **Cypher Query Mode**")
    
    st.sidebar.divider()
    
    # Data loading option
    st.sidebar.divider()
    if st.sidebar.button("📥 Load FPL Data", help="Load CSV data into Neo4j"):
        if st.session_state.neo4j_connected:
            with st.spinner("Loading data... This may take a few minutes."):
                loader = FPLDataLoader(st.session_state.graph_conn)
                stats = loader.load_all(DATA_PATH)
                st.sidebar.success(f"Loaded {stats['total_nodes']:,} nodes!")
        else:
            st.sidebar.error("Connect to Neo4j first!")
    
    return selected_model, retrieval_method


# Main content tabs
def render_qa_tab(selected_model: str, retrieval_method: str):
    """Render the Q&A Assistant tab."""
    st.header("💬 FPL Q&A Assistant")
    st.markdown("Ask questions about Fantasy Premier League players, teams, and statistics.")
    
    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about FPL... (e.g., 'Who scored the most goals in 2022-23?')"):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            if not st.session_state.neo4j_connected:
                st.error("Please connect to Neo4j first!")
                return
            
            with st.spinner("Analyzing your question..."):
                # Step 1: Intent Classification
                intent_result = st.session_state.intent_classifier.classify(prompt)
                
                # Step 2: Entity Extraction
                entities = st.session_state.entity_extractor.extract(prompt)
                
                # Step 3: Get query parameters
                params = st.session_state.entity_extractor.get_query_parameters(entities)
                
                # Step 4: Execute appropriate query
                query_executor = QueryExecutor(st.session_state.graph_conn)
                cypher_context = ""
                executed_query = ""
                results = []
                
                try:
                    query_method = st.session_state.intent_classifier.get_query_type_for_intent(intent_result.intent)
                    
                    if query_method:
                        # Ensure required parameters have defaults
                        if "limit" not in params:
                            params["limit"] = 10
                        if "limit_per_position" not in params:
                            params["limit_per_position"] = 5
                        
                        # Special handling for gameweek queries - use gameweek-specific query
                        if query_method == "get_player_season_stats" and "gameweek" in params:
                            query_method = "get_player_gameweek_performance"
                            # gameweek query requires season - use latest if not specified
                            if "season" not in params:
                                params["season"] = "2022-23"
                        # Special handling for player stats - use all seasons if no season specified
                        elif query_method == "get_player_season_stats" and "season" not in params:
                            query_method = "get_player_all_seasons_stats"
                        
                        # Special handling for transfer queries - requires season and gameweek
                        if query_method == "get_most_transferred_players":
                            if "season" not in params:
                                params["season"] = "2022-23"  # Default to latest season
                            if "gameweek" not in params:
                                params["gameweek"] = 1  # Default to gameweek 1
                            # Detect transfer direction from query
                            if "out" in prompt.lower():
                                params["direction"] = "out"
                            else:
                                params["direction"] = "in"
                        
                        # Special handling for most_selected - requires season and gameweek
                        if query_method == "get_most_selected_players":
                            if "season" not in params:
                                params["season"] = "2022-23"
                            if "gameweek" not in params:
                                params["gameweek"] = 1
                        
                        # Get the query method and filter params to only those it accepts
                        method = getattr(CypherQueries, query_method)
                        sig = inspect.signature(method)
                        
                        # Check if all required parameters are available
                        required_params = [
                            p.name for p in sig.parameters.values() 
                            if p.default == inspect.Parameter.empty
                        ]
                        missing_params = [p for p in required_params if p not in params]
                        
                        if missing_params:
                            # Missing required parameters - use fallback
                            query, query_params = CypherQueries.get_top_players_all_positions(
                                season=params.get("season"),
                                gameweek=params.get("gameweek"),
                                limit_per_position=5
                            )
                            results = st.session_state.graph_conn.execute_query(query, query_params)
                            cypher_context = PromptBuilder.format_kg_context(results)
                            executed_query = query
                        else:
                            valid_params = {k: v for k, v in params.items() if k in sig.parameters}
                            
                            # Execute query
                            query, query_params = method(**valid_params)
                            results = st.session_state.graph_conn.execute_query(query, query_params)
                            
                            # For fixture queries with many results, create a summary
                            if query_method == "get_fixture_results" and len(results) > 60:
                                # Create summary for LLM
                                from collections import defaultdict
                                season_stats = defaultdict(lambda: {"total": 0, "wins": 0, "draws": 0, "losses": 0, "sample_fixtures": []})
                                
                                for idx, r in enumerate(results):
                                    season = r.get("season", "")
                                    team_name = params.get("team_name", "")
                                    is_home = r.get("home_team") == team_name
                                    home_score = r.get("home_score", 0)
                                    away_score = r.get("away_score", 0)
                                    
                                    if is_home:
                                        if home_score > away_score: result = "win"
                                        elif home_score < away_score: result = "loss"
                                        else: result = "draw"
                                    else:
                                        if away_score > home_score: result = "win"
                                        elif away_score < home_score: result = "loss"
                                        else: result = "draw"
                                    
                                    season_stats[season]["total"] += 1
                                    if result == "win": season_stats[season]["wins"] += 1
                                    elif result == "draw": season_stats[season]["draws"] += 1
                                    else: season_stats[season]["losses"] += 1
                                    
                                    if len(season_stats[season]["sample_fixtures"]) < 3:
                                        season_stats[season]["sample_fixtures"].append(r)
                                
                                summary_lines = [f"Total fixtures: {len(results)}"]
                                for season in sorted(season_stats.keys()):
                                    stats = season_stats[season]
                                    summary_lines.append(f"\n{season}: {stats['total']} fixtures - {stats['wins']} wins, {stats['draws']} draws, {stats['losses']} losses")
                                cypher_context = "\n".join(summary_lines)
                                full_context = PromptBuilder.format_kg_context(results, max_items=200)
                            else:
                                cypher_context = PromptBuilder.format_kg_context(results)
                                full_context = cypher_context
                            
                            executed_query = query
                    
                    if not results:
                        query, query_params = CypherQueries.get_top_players_all_positions(
                            season=params.get("season"),
                            gameweek=params.get("gameweek"),
                            limit_per_position=5
                        )
                        results = st.session_state.graph_conn.execute_query(query, query_params)
                        cypher_context = PromptBuilder.format_kg_context(results)
                        executed_query = query
                
                except Exception as e:
                    st.error(f"Query error: {e}")
                    cypher_context = "Error retrieving data from knowledge graph."
                
                # Step 5: Embedding search
                embedding_context = ""
                embedding_used = False
                if retrieval_method in ["Embeddings", "Hybrid"] and st.session_state.embedding_manager:
                    if st.session_state.embeddings_built:
                        try:
                            with st.status("🔮 Searching embeddings...", expanded=False) as status:
                                key_player = None
                                if entities.players:
                                    key_player = entities.players[0]
                                elif results and 'player_name' in results[0]:
                                    key_player = results[0]['player_name']
                                
                                if key_player:
                                    season = entities.seasons[0] if entities.seasons else "2022-23"
                                    similar_players = st.session_state.embedding_manager.find_similar_to_player(
                                        key_player, season=season, top_k=10, exclude_self=False
                                    )
                                else:
                                    similar_players = st.session_state.embedding_manager.find_similar_players(
                                        prompt, top_k=5
                                    )
                                embedding_context = PromptBuilder.format_embedding_context(similar_players)
                                embedding_used = True
                                status.update(label=f"🔮 Found {len(similar_players)} similar players", state="complete")
                        except Exception as e:
                            st.warning(f"Embedding search failed: {e}")
                
                # Step 6: Generate LLM response
                if st.session_state.llm_manager and st.session_state.llm_manager.client:
                    data_scope = f"the {params['season']} season" if "season" in params else "all seasons"
                    full_prompt = PromptTemplates.qa_template(
                        question=prompt,
                        kg_context=cypher_context,
                        embedding_context=embedding_context if embedding_context else None,
                        data_scope=data_scope
                    )
                    
                    with st.spinner("Generating answer..."):
                        response = st.session_state.llm_manager.generate(
                            full_prompt, 
                            model_key=selected_model,
                            timeout=60  # 1 minute timeout for UI
                        )
                        if response.success:
                            answer = response.text
                        else:
                            st.error(f"LLM Error: {response.error}")
                            if "timeout" in str(response.error).lower():
                                st.warning("The request timed out. This can happen with large models or heavy traffic. Try a smaller model like Gemma 2B.")
                            answer = f"Error: {response.error}"
                else:
                    answer = f"**Knowledge Graph data:**\n\n{cypher_context}"
                
                st.markdown(answer)
                
                if 'full_context' in locals() and full_context != cypher_context:
                    with st.expander("📋 Complete Fixture List", expanded=False):
                        st.text(full_context)
                
                st.session_state.last_query_info = {
                    "intent": intent_result.intent.value,
                    "entities": entities.to_dict(),
                    "cypher_query": executed_query,
                    "kg_context": cypher_context,
                    "embedding_context": embedding_context,
                    "embedding_used": embedding_used,
                    "retrieval_method": retrieval_method,
                    "results": results
                }
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Display query details
    if st.session_state.last_query_info:
        with st.expander("🔍 Query Details", expanded=False):
            st.json(st.session_state.last_query_info["entities"])
            st.code(st.session_state.last_query_info["cypher_query"], language="cypher")
            
            # Graph Visualization
            if st.session_state.last_query_info.get("results"):
                fig = create_graph_visualization(st.session_state.last_query_info["results"])
                if fig: st.plotly_chart(fig, use_container_width=True)


def render_trivia_tab():
    """Render the FantasyTrivia tab."""
    st.header("🎯 FPL FantasyTrivia")
    
    # Score display
    st.markdown(f"### Score: {st.session_state.trivia_score} / {st.session_state.trivia_total}")
    
    if st.button("🎲 New Question", type="primary"):
        if st.session_state.neo4j_connected:
            trivia_gen = TriviaGenerator(st.session_state.graph_conn)
            question = trivia_gen.generate_question()
            if question:
                st.session_state.current_trivia = question
        else:
            st.error("Connect to Neo4j first!")
    
    if st.session_state.current_trivia:
        q = st.session_state.current_trivia
        st.info(q.question)
        for i, option in enumerate(q.options):
            if st.button(option, key=f"opt_{i}"):
                trivia_gen = TriviaGenerator(st.session_state.graph_conn)
                correct, feedback = trivia_gen.check_answer(q, option)
                st.session_state.trivia_total += 1
                if correct: 
                    st.session_state.trivia_score += 1
                    st.success(feedback)
                else: 
                    st.error(feedback)
                st.session_state.current_trivia = None
                time.sleep(2)
                st.rerun()


def render_player_search_tab():
    """Render the Player Search tab."""
    st.header("🔎 Player Search")
    if not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j.")
        return
    
    search_query = st.text_input("Enter player name...")
    if search_query:
        query, params = CypherQueries.search_players_by_name(search_query)
        results = st.session_state.graph_conn.execute_query(query, params)
        for p in results[:5]:
            st.write(f"**{p['player_name']}** ({p['position']})")


def display_player_stats(player_name: str, season: str):
    """Display detailed stats for a player."""
    import plotly.express as px
    st.subheader(f"📊 {player_name}")
    query, params = CypherQueries.get_player_season_stats(player_name, season)
    results = st.session_state.graph_conn.execute_query(query, params)
    if results:
        stats = results[0]
        st.write(stats)


def render_comparison_tab(selected_model: str):
    """Render the Player Comparison tab."""
    st.header("⚖️ Player Comparison")
    if not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j.")
        return
    
    p1 = st.text_input("Player 1")
    p2 = st.text_input("Player 2")
    
    if st.button("Compare"):
        query, params = CypherQueries.compare_players(p1, p2)
        results = st.session_state.graph_conn.execute_query(query, params)
        if results:
            st.dataframe(pd.DataFrame(results))


def render_model_comparison_tab():
    """Render the Model Comparison tab."""
    st.header("🔬 Model Comparison")
    st.write("Compare different LLM models here.")


def main():
    """Main application entry point."""
    init_session_state()
    selected_model, retrieval_method = render_sidebar()
    
    st.title(f"{APP_ICON} {APP_TITLE}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Q&A Assistant", "🎯 FantasyTrivia", "🔎 Player Search", "⚖️ Compare Players", "🔬 Model Comparison"
    ])
    
    with tab1: render_qa_tab(selected_model, retrieval_method)
    with tab2: render_trivia_tab()
    with tab3: render_player_search_tab()
    with tab4: render_comparison_tab(selected_model)
    with tab5: render_model_comparison_tab()

if __name__ == "__main__":
    main()
