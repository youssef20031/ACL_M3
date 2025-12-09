"""
FPL FantasyTrivia - Graph-RAG Streamlit Application
Main entry point for the application
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
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
def init_session_state():
    """Initialize session state variables."""
    if "neo4j_connected" not in st.session_state:
        st.session_state.neo4j_connected = False
    if "graph_conn" not in st.session_state:
        st.session_state.graph_conn = None
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = None
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = None
    if "intent_classifier" not in st.session_state:
        st.session_state.intent_classifier = IntentClassifier()
    if "entity_extractor" not in st.session_state:
        st.session_state.entity_extractor = EntityExtractor()
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
        options=["gemma-2-2b", "mistral-7b", "llama-3-8b", "phi-3-mini", "qwen-2.5-72b"],
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
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            options=["minilm", "mpnet"],
            format_func=lambda x: "MiniLM (Fast)" if x == "minilm" else "MPNet (Quality)",
            key="embedding_model"
        )
    
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
                        
                        # Special handling for compare_players - use all seasons if no season specified  
                        if query_method == "compare_players" and "season" not in params:
                            # compare_players works without season parameter
                            pass
                        
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
                            cypher_context = PromptBuilder.format_kg_context(results)
                            executed_query = query
                    
                    # If no results or no query method, try a fallback
                    if not results:
                        # Fallback to top players by all positions
                        query, query_params = CypherQueries.get_top_players_all_positions(
                            limit_per_position=5
                        )
                        results = st.session_state.graph_conn.execute_query(query, query_params)
                        cypher_context = PromptBuilder.format_kg_context(results)
                        executed_query = query
                
                except Exception as e:
                    st.error(f"Query error: {e}")
                    # Final fallback - get all seasons summary
                    try:
                        query, query_params = CypherQueries.get_all_seasons_summary()
                        results = st.session_state.graph_conn.execute_query(query, query_params)
                        cypher_context = PromptBuilder.format_kg_context(results)
                        executed_query = query
                    except:
                        cypher_context = "Error retrieving data from knowledge graph."
                
                # Step 5: Embedding search (if hybrid/embedding mode)
                embedding_context = ""
                if retrieval_method in ["Embeddings", "Hybrid"] and st.session_state.embedding_manager:
                    try:
                        similar_players = st.session_state.embedding_manager.find_similar_players(
                            prompt, top_k=5
                        )
                        embedding_context = PromptBuilder.format_embedding_context(similar_players)
                    except:
                        embedding_context = "Embedding search not available."
                
                # Determine data scope for LLM context
                if "season" in params and params["season"]:
                    data_scope = f"the {params['season']} season"
                else:
                    data_scope = "all seasons (2020-21, 2021-22, 2022-23) - aggregated totals"
                
                # Step 6: Generate LLM response
                if st.session_state.llm_manager and st.session_state.llm_manager.client:
                    full_prompt = PromptTemplates.qa_template(
                        question=prompt,
                        kg_context=cypher_context,
                        embedding_context=embedding_context if embedding_context else None,
                        data_scope=data_scope
                    )
                    
                    response = st.session_state.llm_manager.generate(
                        full_prompt,
                        model_key=selected_model
                    )
                    
                    if response.success:
                        answer = response.text
                    else:
                        answer = f"LLM Error: {response.error}\n\nBased on the data:\n{cypher_context}"
                else:
                    answer = f"**Based on Knowledge Graph data:**\n\n{cypher_context}"
                
                st.markdown(answer)
                
                # Store query info for display
                st.session_state.last_query_info = {
                    "intent": intent_result.intent.value,
                    "entities": entities.to_dict(),
                    "cypher_query": executed_query,
                    "kg_context": cypher_context,
                    "embedding_context": embedding_context
                }
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Display query details (collapsible)
    if st.session_state.last_query_info:
        with st.expander("🔍 Query Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Intent & Entities")
                st.json({
                    "intent": st.session_state.last_query_info["intent"],
                    "entities": st.session_state.last_query_info["entities"]
                })
            
            with col2:
                st.subheader("Cypher Query")
                st.code(st.session_state.last_query_info["cypher_query"], language="cypher")
            
            st.subheader("Knowledge Graph Context")
            st.text(st.session_state.last_query_info["kg_context"])
            
            if st.session_state.last_query_info["embedding_context"]:
                st.subheader("Embedding Search Results")
                st.text(st.session_state.last_query_info["embedding_context"])


def render_trivia_tab():
    """Render the FantasyTrivia tab."""
    st.header("🎯 FPL FantasyTrivia")
    st.markdown("Test your Fantasy Premier League knowledge!")
    
    # Score display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
            <h2>Score: {st.session_state.trivia_score} / {st.session_state.trivia_total}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Trivia settings
    col1, col2, col3 = st.columns(3)
    with col1:
        difficulty = st.selectbox(
            "Difficulty",
            options=["easy", "medium", "hard"],
            format_func=str.title,
            key="trivia_difficulty"
        )
    with col2:
        category = st.selectbox(
            "Category",
            options=["random", "top_scorers", "player_stats", "records", "true_false", "multiple_choice"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="trivia_category"
        )
    with col3:
        if st.button("🎲 New Question", type="primary"):
            if st.session_state.neo4j_connected:
                trivia_gen = TriviaGenerator(st.session_state.graph_conn)
                
                cat = None if category == "random" else TriviaCategory(category)
                diff = Difficulty(difficulty)
                
                question = trivia_gen.generate_question(
                    category=cat,
                    difficulty=diff
                )
                
                if question:
                    st.session_state.current_trivia = question
                    st.session_state.trivia_answered = False
                else:
                    st.error("Could not generate question. Try again!")
            else:
                st.error("Connect to Neo4j first!")
    
    st.divider()
    
    # Display current question
    if st.session_state.current_trivia:
        question = st.session_state.current_trivia
        
        st.markdown(f"""
        <div class='trivia-question'>
            {question.question}
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"Category: {question.category.value.replace('_', ' ').title()} | Difficulty: {question.difficulty.value.title()}")
        
        # Answer options
        if question.category == TriviaCategory.TRUE_FALSE:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ True", use_container_width=True, key="btn_true"):
                    check_trivia_answer("True", question)
            with col2:
                if st.button("❌ False", use_container_width=True, key="btn_false"):
                    check_trivia_answer("False", question)
        else:
            # Multiple choice
            for i, option in enumerate(question.options):
                if st.button(f"{chr(65+i)}. {option}", key=f"option_{i}", use_container_width=True):
                    check_trivia_answer(option, question)
    else:
        st.info("Click 'New Question' to start playing!")
    
    # Reset score button
    st.divider()
    if st.button("🔄 Reset Score"):
        st.session_state.trivia_score = 0
        st.session_state.trivia_total = 0
        st.session_state.current_trivia = None
        st.rerun()


def check_trivia_answer(user_answer: str, question):
    """Check trivia answer and update score."""
    trivia_gen = TriviaGenerator(st.session_state.graph_conn)
    is_correct, feedback = trivia_gen.check_answer(question, user_answer)
    
    st.session_state.trivia_total += 1
    if is_correct:
        st.session_state.trivia_score += 1
        st.success(feedback)
        st.balloons()
    else:
        st.error(feedback)
    
    st.session_state.current_trivia = None
    time.sleep(2)
    st.rerun()


def render_player_search_tab():
    """Render the Player Search tab."""
    st.header("🔎 Player Search & Analysis")
    
    if not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j to search players.")
        return
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search Player",
            placeholder="Enter player name...",
            key="player_search"
        )
    with col2:
        position_filter = st.selectbox(
            "Position",
            options=["All", "GK", "DEF", "MID", "FWD"],
            key="position_filter"
        )
    
    if search_query:
        # Search players
        query, params = CypherQueries.search_players_by_name(search_query)
        results = st.session_state.graph_conn.execute_query(query, params)
        
        if position_filter != "All":
            results = [r for r in results if r.get("position") == position_filter]
        
        if results:
            st.subheader(f"Found {len(results)} players")
            
            # Display as cards
            cols = st.columns(3)
            for i, player in enumerate(results[:9]):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{player['player_name']}**")
                        st.caption(f"Position: {player['position']}")
                        
                        if st.button("View Stats", key=f"view_{i}"):
                            st.session_state.selected_player = player['player_name']
            
            # Display selected player stats
            if hasattr(st.session_state, 'selected_player') and st.session_state.selected_player:
                st.divider()
                display_player_stats(st.session_state.selected_player)
        else:
            st.info("No players found matching your search.")


def display_player_stats(player_name: str):
    """Display detailed stats for a player."""
    st.subheader(f"📊 {player_name} - All Seasons")
    
    query, params = CypherQueries.get_player_all_seasons_stats(player_name)
    results = st.session_state.graph_conn.execute_query(query, params)
    
    if results:
        stats = results[0]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", stats.get("total_points", 0))
        with col2:
            st.metric("Goals", stats.get("goals", 0))
        with col3:
            st.metric("Assists", stats.get("assists", 0))
        with col4:
            st.metric("Bonus", stats.get("bonus", 0))
        
        # Form chart
        form_query, form_params = CypherQueries.get_player_form_history(player_name, season)
        form_data = st.session_state.graph_conn.execute_query(form_query, form_params)
        
        if form_data:
            df = pd.DataFrame(form_data)
            fig = px.line(
                df, x="gameweek", y="points",
                title=f"{player_name} Points by Gameweek",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stats found for this player in the selected season.")


def render_comparison_tab(selected_model: str):
    """Render the Player Comparison tab."""
    st.header("⚖️ Player Comparison")
    
    if not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j to compare players.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Player 1", placeholder="e.g., Mohamed Salah", key="compare_p1")
    with col2:
        player2 = st.text_input("Player 2", placeholder="e.g., Erling Haaland", key="compare_p2")
    
    if st.button("Compare", type="primary") and player1 and player2:
        query, params = CypherQueries.compare_players(player1, player2)
        results = st.session_state.graph_conn.execute_query(query, params)
        
        if len(results) >= 2:
            # Display comparison table
            df = pd.DataFrame(results)
            
            st.subheader("Statistics Comparison")
            st.dataframe(df, use_container_width=True)
            
            # Radar chart
            categories = ["total_points", "goals", "assists", "bonus", "games"]
            
            fig = go.Figure()
            
            for _, row in df.iterrows():
                values = [row.get(cat, 0) for cat in categories]
                values.append(values[0])  # Close the radar
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=row['player_name']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Player Comparison Radar"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # LLM Analysis
            if st.session_state.llm_manager and st.session_state.llm_manager.client:
                with st.spinner("Generating analysis..."):
                    context = PromptBuilder.format_kg_context(results)
                    prompt = PromptTemplates.comparison_template(player1, player2, context)
                    
                    response = st.session_state.llm_manager.generate(prompt, model_key=selected_model)
                    
                    if response.success:
                        st.subheader("🤖 AI Analysis")
                        st.markdown(response.text)
        else:
            st.warning("Could not find both players. Please check the names.")


def render_model_comparison_tab():
    """Render the Model Comparison tab."""
    st.header("🔬 LLM Model Comparison")
    st.markdown("Compare responses from different language models.")
    
    test_query = st.text_area(
        "Test Query",
        value="Who was the best FPL pick in the 2022-23 season and why?",
        height=100
    )
    
    if st.button("Compare All Models", type="primary"):
        if not st.session_state.llm_manager or not st.session_state.llm_manager.client:
            st.error("Please set HuggingFace API token first!")
            return
        
        # Get context
        if st.session_state.neo4j_connected:
            query, params = CypherQueries.get_top_points_by_position(
                position="MID",
                limit=5
            )
            results = st.session_state.graph_conn.execute_query(query, params)
            context = PromptBuilder.format_kg_context(results)
        else:
            context = "No database connection. Using general knowledge."
        
        prompt = PromptTemplates.qa_template(test_query, context)
        
        # Compare models
        responses = {}
        models = ["gemma-2-2b", "mistral-7b", "llama-3-8b"]
        
        progress = st.progress(0)
        for i, model in enumerate(models):
            with st.spinner(f"Generating with {model}..."):
                responses[model] = st.session_state.llm_manager.generate(prompt, model_key=model)
                progress.progress((i + 1) / len(models))
        
        # Display results
        st.subheader("Results")
        
        cols = st.columns(len(models))
        for i, (model, response) in enumerate(responses.items()):
            with cols[i]:
                st.markdown(f"**{model}**")
                if response.success:
                    st.markdown(response.text)
                    st.caption(f"Time: {response.response_time:.2f}s")
                else:
                    st.error(response.error)


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get settings
    selected_model, retrieval_method = render_sidebar()
    
    # Main content area
    st.title(f"{APP_ICON} {APP_TITLE}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Q&A Assistant",
        "🎯 FantasyTrivia", 
        "🔎 Player Search",
        "⚖️ Compare Players",
        "🔬 Model Comparison"
    ])
    
    with tab1:
        render_qa_tab(selected_model, retrieval_method)
    
    with tab2:
        render_trivia_tab()
    
    with tab3:
        render_player_search_tab()
    
    with tab4:
        render_comparison_tab(selected_model)
    
    with tab5:
        render_model_comparison_tab()


if __name__ == "__main__":
    main()
