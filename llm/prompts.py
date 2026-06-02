"""
Prompt Templates for FPL Graph-RAG System
"""
from typing import Dict, Any, Optional, List


class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    # System personas
    PERSONAS = {
        "fpl_expert": """You are an expert Fantasy Premier League (FPL) analyst with deep knowledge of player statistics, 
team formations, and game strategies. You provide accurate, data-driven insights based on the information provided.""",
        
        "trivia_master": """You are an entertaining FPL Trivia Master who makes learning about Fantasy Premier League fun. 
You provide accurate answers with interesting facts and context.""",
        
        "transfer_advisor": """You are a strategic FPL Transfer Advisor who helps managers make optimal transfer decisions. 
You consider form, fixtures, value, and team balance when making recommendations.""",
        
        "stats_analyst": """You are a detailed FPL Statistics Analyst who excels at breaking down player and team performance metrics. 
You present data clearly and draw meaningful conclusions."""
    }

    # Friendly, conversational persona to improve user interactions
    PERSONAS["conversational_fpl"] = (
        "You are a friendly, conversational Fantasy Premier League (FPL) assistant. Speak naturally and warmly, acknowledge the user, "
        "and ask a brief clarifying question if the query is ambiguous. Keep responses concise and helpful while using the provided data. "
        "When appropriate, offer next steps or follow-up suggestions (e.g., 'Would you like a player comparison?')."
    )
    
    @staticmethod
    def qa_template(
        question: str,
        kg_context: str,
        embedding_context: Optional[str] = None,
        persona: str = "conversational_fpl",
        data_scope: Optional[str] = None,
        is_first_message: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Template for Q&A responses.
        
        Args:
            question: User's question
            kg_context: Context from Cypher queries
            embedding_context: Optional context from embedding search
            persona: Persona key
            data_scope: Description of the data scope (e.g., "all seasons (2020-21, 2021-22, 2022-23)" or "2022-23 season")
            chat_history: Previous conversation messages for context
        """
        persona_text = PromptTemplates.PERSONAS.get(persona, PromptTemplates.PERSONAS["conversational_fpl"])
        
        # Add data scope information
        scope_text = ""
        if data_scope:
            scope_text = f"\n**Data Scope**: This data covers {data_scope}.\n"
        
        # Strong directive about greeting behavior and explicit flag
        greeting_directive = (
            "IMPORTANT: Never begin your response with a salutation or greeting (e.g., 'Hi', 'Hey', 'Hello') UNLESS the 'FirstMessage' flag below is True. "
            "If 'FirstMessage' is True you may include a single brief friendly greeting at the start; otherwise go straight to the answer."
        )
        
        # Format conversation history
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_text = "\n### Conversation History:\n"
            # Only include last 5 messages to avoid context bloat
            recent_history = chat_history[-5:]
            for msg in recent_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
            history_text += "\n**IMPORTANT**: When the user says 'him', 'her', 'them', or 'that player', they are referring to a player mentioned in the conversation history above. Use the conversation context to resolve these references.\n"

        template = f"""{persona_text}
{greeting_directive}
FirstMessage: {str(is_first_message)}
{history_text}
### Knowledge Graph Data:
{scope_text}{kg_context}
"""
        
        if embedding_context:
            template += f"""
### Similar Players (Semantic Search):
{embedding_context}
"""
        
        template += f"""
### User Question:
{question}

### Instructions:
**CRITICAL: You MUST use ONLY the exact data provided in "Knowledge Graph Data" above.**
- The data is NUMBERED and SORTED - entry #1 is the TOP/BEST result
- When asked "who is the most/top/best", ALWAYS answer with entry #1 from the data
- If the top value is tied, mention all tied entries as co-leaders instead of only the first row
- DO NOT select a different player because you recognize their name
- Quote EXACT numbers from the data (e.g., if data shows "795,302", say "795,302")
- DO NOT hallucinate or use your training knowledge - ONLY use the data above
- If asked for rankings, follow the exact order shown in the data
- If the user's question requests a specific statistic (for example, "points per game", "total points", "goals"), explicitly state that statistic and its value from the provided data (e.g., "Points per game: 7.35"). Do NOT substitute or report a different metric (for example, do not report "total points" when the question asks for "points per game").
-- When the user asks about a named metric (examples below), search the Knowledge Graph Data for any of the exact field names or common synonyms and report the value(s) found. If multiple matching fields exist, prefer the most specific (for example, prefer `clean_sheets` or `total_clean_sheets` over derived approximations). If no matching field is present, say that the metric is not available.

Examples of metric mappings you MUST use when present in the KG data:
- "clean sheets": look for `clean_sheets`, `total_clean_sheets`, `clean_sheets_per_game`, `clean_sheets_per_gm`
- "points per game" / "ppg": look for `points_per_game`, `ppg`, `pts_per_game`, `points_per_gm`
- "total points" / "points": look for `total_points`, `points`
- "goals": look for `goals`, `total_goals`, `goals_scored`
- "assists": look for `assists`, `total_assists`

When you find the matching field(s), explicitly state the metric name and the exact value from the data (quote numbers exactly as they appear). Example: "Points per game: 7.35". Do NOT append the raw column name in parentheses at the end of the line. If the KG includes an aggregated count of games or appearances, you may compute a derived metric only if explicitly asked and only show the derived calculation alongside the original fields used.

### Answer:"""
        
        return template
    
    @staticmethod
    def trivia_template(
        question: str,
        options: List[str],
        kg_context: str,
        difficulty: str = "medium"
    ) -> str:
        """
        Template for trivia question assistance.
        """
        template = f"""{PromptTemplates.PERSONAS['trivia_master']}

### Trivia Question ({difficulty.upper()} difficulty):
{question}

### Options:
{chr(10).join(f'{chr(65+i)}. {opt}' for i, opt in enumerate(options))}

### Available Data:
{kg_context}

### Instructions:
1. Identify the correct answer from the options
2. Explain why it's correct with specific data
3. Briefly mention why other options are incorrect
4. Add an interesting related fact if possible

### Response:"""
        
        return template
    
    @staticmethod
    def comparison_template(
        player1: str,
        player2: str,
        kg_context: str,
        comparison_type: str = "general"
    ) -> str:
        """
        Template for player comparisons.
        """
        template = f"""{PromptTemplates.PERSONAS['stats_analyst']}

### Player Comparison: {player1} vs {player2}

### Statistics:
{kg_context}

### Comparison Focus: {comparison_type.replace('_', ' ').title()}

### Instructions:
1. Compare both players across key metrics
2. Highlight strengths and weaknesses of each
3. Consider value for money (points per million)
4. Provide a clear recommendation with reasoning
5. Format your response with clear sections

### Analysis:"""
        
        return template
    
    @staticmethod
    def recommendation_template(
        request: str,
        kg_context: str,
        budget: Optional[float] = None,
        position: Optional[str] = None,
        num_picks: int = 3
    ) -> str:
        """
        Template for player recommendations.
        """
        constraints = []
        if budget:
            constraints.append(f"Budget: £{budget}m")
        if position:
            position_names = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}
            constraints.append(f"Position: {position_names.get(position, position)}")
        
        constraint_text = "\n".join(constraints) if constraints else "No specific constraints"
        
        template = f"""{PromptTemplates.PERSONAS['transfer_advisor']}

### Request:
{request}

### Constraints:
{constraint_text}

### Available Player Data:
{kg_context}

### Instructions:
1. Recommend {num_picks} players that best fit the request
2. For each player, explain:
   - Key statistics supporting the pick
   - Recent form and consistency
   - Value proposition (points per million)
   - Any risks or concerns
3. Rank recommendations by priority
4. Consider team balance if mentioned

### Recommendations:"""
        
        return template
    
    @staticmethod
    def summary_template(
        topic: str,
        kg_context: str,
        time_period: str = "season"
    ) -> str:
        """
        Template for generating summaries.
        """
        template = f"""{PromptTemplates.PERSONAS['conversational_fpl']}

### Topic: {topic}
### Period: {time_period}

### Data:
{kg_context}

### Instructions:
1. Provide a comprehensive summary of the topic
2. Highlight key statistics and trends
3. Identify standout performers
4. Note any interesting patterns or anomalies
5. Keep the summary engaging and informative

### Summary:"""
        
        return template
    
    @staticmethod
    def error_handling_template(
        original_query: str,
        error_type: str
    ) -> str:
        """
        Template for handling errors gracefully.
        """
        error_messages = {
            "no_data": "I couldn't find specific data for your query in the knowledge graph.",
            "ambiguous": "Your query seems ambiguous. Could you be more specific?",
            "invalid_entity": "I couldn't identify the player or team you mentioned.",
            "no_context": "I don't have enough context to answer this question accurately."
        }
        
        message = error_messages.get(error_type, "I encountered an issue processing your request.")
        
        template = f"""I apologize, but {message}

### Your Question:
{original_query}

### Suggestions:
- Try specifying a season (e.g., '2022-23')
- Use full player names when possible
- Ask about specific statistics (goals, assists, points)
- Mention the team or position if relevant

### Example Queries:
- "Who was the top scorer in 2022-23?"
- "Compare Mohamed Salah and Erling Haaland"
- "Best value midfielders last season"
- "How many clean sheets did Arsenal keep?"

Would you like to try rephrasing your question?"""
        
        return template
    
    @staticmethod
    def hybrid_context_template(
        question: str,
        cypher_context: str,
        embedding_context: str,
        retrieval_method: str = "hybrid"
    ) -> str:
        """
        Template that combines both retrieval methods.
        """
        template = f"""{PromptTemplates.PERSONAS['conversational_fpl']}

### Retrieval Method: {retrieval_method.upper()}

### Structured Query Results (Cypher):
{cypher_context}

### Semantic Search Results (Embeddings):
{embedding_context}

### Question:
{question}

### Instructions:
1. Consider information from BOTH retrieval methods
2. Prioritize structured data for specific statistics
3. Use semantic results for context and similar players
4. Clearly state which source supports your answer
5. Note any discrepancies between sources

### Answer:"""
        
        return template
