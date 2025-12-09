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
    
    @staticmethod
    def qa_template(
        question: str,
        kg_context: str,
        embedding_context: Optional[str] = None,
        persona: str = "fpl_expert",
        data_scope: Optional[str] = None
    ) -> str:
        """
        Template for Q&A responses.
        
        Args:
            question: User's question
            kg_context: Context from Cypher queries
            embedding_context: Optional context from embedding search
            persona: Persona key
            data_scope: Description of the data scope (e.g., "all seasons (2020-21, 2021-22, 2022-23)" or "2022-23 season")
        """
        persona_text = PromptTemplates.PERSONAS.get(persona, PromptTemplates.PERSONAS["fpl_expert"])
        
        # Add data scope information
        scope_text = ""
        if data_scope:
            scope_text = f"\n**Data Scope**: This data covers {data_scope}.\n"
        
        template = f"""{persona_text}

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
1. Answer the question using ONLY the data provided above
2. If the data is insufficient, acknowledge what's missing
3. Be specific with numbers and statistics
4. Keep your response concise but informative
5. If the data scope mentions "all seasons", clearly state that the statistics are aggregated across all available seasons

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
        template = f"""{PromptTemplates.PERSONAS['fpl_expert']}

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
        template = f"""{PromptTemplates.PERSONAS['fpl_expert']}

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
