"""
LLM Manager for FPL Graph-RAG System
Handles integration with multiple LLMs via HuggingFace
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from huggingface_hub import InferenceClient
import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response with metadata."""
    text: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None


class LLMManager:
    """
    Manages LLM interactions for the Graph-RAG system.
    Supports multiple models via HuggingFace Inference API.
    """
    
    # Available models configuration
    # Using models that support chat/conversational task on HuggingFace Inference API
    MODELS = {
        "gemma-2-2b": {
            "name": "google/gemma-2-2b-it",
            "display_name": "Gemma 2 2B",
            "description": "Google's lightweight instruction-tuned model",
            "max_tokens": 1024,
            "temperature": 0.7,
            "use_chat": True,
        },
        "mistral-7b": {
            "name": "mistralai/Mistral-7B-Instruct-v0.3",
            "display_name": "Mistral 7B",
            "description": "High-quality open-source model from Mistral AI",
            "max_tokens": 1024,
            "temperature": 0.7,
            "use_chat": True,
        },
        "llama-3-8b": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "display_name": "Llama 3.1 8B",
            "description": "Meta's latest instruction-tuned model",
            "max_tokens": 1024,
            "temperature": 0.7,
            "use_chat": True,
        },
        "phi-3-mini": {
            "name": "microsoft/Phi-3-mini-4k-instruct",
            "display_name": "Phi-3 Mini",
            "description": "Microsoft's compact but powerful model",
            "max_tokens": 1024,
            "temperature": 0.7,
            "use_chat": True,
        },
        "qwen-2.5-72b": {
            "name": "Qwen/Qwen2.5-72B-Instruct",
            "display_name": "Qwen 2.5 72B",
            "description": "Alibaba's powerful instruction model",
            "max_tokens": 1024,
            "temperature": 0.7,
            "use_chat": True,
        }
    }
    
    def __init__(self, api_token: Optional[str] = None, default_model: str = "gemma-2-2b"):
        """
        Initialize LLM manager.
        
        Args:
            api_token: HuggingFace API token
            default_model: Default model to use
        """
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.default_model = default_model
        self.client = None
        self.response_history: List[LLMResponse] = []
        
        if self.api_token:
            self._init_client()
        else:
            logger.warning("No HuggingFace API token provided. Set HUGGINGFACE_API_TOKEN environment variable.")
    
    def _init_client(self):
        """Initialize HuggingFace inference client."""
        try:
            self.client = InferenceClient(token=self.api_token)
            logger.info("HuggingFace client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            self.client = None
    
    def set_api_token(self, token: str):
        """
        Set or update the API token.
        
        Args:
            token: HuggingFace API token
        """
        self.api_token = token
        self._init_client()
    
    def generate(
        self, 
        prompt: str, 
        model_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            model_key: Model to use (default if None)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with generated text and metadata
        """
        model_key = model_key or self.default_model
        
        if model_key not in self.MODELS:
            return LLMResponse(
                text="",
                model=model_key,
                tokens_used=0,
                response_time=0,
                success=False,
                error=f"Unknown model: {model_key}"
            )
        
        model_config = self.MODELS[model_key]
        model_name = model_config["name"]
        max_tokens = max_tokens or model_config["max_tokens"]
        temperature = temperature or model_config["temperature"]
        
        if not self.client:
            return LLMResponse(
                text="",
                model=model_key,
                tokens_used=0,
                response_time=0,
                success=False,
                error="HuggingFace client not initialized. Please provide API token."
            )
        
        start_time = time.time()
        
        try:
            # Use chat_completion for instruction-tuned models
            if model_config.get("use_chat", True):
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat_completion(
                    messages=messages,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                generated_text = response.choices[0].message.content.strip()
            else:
                # Fallback to text generation for non-chat models
                response = self.client.text_generation(
                    prompt,
                    model=model_name,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_full_text=False
                )
                generated_text = response.strip()
            
            response_time = time.time() - start_time
            
            # Estimate tokens (rough approximation)
            tokens_used = len(prompt.split()) + len(generated_text.split())
            
            result = LLMResponse(
                text=generated_text,
                model=model_key,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            result = LLMResponse(
                text="",
                model=model_key,
                tokens_used=0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
        
        self.response_history.append(result)
        return result
    
    def generate_with_all_models(
        self, 
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, LLMResponse]:
        """
        Generate responses from all available models for comparison.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            
        Returns:
            Dictionary mapping model key to response
        """
        results = {}
        
        for model_key in self.MODELS.keys():
            logger.info(f"Generating with {model_key}...")
            results[model_key] = self.generate(
                prompt, 
                model_key=model_key,
                max_tokens=max_tokens
            )
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return results
    
    def get_model_info(self, model_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_key: Model key (all models if None)
            
        Returns:
            Model information dictionary
        """
        if model_key:
            if model_key in self.MODELS:
                return {model_key: self.MODELS[model_key]}
            return {}
        return self.MODELS.copy()
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models with display info."""
        return [
            {
                "key": key,
                "name": config["display_name"],
                "description": config["description"]
            }
            for key, config in self.MODELS.items()
        ]
    
    def get_response_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about LLM responses.
        
        Returns:
            Dictionary with response statistics
        """
        if not self.response_history:
            return {"total_requests": 0}
        
        successful = [r for r in self.response_history if r.success]
        failed = [r for r in self.response_history if not r.success]
        
        stats = {
            "total_requests": len(self.response_history),
            "successful": len(successful),
            "failed": len(failed),
            "models_used": {},
        }
        
        if successful:
            stats["avg_response_time"] = sum(r.response_time for r in successful) / len(successful)
            stats["total_tokens"] = sum(r.tokens_used for r in successful)
            
            for response in successful:
                model = response.model
                if model not in stats["models_used"]:
                    stats["models_used"][model] = {"count": 0, "total_time": 0}
                stats["models_used"][model]["count"] += 1
                stats["models_used"][model]["total_time"] += response.response_time
        
        return stats
    
    def clear_history(self):
        """Clear response history."""
        self.response_history = []


class PromptBuilder:
    """Builds structured prompts for the Graph-RAG system."""
    
    @staticmethod
    def build_qa_prompt(
        question: str,
        context: str,
        persona: str = "FPL Expert"
    ) -> str:
        """
        Build a Q&A prompt with context and persona.
        
        Args:
            question: User's question
            context: Knowledge graph context
            persona: Assistant persona
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a {persona} Assistant specializing in Fantasy Premier League (FPL) statistics and analysis.

### Context (from Knowledge Graph):
{context}

### Task:
Answer the following question using ONLY the information provided in the context above. 
If the context doesn't contain enough information to answer, say so clearly.
Be concise, accurate, and helpful.

### Question:
{question}

### Answer:"""
        
        return prompt
    
    @staticmethod
    def build_recommendation_prompt(
        request: str,
        context: str,
        constraints: Optional[str] = None
    ) -> str:
        """
        Build a recommendation prompt.
        
        Args:
            request: User's recommendation request
            context: Player/team data from KG
            constraints: Optional constraints (budget, position, etc.)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an FPL Expert Assistant helping managers make informed decisions.

### Available Player Data:
{context}

### Request:
{request}

"""
        if constraints:
            prompt += f"""### Constraints:
{constraints}

"""
        
        prompt += """### Task:
Based on the player data provided, give specific recommendations with clear reasoning.
Consider factors like: form, fixtures, value, and consistency.
Provide actionable advice that the user can apply to their FPL team.

### Recommendations:"""
        
        return prompt
    
    @staticmethod
    def build_trivia_prompt(
        question: str,
        context: str,
        hint: Optional[str] = None
    ) -> str:
        """
        Build a prompt for trivia assistance.
        
        Args:
            question: Trivia question
            context: Relevant data from KG
            hint: Optional hint
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an FPL Trivia Master with encyclopedic knowledge of Fantasy Premier League.

### Question:
{question}

### Available Information:
{context}

"""
        if hint:
            prompt += f"""### Hint:
{hint}

"""
        
        prompt += """### Task:
Provide the correct answer and a brief, interesting explanation.
If the question is multiple choice, also explain why other options are incorrect.

### Answer:"""
        
        return prompt
    
    @staticmethod
    def build_comparison_prompt(
        player1: str,
        player2: str,
        context: str
    ) -> str:
        """
        Build a player comparison prompt.
        
        Args:
            player1: First player name
            player2: Second player name
            context: Statistics for both players
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an FPL Expert analyzing player performance.

### Player Statistics:
{context}

### Task:
Compare {player1} and {player2} based on the statistics provided.
Consider:
1. Overall FPL points and value
2. Goal contributions (goals + assists)
3. Consistency and form
4. Value for money (points per million)

Provide a clear recommendation on which player would be the better FPL pick and why.

### Comparison Analysis:"""
        
        return prompt
    
    @staticmethod
    def format_kg_context(results: List[Dict[str, Any]], max_items: int = 25) -> str:
        """
        Format Knowledge Graph results into readable context.
        
        Args:
            results: List of result dictionaries from Cypher query
            max_items: Maximum items to include (default 25 for multi-position queries)
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant data found in the Knowledge Graph."
        
        context_lines = []
        
        for i, result in enumerate(results[:max_items]):
            line_parts = []
            for key, value in result.items():
                # Format key name
                display_key = key.replace("_", " ").title()
                
                # Format value based on type
                if isinstance(value, float):
                    if value == int(value):
                        display_value = str(int(value))
                    else:
                        display_value = f"{value:.2f}"
                elif isinstance(value, int) and value > 10000:
                    display_value = f"{value:,}"
                else:
                    display_value = str(value)
                
                line_parts.append(f"{display_key}: {display_value}")
            
            context_lines.append(f"{i+1}. " + " | ".join(line_parts))
        
        return "\n".join(context_lines)
    
    @staticmethod
    def format_embedding_context(results: List[Any], max_items: int = 5) -> str:
        """
        Format embedding search results into context.
        
        Args:
            results: List of EmbeddingResult objects
            max_items: Maximum items to include
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No similar players found."
        
        context_lines = ["Similar players found (by semantic similarity):"]
        
        for i, result in enumerate(results[:max_items]):
            similarity_pct = result.similarity_score * 100
            metadata = result.metadata
            
            line = f"{i+1}. {result.player_name} (Similarity: {similarity_pct:.1f}%)"
            
            # Add key stats if available
            stats_parts = []
            if "total_points" in metadata:
                stats_parts.append(f"Points: {metadata['total_points']}")
            if "position" in metadata:
                stats_parts.append(f"Position: {metadata['position']}")
            if "season" in metadata:
                stats_parts.append(f"Season: {metadata['season']}")
            
            if stats_parts:
                line += f" - {', '.join(stats_parts)}"
            
            context_lines.append(line)
        
        return "\n".join(context_lines)
