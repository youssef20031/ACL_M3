import os
import time
import random
from huggingface_hub import InferenceClient

class IntentClassifier:
    def __init__(self, rate_limit_delay: float = 0.5, max_retries: int = 3, base_retry_delay: float = 1.0):
        """
        Initialize the IntentClassifier.
        
        Args:
            rate_limit_delay: Minimum seconds between API calls (default 0.5s)
            max_retries: Maximum number of retry attempts (default 3)
            base_retry_delay: Base delay in seconds for exponential backoff (default 1.0s)
        """
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set.")

        self.client = InferenceClient(token=hf_token)
        self.model_name = "google/gemma-2-9b-it"
        self.rate_limit_delay = rate_limit_delay + 2
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self._last_request_time = 0.0

    def _apply_rate_limit(self):
        """Enforce rate limiting by waiting if necessary."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _make_api_call(self, content: str):
        """Make API call with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self._apply_rate_limit()
                response = self.client.chat_completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=10,
                    temperature=0.1
                )
                return response
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    delay = self.base_retry_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    print(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    print(f"API call failed after {self.max_retries + 1} attempts: {e}")
        
        raise last_exception

    def classify(self, question):
        intents = [
            "player_stats",
            "player_comparison",
            "player_search",
            "top_scorers",
            "top_assisters",
            "top_points",
            "best_value",
            "team_analysis",
            "head_to_head",
            "fixture_results",
            "clean_sheets",
            "bonus_points",
            "ict_index",
            "transfers",
            "most_selected",
            "trivia",
            "recommendation",
            "general_question",
            "season_summary",
            "unknown"
        ]

        content = f"Classify the following question into exactly one of these categories: {intents}\nQuestion: '{question}'\nReturn only the category name."

        try:
            response = self._make_api_call(content)
            
            predicted_intent = response.choices[0].message.content.strip().lower()

            for intent in intents:
                if intent in predicted_intent:
                    return intent
            
            return "general question"

        except Exception as e:
            print(f"Error calling Cloud API: {e}")
            return "general question"