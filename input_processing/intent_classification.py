import os
from huggingface_hub import InferenceClient

class IntentClassifier:
    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set.")

        self.client = InferenceClient(token=hf_token)
        self.model_name = "google/gemma-2-2b-it"

    def classify(self, question):
        intents = ["player performance", "general question", "recommendation", "entity search"]
        content = f"Classify the following question into exactly one of these categories: {intents}\nQuestion: '{question}'\nReturn only the category name."

        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=10,
                temperature=0.1
            )
            
            predicted_intent = response.choices[0].message.content.strip().lower()

            for intent in intents:
                if intent in predicted_intent:
                    return intent
            
            return "general question"

        except Exception as e:
            print(f"Error calling Cloud API: {e}")
            return "general question"