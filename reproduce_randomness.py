import sys
import os
import random

# Add the current directory to sys.path
sys.path.append(os.getcwd())

from trivia.trivia_generator import TriviaGenerator, TriviaCategory, Difficulty

class MockConnection:
    def execute_query(self, query, params=None):
        # Mock season query
        if "MATCH (s:Season) RETURN s.id" in query:
            return [{"season": "2020-21"}]
        
        # Mock Top Scorers Query (Deterministic)
        if "goals_scored" in query and "ORDER BY goals DESC" in query:
            return [{"answer": "Harry Kane", "goals": 23}]
        
        return []

def reproduction():
    mock_conn = MockConnection()
    generator = TriviaGenerator(mock_conn)
    
    print("Testing TOP_SCORERS determinism for Season 2020-21...")
    questions = []
    headers = []
    
    # Generate 5 questions
    for _ in range(5):
        q = generator.generate_question(category=TriviaCategory.TOP_SCORERS, season="2020-21")
        if q:
            questions.append(q.question)
            if q.question not in headers:
                headers.append(q.question)
    
    print(f"Generated {len(questions)} questions.")
    unique_questions = set(questions)
    print(f"Unique questions: {len(unique_questions)}")
    for uq in unique_questions:
        print(f" - {uq}")
        
    if len(questions) > 1 and len(unique_questions) == 1:
        print("CONFIRMED: Questions are identical for the same season.")
    else:
        print("Note: Questions vary (likely due to different templates or random positions).")

if __name__ == "__main__":
    reproduction()
