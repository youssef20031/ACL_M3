class IntentClassifier:
    def __init__(self):
        pass
    def classify(self, question):
        # Rule-based intent classification for different domains
        question_lower = question.lower()

        # Player performance analysis intents
        player_keywords = ['player', 'performance', 'stats', 'score', 'goal', 'assist', 'match', 'team', 'season']
        if any(keyword in question_lower for keyword in player_keywords):
            return 'player_performance'

        # General question intents
        question_keywords = ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'could', 'should']
        if any(keyword in question_lower for keyword in question_keywords):
            return 'general_question'

        # Recommendation intents
        recommendation_keywords = ['recommend', 'suggest', 'better', 'best', 'top', 'favorite', 'like']
        if any(keyword in question_lower for keyword in recommendation_keywords):
            return 'recommendation'

        # Entity search intents
        entity_keywords = ['who is', 'what is', 'where is', 'tell me about', 'describe', 'information']
        if any(keyword in question_lower for keyword in entity_keywords):
            return 'entity_search'

        # Default to general question
        return 'general_question'