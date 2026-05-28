import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { RefreshCw, Loader2, AlertCircle, CheckCircle, XCircle, Target } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import type { TriviaQuestion } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';

type AnswerState = 'idle' | 'correct' | 'wrong';

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: 'bg-green-100 text-green-700',
  medium: 'bg-yellow-100 text-yellow-700',
  hard: 'bg-red-100 text-red-700',
};

const CATEGORY_LABELS: Record<string, string> = {
  top_scorers: '⚽ Top Scorers',
  player_stats: '📊 Player Stats',
  team_facts: '🏟️ Team Facts',
  records: '🏆 Records',
  comparisons: '⚖️ Comparisons',
  true_false: '✅ True / False',
  multiple_choice: '🔢 Multiple Choice',
};

export function Trivia() {
  const { neo4jConnected, triviaScore, triviaTotal, incrementScore, incrementTotal, resetTriviaScore } =
    useAppStore();

  const [currentQuestion, setCurrentQuestion] = useState<TriviaQuestion | null>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [answerState, setAnswerState] = useState<AnswerState>('idle');
  const [feedback, setFeedback] = useState('');
  const [correctAnswer, setCorrectAnswer] = useState('');

  const questionQuery = useQuery({
    queryKey: ['trivia-question'],
    queryFn: () => apiService.getNewTriviaQuestion(),
    enabled: false,
  });

  const answerMutation = useMutation({
    mutationFn: ({ questionId, answer }: { questionId: string; answer: string }) =>
      apiService.checkTriviaAnswer(questionId, answer),
    onSuccess: (data) => {
      incrementTotal();
      if (data.correct) {
        incrementScore();
        setAnswerState('correct');
        setFeedback('Correct! Well done.');
      } else {
        setAnswerState('wrong');
        setFeedback(data.feedback || 'Not quite right.');
        setCorrectAnswer(data.correct_answer || '');
      }
    },
  });

  const loadNewQuestion = async () => {
    setSelectedAnswer(null);
    setAnswerState('idle');
    setFeedback('');
    setCorrectAnswer('');
    const result = await questionQuery.refetch();
    if (result.data) {
      setCurrentQuestion(result.data);
    }
  };

  const handleAnswer = (answer: string) => {
    if (answerState !== 'idle' || !currentQuestion) return;
    setSelectedAnswer(answer);
    answerMutation.mutate({ questionId: currentQuestion.question_id, answer });
  };

  const accuracy = triviaTotal > 0 ? Math.round((triviaScore / triviaTotal) * 100) : 0;

  return (
    <div className="flex flex-col h-full">
      <div className="bg-white border-b px-4 py-4 sm:px-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <h2 className="text-xl font-bold text-gray-900 sm:text-2xl">🎯 FPL FantasyTrivia</h2>
          <p className="text-sm text-gray-600">Test your Fantasy Premier League knowledge</p>
        </div>
        <button
          onClick={resetTriviaScore}
          className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700 border rounded-lg transition-colors"
        >
          Reset Score
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6 max-w-2xl mx-auto w-full sm:px-6">
        {/* Score cards */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 sm:gap-4">
          <ScoreCard label="Score" value={`${triviaScore} / ${triviaTotal}`} icon="🏆" />
          <ScoreCard label="Accuracy" value={`${accuracy}%`} icon="🎯" />
          <ScoreCard
            label="Last"
            value={triviaTotal === 0 ? '—' : answerState === 'correct' ? '✅' : answerState === 'wrong' ? '❌' : '—'}
            icon="🔥"
          />
        </div>

        {!neo4jConnected && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-yellow-900">Neo4j Not Connected</p>
              <p className="text-sm text-yellow-700 mt-1">Connecting to Neo4j... Please wait.</p>
            </div>
          </div>
        )}

        {!currentQuestion && (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-purple-100 rounded-full mb-4">
              <Target className="w-10 h-10 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready to play?</h3>
            <p className="text-gray-500 text-sm mb-6">
              Questions are generated live from the FPL knowledge graph.
            </p>
            <button
              onClick={loadNewQuestion}
              disabled={!neo4jConnected || questionQuery.isFetching}
              className="w-full px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mx-auto sm:w-auto"
            >
              {questionQuery.isFetching ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" /> Generating...
                </>
              ) : (
                '🎮 Start Playing'
              )}
            </button>
          </div>
        )}

        {questionQuery.isError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
            {handleApiError(questionQuery.error)}
          </div>
        )}

        <AnimatePresence mode="wait">
          {currentQuestion && (
            <motion.div
              key={currentQuestion.question_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-full font-medium">
                  {CATEGORY_LABELS[currentQuestion.category] ?? currentQuestion.category}
                </span>
                <span
                  className={`text-xs px-2 py-1 rounded-full font-medium capitalize ${
                    DIFFICULTY_COLORS[currentQuestion.difficulty] ?? 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {currentQuestion.difficulty}
                </span>
              </div>

              <div className="bg-gradient-to-br from-purple-600 to-indigo-700 text-white rounded-xl p-6">
                <p className="text-lg font-semibold leading-snug">{currentQuestion.question}</p>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {currentQuestion.options.map((option) => {
                  let style =
                    'border-gray-200 bg-white hover:border-purple-400 hover:bg-purple-50 cursor-pointer';

                  if (answerState !== 'idle') {
                    if (option === correctAnswer && answerState === 'wrong') {
                      style = 'border-green-400 bg-green-50 cursor-default';
                    } else if (option === selectedAnswer) {
                      style =
                        answerState === 'correct'
                          ? 'border-green-400 bg-green-50 cursor-default'
                          : 'border-red-400 bg-red-50 cursor-default';
                    } else {
                      style = 'border-gray-200 bg-gray-50 opacity-60 cursor-default';
                    }
                  }

                  return (
                    <button
                      key={option}
                      onClick={() => handleAnswer(option)}
                      disabled={answerState !== 'idle' || answerMutation.isPending}
                      className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-all text-sm font-medium flex items-center justify-between gap-3 ${style}`}
                    >
                      <span>{option}</span>
                      {answerState !== 'idle' && option === selectedAnswer && answerState === 'correct' && (
                        <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                      )}
                      {answerState !== 'idle' && option === selectedAnswer && answerState === 'wrong' && (
                        <XCircle className="w-5 h-5 text-red-600 shrink-0" />
                      )}
                      {answerState === 'wrong' && option === correctAnswer && (
                        <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>

              {answerState !== 'idle' && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`rounded-lg p-4 text-sm ${
                    answerState === 'correct'
                      ? 'bg-green-50 border border-green-200 text-green-800'
                      : 'bg-red-50 border border-red-200 text-red-800'
                  }`}
                >
                  {feedback}
                </motion.div>
              )}

              {answerState !== 'idle' && (
                <button
                  onClick={loadNewQuestion}
                  disabled={questionQuery.isFetching}
                  className="w-full py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {questionQuery.isFetching ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" /> Loading...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-5 h-5" /> Next Question
                    </>
                  )}
                </button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function ScoreCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="bg-white border rounded-xl p-4 text-center">
      <div className="text-2xl mb-1">{icon}</div>
      <div className="text-xl font-bold text-gray-900">{value}</div>
      <div className="text-xs text-gray-500 mt-0.5">{label}</div>
    </div>
  );
}
