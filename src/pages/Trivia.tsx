import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { RefreshCw, Loader2, AlertCircle, CheckCircle, XCircle, Target } from "lucide-react";
import { useAppStore } from "../store/appStore";
import { apiService, handleApiError } from "../services/api";
import type { TriviaQuestion } from "../services/api";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../utils/cn";

type AnswerState = "idle" | "correct" | "wrong";

const CATEGORY_LABELS: Record<string, string> = {
  top_scorers: "⚽ Top Scorers",
  player_stats: "📊 Player Stats",
  team_facts: "🏟️ Team Facts",
  records: "🏆 Records",
  comparisons: "⚖️ Comparisons",
  true_false: "✅ True / False",
  multiple_choice: "🔢 Multiple Choice",
};

export function Trivia() {
  const {
    neo4jConnected,
    triviaScore,
    triviaTotal,
    incrementScore,
    incrementTotal,
    resetTriviaScore,
    theme,
  } = useAppStore();
  const isDark = theme === "dark";

  const [currentQuestion, setCurrentQuestion] = useState<TriviaQuestion | null>(
    null
  );
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [answerState, setAnswerState] = useState<AnswerState>("idle");
  const [feedback, setFeedback] = useState("");
  const [correctAnswer, setCorrectAnswer] = useState("");

  const questionQuery = useQuery({
    queryKey: ["trivia-question"],
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
        setAnswerState("correct");
        setFeedback("Correct! Well done.");
      } else {
        setAnswerState("wrong");
        setFeedback(data.feedback || "Not quite right.");
        setCorrectAnswer(data.correct_answer || "");
      }
    },
  });

  const loadNewQuestion = async () => {
    setSelectedAnswer(null);
    setAnswerState("idle");
    setFeedback("");
    setCorrectAnswer("");
    const result = await questionQuery.refetch();
    if (result.data) {
      setCurrentQuestion(result.data);
    }
  };

  const handleAnswer = (answer: string) => {
    if (answerState !== "idle" || !currentQuestion) return;
    setSelectedAnswer(answer);
    answerMutation.mutate({
      questionId: currentQuestion.question_id,
      answer,
    });
  };

  const accuracy =
    triviaTotal > 0 ? Math.round((triviaScore / triviaTotal) * 100) : 0;

  return (
    <div className="flex flex-col h-full">
      <div
        className={cn(
          "border-b px-4 py-4 sm:px-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between",
          isDark ? "border-slate-800 bg-slate-950/80" : "bg-white"
        )}
      >
        <div className="min-w-0">
          <h2
            className={cn(
              "text-xl font-bold sm:text-2xl",
              isDark ? "text-slate-100" : "text-gray-900"
            )}
          >
            🎯 FPL FantasyTrivia
          </h2>
          <p
            className={cn(
              "text-sm",
              isDark ? "text-slate-400" : "text-gray-600"
            )}
          >
            Test your Fantasy Premier League knowledge
          </p>
        </div>
        <button
          onClick={resetTriviaScore}
          className={cn(
            "rounded-lg border px-3 py-1.5 text-xs transition-colors",
            isDark
              ? "border-slate-700 text-slate-400 hover:text-slate-200"
              : "border-gray-300 text-gray-500 hover:text-gray-700"
          )}
        >
          Reset Score
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6 max-w-2xl mx-auto w-full sm:px-6">
        {!neo4jConnected && (
          <div
            className={cn(
              "rounded-lg border p-4 flex items-start gap-3",
              isDark ? "border-amber-400/20 bg-amber-500/10" : "border-yellow-200 bg-yellow-50"
            )}
          >
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 shrink-0" />
            <div>
              <p
                className={cn(
                  "font-medium",
                  isDark ? "text-amber-100" : "text-yellow-900"
                )}
              >
                Neo4j Not Connected
              </p>
              <p
                className={cn(
                  "text-sm mt-1",
                  isDark ? "text-amber-200" : "text-yellow-700"
                )}
              >
                Connecting to Neo4j... Please wait.
              </p>
            </div>
          </div>
        )}

        {!currentQuestion && (
          <div className="text-center py-12">
            <div
              className={cn(
                "mb-4 inline-flex h-20 w-20 items-center justify-center rounded-full",
                isDark ? "bg-violet-500/15" : "bg-purple-100"
              )}
            >
              <Target
                className={cn(
                  "h-10 w-10",
                  isDark ? "text-violet-200" : "text-purple-600"
                )}
              />
            </div>
            <h3
              className={cn(
                "mb-2 text-lg font-semibold",
                isDark ? "text-slate-100" : "text-gray-900"
              )}
            >
              Ready to play?
            </h3>
            <p
              className={cn(
                "mb-6 text-sm",
                isDark ? "text-slate-400" : "text-gray-500"
              )}
            >
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
                "🎮 Start Playing"
              )}
            </button>
          </div>
        )}

        {questionQuery.isError && (
          <div
            className={cn(
              "rounded-lg border p-4 text-sm",
              isDark
                ? "border-red-400/20 bg-red-500/10 text-red-200"
                : "border-red-200 bg-red-50 text-red-700"
            )}
          >
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
                <span
                  className={cn(
                    "rounded-full px-2 py-1 text-xs font-medium",
                    isDark
                      ? "bg-violet-500/15 text-violet-200"
                      : "bg-purple-100 text-purple-700"
                  )}
                >
                  {CATEGORY_LABELS[currentQuestion.category] ??
                    currentQuestion.category}
                </span>
                <span
                  className={cn(
                    "rounded-full px-2 py-1 text-xs font-medium capitalize",
                    currentQuestion.difficulty === "easy" &&
                      (isDark
                        ? "bg-emerald-500/15 text-emerald-200"
                        : "bg-green-100 text-green-700"),
                    currentQuestion.difficulty === "medium" &&
                      (isDark
                        ? "bg-amber-500/15 text-amber-200"
                        : "bg-yellow-100 text-yellow-700"),
                    currentQuestion.difficulty === "hard" &&
                      (isDark
                        ? "bg-red-500/15 text-red-200"
                        : "bg-red-100 text-red-700")
                  )}
                >
                  {currentQuestion.difficulty}
                </span>
              </div>

              <div className="bg-gradient-to-br from-purple-600 to-indigo-700 text-white rounded-xl p-6">
                <p className="text-lg font-semibold leading-snug">
                  {currentQuestion.question}
                </p>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {currentQuestion.options.map((option) => {
                  let style =
                    isDark
                      ? "border-slate-700 bg-slate-900/60 hover:border-violet-400/40 hover:bg-slate-900 cursor-pointer text-slate-100"
                      : "border-gray-200 bg-white hover:border-purple-400 hover:bg-purple-50 cursor-pointer";

                  if (answerState !== "idle") {
                    if (option === correctAnswer && answerState === "wrong") {
                      style = isDark
                        ? "border-emerald-500/40 bg-emerald-500/15 cursor-default text-slate-100"
                        : "border-green-400 bg-green-100 cursor-default";
                    } else if (option === selectedAnswer) {
                      style =
                        answerState === "correct"
                          ? isDark
                            ? "border-emerald-500/40 bg-emerald-500/15 cursor-default text-slate-100"
                            : "border-green-400 bg-green-100 cursor-default"
                          : isDark
                          ? "border-red-500/40 bg-red-500/15 cursor-default text-slate-100"
                          : "border-red-400 bg-red-50 cursor-default";
                    } else {
                      style = isDark
                        ? "border-slate-700 bg-slate-900/40 opacity-60 cursor-default text-slate-100"
                        : "border-gray-200 bg-gray-50 opacity-60 cursor-default";
                    }
                  }

                  return (
                    <button
                      key={option}
                      onClick={() => handleAnswer(option)}
                      disabled={
                        answerState !== "idle" || answerMutation.isPending
                      }
                      className={cn(
                        "flex w-full items-center justify-between gap-3 rounded-lg border-2 px-4 py-3 text-left text-sm font-medium transition-all",
                        style
                      )}
                    >
                      <span>{option}</span>
                      {answerState !== "idle" &&
                        option === selectedAnswer &&
                        answerState === "correct" && (
                          <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                        )}
                      {answerState !== "idle" &&
                        option === selectedAnswer &&
                        answerState === "wrong" && (
                          <XCircle className="w-5 h-5 text-red-600 shrink-0" />
                        )}
                      {answerState === "wrong" &&
                        option === correctAnswer && (
                          <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                        )}
                    </button>
                  );
                })}
              </div>

              {answerState !== "idle" && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    "rounded-lg border p-4 text-sm",
                    answerState === "correct"
                      ? isDark
                        ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-100"
                        : "border-green-300 bg-green-100 text-green-900"
                      : isDark
                      ? "border-red-500/40 bg-red-500/15 text-red-100"
                      : "border-red-200 bg-red-50 text-red-800"
                  )}
                >
                  {feedback}
                </motion.div>
              )}

              {answerState !== "idle" && (
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

        {currentQuestion && (
          <div className="grid grid-cols-2 gap-3 sm:gap-4">
            <ScoreCard
              label="Score"
              value={`${triviaScore} / ${triviaTotal}`}
              icon="🏆"
            />
            <ScoreCard
              label="Accuracy"
              value={`${accuracy}%`}
              icon="🎯"
            />
          </div>
        )}
      </div>
    </div>
  );
}

function ScoreCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: string;
}) {
  const { theme } = useAppStore();
  const isDark = theme === "dark";

  return (
    <div
      className={cn(
        "rounded-xl border p-4 text-center",
        isDark
          ? "border-slate-800 bg-slate-900/80 text-slate-100"
          : "border-gray-200 bg-white"
      )}
    >
      <div className="mb-1 text-2xl">{icon}</div>
      <div
        className={cn(
          "text-xl font-bold",
          isDark ? "text-slate-100" : "text-gray-900"
        )}
      >
        {value}
      </div>
      <div
        className={cn(
          "mt-0.5 text-xs",
          isDark ? "text-slate-400" : "text-gray-500"
        )}
      >
        {label}
      </div>
    </div>
  );
}
