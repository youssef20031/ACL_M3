import { useState, useCallback, useEffect, useRef } from "react";
import { CheckCircle, XCircle, Target, Loader2, AlertCircle } from "lucide-react";
import { useAppStore } from "../store/appStore";
import { apiService, handleApiError } from "../services/api";
import type { TriviaQuestion } from "../services/api";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../utils/cn";

// ─── Types ────────────────────────────────────────────────────────────────────

type Difficulty = "easy" | "medium" | "hard";
type View = "select" | "quiz" | "results";
type AnswerState = "idle" | "correct" | "wrong";

interface QuizAnswer {
  questionText: string;
  selected: string;
  correct: string;
  isCorrect: boolean;
}

const QUESTIONS_PER_TEST = 15;

const DIFFICULTY_CONFIG: Record<
  Difficulty,
  { label: string; color: string; bg: string; darkBg: string; darkColor: string; emoji: string; description: string }
> = {
  easy: {
    label: "Easy",
    emoji: "🟢",
    description: "Basic FPL facts — positions, top scorers, simple stats",
    color: "text-emerald-700",
    bg: "bg-emerald-50 border-emerald-200 hover:bg-emerald-100",
    darkColor: "text-emerald-300",
    darkBg: "bg-emerald-500/10 border-emerald-500/30 hover:bg-emerald-500/20",
  },
  medium: {
    label: "Medium",
    emoji: "🟡",
    description: "Season stats, assists, comparisons between players",
    color: "text-amber-700",
    bg: "bg-amber-50 border-amber-200 hover:bg-amber-100",
    darkColor: "text-amber-300",
    darkBg: "bg-amber-500/10 border-amber-500/30 hover:bg-amber-500/20",
  },
  hard: {
    label: "Hard",
    emoji: "🔴",
    description: "Records, GW scores, goalkeeper saves, tricky facts",
    color: "text-red-700",
    bg: "bg-red-50 border-red-200 hover:bg-red-100",
    darkColor: "text-red-300",
    darkBg: "bg-red-500/10 border-red-500/30 hover:bg-red-500/20",
  },
};

const CATEGORY_LABELS: Record<string, string> = {
  top_scorers: "⚽ Top Scorers",
  player_stats: "📊 Player Stats",
  team_facts: "🏟️ Team Facts",
  records: "🏆 Records",
  comparisons: "⚖️ Comparisons",
  true_false: "✅ True / False",
  multiple_choice: "🔢 Multiple Choice",
};

// ─── Difficulty Selection Screen ──────────────────────────────────────────────

function DifficultySelect({
  onSelect,
  isDark,
  neo4jConnected,
}: {
  onSelect: (d: Difficulty) => void;
  isDark: boolean;
  neo4jConnected: boolean;
}) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 px-4 py-12">
      <div
        className={cn(
          "mb-5 inline-flex h-20 w-20 items-center justify-center rounded-full",
          isDark ? "bg-violet-500/15" : "bg-purple-100"
        )}
      >
        <Target className={cn("h-10 w-10", isDark ? "text-violet-300" : "text-purple-600")} />
      </div>
      <h2 className={cn("text-2xl font-bold mb-2", isDark ? "text-slate-100" : "text-gray-900")}>
        FPL FantasyTrivia
      </h2>
      <p className={cn("text-sm mb-8 text-center max-w-sm", isDark ? "text-slate-400" : "text-gray-500")}>
        {QUESTIONS_PER_TEST} questions per test, randomly drawn from the question bank. Choose your difficulty:
      </p>

      {!neo4jConnected && (
        <div
          className={cn(
            "mb-6 rounded-lg border p-4 flex items-start gap-3 w-full max-w-sm",
            isDark ? "border-amber-400/20 bg-amber-500/10" : "border-yellow-200 bg-yellow-50"
          )}
        >
          <AlertCircle className="w-5 h-5 text-yellow-500 mt-0.5 shrink-0" />
          <p className={cn("text-sm", isDark ? "text-amber-200" : "text-yellow-800")}>
            Connecting to database… please wait.
          </p>
        </div>
      )}

      <div className="flex flex-col gap-3 w-full max-w-sm">
        {(["easy", "medium", "hard"] as Difficulty[]).map((d) => {
          const cfg = DIFFICULTY_CONFIG[d];
          return (
            <button
              key={d}
              onClick={() => onSelect(d)}
              disabled={!neo4jConnected}
              className={cn(
                "rounded-xl border-2 px-5 py-4 text-left transition-all disabled:opacity-50 disabled:cursor-not-allowed",
                isDark ? cfg.darkBg + " " + cfg.darkColor : cfg.bg + " " + cfg.color
              )}
            >
              <div className="flex items-center gap-2 font-bold text-base mb-0.5">
                <span>{cfg.emoji}</span>
                <span>{cfg.label}</span>
              </div>
              <p className={cn("text-xs", isDark ? "text-slate-400" : "text-gray-500")}>
                {cfg.description}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Quiz Screen ──────────────────────────────────────────────────────────────

function QuizScreen({
  difficulty,
  isDark,
  onFinish,
}: {
  difficulty: Difficulty;
  isDark: boolean;
  onFinish: (answers: QuizAnswer[], score: number) => void;
}) {
  const [questionNumber, setQuestionNumber] = useState(1);
  const [currentQuestion, setCurrentQuestion] = useState<TriviaQuestion | null>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [answerState, setAnswerState] = useState<AnswerState>("idle");
  const [feedback, setFeedback] = useState("");
  const [correctAnswer, setCorrectAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [answers, setAnswers] = useState<QuizAnswer[]>([]);
  const [score, setScore] = useState(0);
  const [answering, setAnswering] = useState(false);

  const cfg = DIFFICULTY_CONFIG[difficulty];
  const isMounted = useRef(true);
  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; };
  }, []);

  const fetchQuestion = useCallback(async () => {
    setLoading(true);
    setError("");
    setSelectedAnswer(null);
    setAnswerState("idle");
    setFeedback("");
    setCorrectAnswer("");
    try {
      const q = await apiService.getNewTriviaQuestion(difficulty);
      if (isMounted.current) setCurrentQuestion(q);
    } catch (e) {
      if (isMounted.current) setError(handleApiError(e));
    } finally {
      if (isMounted.current) setLoading(false);
    }
  }, [difficulty]);

  // Load first question on mount
  useEffect(() => { fetchQuestion(); }, [fetchQuestion]);

  const handleAnswer = async (option: string) => {
    if (answerState !== "idle" || !currentQuestion || answering) return;
    setSelectedAnswer(option);
    setAnswering(true);
    try {
      const result = await apiService.checkTriviaAnswer(currentQuestion.question_id, option);
      if (!isMounted.current) return;
      const isCorrect = result.correct;
      setAnswerState(isCorrect ? "correct" : "wrong");
      setFeedback(result.feedback || (isCorrect ? "Correct!" : "Wrong!"));
      setCorrectAnswer(result.correct_answer || "");
      const newAnswer: QuizAnswer = {
        questionText: currentQuestion.question,
        selected: option,
        correct: result.correct_answer || option,
        isCorrect,
      };
      const newAnswers = [...answers, newAnswer];
      const newScore = score + (isCorrect ? 1 : 0);
      setAnswers(newAnswers);
      setScore(newScore);

      // If last question, finish after short delay
      if (questionNumber >= QUESTIONS_PER_TEST) {
        setTimeout(() => { if (isMounted.current) onFinish(newAnswers, newScore); }, 1800);
      }
    } catch (e) {
      if (isMounted.current) setError(handleApiError(e));
    } finally {
      if (isMounted.current) setAnswering(false);
    }
  };

  const handleNext = () => {
    if (questionNumber >= QUESTIONS_PER_TEST) return;
    setQuestionNumber((n) => n + 1);
    fetchQuestion();
  };

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Quiz header */}
      <div
        className={cn(
          "px-4 py-3 flex items-center justify-between border-b",
          isDark ? "border-slate-800 bg-slate-950/80" : "bg-white border-gray-200"
        )}
      >
        <div className="flex items-center gap-2">
          <span className={cn("text-sm font-semibold", isDark ? cfg.darkColor : cfg.color)}>
            {cfg.emoji} {cfg.label}
          </span>
          <span className={cn("text-xs", isDark ? "text-slate-400" : "text-gray-500")}>
            · Question {questionNumber} / {QUESTIONS_PER_TEST}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className={cn("text-xs font-medium", isDark ? "text-emerald-300" : "text-green-600")}>
            {score} correct
          </span>
          {/* Progress bar */}
          <div className={cn("w-24 h-2 rounded-full overflow-hidden", isDark ? "bg-slate-800" : "bg-gray-200")}>
            <div
              className="h-full bg-purple-500 transition-all duration-300"
              style={{ width: `${((questionNumber - 1) / QUESTIONS_PER_TEST) * 100}%` }}
            />
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 max-w-2xl mx-auto w-full sm:px-6">
        {error && (
          <div
            className={cn(
              "rounded-lg border p-3 text-sm",
              isDark ? "border-red-400/20 bg-red-500/10 text-red-200" : "border-red-200 bg-red-50 text-red-700"
            )}
          >
            {error}
            <button onClick={fetchQuestion} className="ml-2 underline">Retry</button>
          </div>
        )}

        {loading && !currentQuestion && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className={cn("h-8 w-8 animate-spin", isDark ? "text-violet-400" : "text-purple-500")} />
          </div>
        )}

        <AnimatePresence mode="wait">
          {currentQuestion && !loading && (
            <motion.div
              key={currentQuestion.question_id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              {/* Category badge */}
              <div className="flex gap-2 flex-wrap">
                <span
                  className={cn(
                    "rounded-full px-2 py-1 text-xs font-medium",
                    isDark ? "bg-violet-500/15 text-violet-200" : "bg-purple-100 text-purple-700"
                  )}
                >
                  {CATEGORY_LABELS[currentQuestion.category] ?? currentQuestion.category}
                </span>
              </div>

              {/* Question card */}
              <div className="bg-gradient-to-br from-purple-600 to-indigo-700 text-white rounded-xl p-5">
                <p className="text-base font-semibold leading-snug">{currentQuestion.question}</p>
              </div>

              {/* Options */}
              <div className="grid grid-cols-1 gap-3">
                {currentQuestion.options.map((option) => {
                  let style = isDark
                    ? "border-slate-700 bg-slate-900/60 hover:border-violet-400/40 hover:bg-slate-900 cursor-pointer text-slate-100"
                    : "border-gray-200 bg-white hover:border-purple-400 hover:bg-purple-50 cursor-pointer";

                  if (answerState !== "idle") {
                    if (option === correctAnswer && answerState === "wrong") {
                      style = isDark
                        ? "border-emerald-500/40 bg-emerald-500/15 cursor-default text-slate-100"
                        : "border-green-400 bg-green-100 cursor-default";
                    } else if (option === selectedAnswer) {
                      style = answerState === "correct"
                        ? isDark
                          ? "border-emerald-500/40 bg-emerald-500/15 cursor-default text-slate-100"
                          : "border-green-400 bg-green-100 cursor-default"
                        : isDark
                        ? "border-red-500/40 bg-red-500/15 cursor-default text-slate-100"
                        : "border-red-400 bg-red-50 cursor-default";
                    } else {
                      style = isDark
                        ? "border-slate-700 bg-slate-900/40 opacity-50 cursor-default text-slate-100"
                        : "border-gray-200 bg-gray-50 opacity-50 cursor-default";
                    }
                  }

                  return (
                    <button
                      key={option}
                      onClick={() => handleAnswer(option)}
                      disabled={answerState !== "idle" || answering}
                      className={cn(
                        "flex w-full items-center justify-between gap-3 rounded-lg border-2 px-4 py-3 text-left text-sm font-medium transition-all",
                        style
                      )}
                    >
                      <span>{option}</span>
                      {answerState !== "idle" && option === selectedAnswer && answerState === "correct" && (
                        <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                      )}
                      {answerState !== "idle" && option === selectedAnswer && answerState === "wrong" && (
                        <XCircle className="w-5 h-5 text-red-500 shrink-0" />
                      )}
                      {answerState === "wrong" && option === correctAnswer && (
                        <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Feedback */}
              {answerState !== "idle" && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    "rounded-lg border p-3 text-sm",
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

              {/* Next button — only show if NOT the last question */}
              {answerState !== "idle" && questionNumber < QUESTIONS_PER_TEST && (
                <button
                  onClick={handleNext}
                  disabled={loading}
                  className="w-full py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Next Question →"}
                </button>
              )}

              {/* Last question — show finishing message */}
              {answerState !== "idle" && questionNumber >= QUESTIONS_PER_TEST && (
                <div className={cn("text-center text-sm py-2", isDark ? "text-slate-400" : "text-gray-500")}>
                  <Loader2 className="inline w-4 h-4 animate-spin mr-1" />
                  Loading your results…
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ─── Results Screen ───────────────────────────────────────────────────────────

function ResultsScreen({
  difficulty,
  score,
  answers,
  isDark,
  onFinish,
}: {
  difficulty: Difficulty;
  score: number;
  answers: QuizAnswer[];
  isDark: boolean;
  onFinish: () => void;
}) {
  const total = QUESTIONS_PER_TEST;
  const accuracy = Math.round((score / total) * 100);
  const cfg = DIFFICULTY_CONFIG[difficulty];

  const grade =
    accuracy >= 90 ? { label: "Outstanding!", emoji: "🏆", color: isDark ? "text-yellow-300" : "text-yellow-600" } :
    accuracy >= 70 ? { label: "Great job!", emoji: "🎯", color: isDark ? "text-emerald-300" : "text-green-600" } :
    accuracy >= 50 ? { label: "Not bad!", emoji: "👍", color: isDark ? "text-blue-300" : "text-blue-600" } :
    { label: "Keep practicing!", emoji: "📚", color: isDark ? "text-slate-300" : "text-gray-600" };

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 max-w-2xl mx-auto w-full sm:px-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="space-y-6"
      >
        {/* Score card */}
        <div
          className={cn(
            "rounded-2xl border p-6 text-center",
            isDark ? "border-slate-700 bg-slate-900/80" : "border-gray-200 bg-white"
          )}
        >
          <div className="text-5xl mb-3">{grade.emoji}</div>
          <h2 className={cn("text-2xl font-bold mb-1", grade.color)}>{grade.label}</h2>
          <p className={cn("text-sm mb-5", isDark ? "text-slate-400" : "text-gray-500")}>
            {cfg.emoji} {cfg.label} difficulty
          </p>
          <div className="flex justify-center gap-6">
            <div className="text-center">
              <div className={cn("text-3xl font-bold", isDark ? "text-slate-100" : "text-gray-900")}>
                {score} / {total}
              </div>
              <div className={cn("text-xs mt-0.5", isDark ? "text-slate-400" : "text-gray-500")}>Score</div>
            </div>
            <div className={cn("w-px self-stretch", isDark ? "bg-slate-700" : "bg-gray-200")} />
            <div className="text-center">
              <div className={cn("text-3xl font-bold", isDark ? "text-slate-100" : "text-gray-900")}>
                {accuracy}%
              </div>
              <div className={cn("text-xs mt-0.5", isDark ? "text-slate-400" : "text-gray-500")}>Accuracy</div>
            </div>
          </div>
        </div>

        {/* Question review */}
        <div>
          <h3 className={cn("font-semibold text-sm mb-3", isDark ? "text-slate-300" : "text-gray-700")}>
            Review
          </h3>
          <div className="space-y-2">
            {answers.map((a, i) => (
              <div
                key={i}
                className={cn(
                  "rounded-lg border p-3",
                  a.isCorrect
                    ? isDark ? "border-emerald-500/30 bg-emerald-500/10" : "border-green-200 bg-green-50"
                    : isDark ? "border-red-500/30 bg-red-500/10" : "border-red-100 bg-red-50"
                )}
              >
                <div className="flex items-start gap-2">
                  {a.isCorrect
                    ? <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                    : <XCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />}
                  <div className="min-w-0">
                    <p className={cn("text-xs font-medium leading-snug", isDark ? "text-slate-200" : "text-gray-800")}>
                      {i + 1}. {a.questionText}
                    </p>
                    {!a.isCorrect && (
                      <p className={cn("text-xs mt-1", isDark ? "text-slate-400" : "text-gray-500")}>
                        Your answer: <span className="text-red-400">{a.selected}</span>
                        {" · "}
                        Correct: <span className={isDark ? "text-emerald-300" : "text-green-700"}>{a.correct}</span>
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Finish button */}
        <button
          onClick={onFinish}
          className="w-full py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-bold text-base transition-colors"
        >
          Finish
        </button>
      </motion.div>
    </div>
  );
}

// ─── Main Trivia Page ─────────────────────────────────────────────────────────

export function Trivia() {
  const { neo4jConnected, theme } = useAppStore();
  const isDark = theme === "dark";

  const [view, setView] = useState<View>("select");
  const [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [quizAnswers, setQuizAnswers] = useState<QuizAnswer[]>([]);
  const [quizScore, setQuizScore] = useState(0);

  const handleSelectDifficulty = (d: Difficulty) => {
    setDifficulty(d);
    setQuizAnswers([]);
    setQuizScore(0);
    setView("quiz");
  };

  const handleQuizFinish = (answers: QuizAnswer[], score: number) => {
    setQuizAnswers(answers);
    setQuizScore(score);
    setView("results");
  };

  const handleReturnToSelect = () => {
    setView("select");
  };

  return (
    <div className="flex flex-col h-full">
      {/* Page header */}
      <div
        className={cn(
          "border-b px-4 py-4 sm:px-6 flex items-center justify-between shrink-0",
          isDark ? "border-slate-800 bg-slate-950/80" : "bg-white border-gray-200"
        )}
      >
        <div>
          <h2 className={cn("text-xl font-bold sm:text-2xl", isDark ? "text-slate-100" : "text-gray-900")}>
            🎯 FPL FantasyTrivia
          </h2>
          <p className={cn("text-sm", isDark ? "text-slate-400" : "text-gray-600")}>
            Test your Fantasy Premier League knowledge
          </p>
        </div>
        {view !== "select" && (
          <button
            onClick={handleReturnToSelect}
            className={cn(
              "rounded-lg border px-3 py-1.5 text-xs transition-colors",
              isDark ? "border-slate-700 text-slate-400 hover:text-slate-200" : "border-gray-300 text-gray-500 hover:text-gray-700"
            )}
          >
            ← Back
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
        <AnimatePresence mode="wait">
          {view === "select" && (
            <motion.div key="select" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col flex-1">
              <DifficultySelect onSelect={handleSelectDifficulty} isDark={isDark} neo4jConnected={neo4jConnected} />
            </motion.div>
          )}

          {view === "quiz" && (
            <motion.div key="quiz" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col flex-1 min-h-0">
              <QuizScreen difficulty={difficulty} isDark={isDark} onFinish={handleQuizFinish} />
            </motion.div>
          )}

          {view === "results" && (
            <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col flex-1 min-h-0">
              <ResultsScreen
                difficulty={difficulty}
                score={quizScore}
                answers={quizAnswers}
                isDark={isDark}
                onFinish={handleReturnToSelect}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
