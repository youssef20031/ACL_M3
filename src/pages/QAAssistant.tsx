import { useState, useRef, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { AlertTriangle, Loader2, MessageCircle, PanelRightClose, PanelRightOpen, Trash2 } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import type { QueryResponse } from '../services/api';
import { Button } from '../components/ui';
import { ChatMessage, ThinkingMessage } from '../components/assistant/ChatMessage';
import { Composer } from '../components/assistant/Composer';
import { QueryDetails } from '../components/assistant/QueryDetails';
import { cn } from '../utils/cn';

const EXAMPLES = [
  'Who scored the most goals in 2022-23?',
  'Which midfielder had the best points per game?',
  'Show me top defenders by clean sheets',
  'Who should I captain this gameweek?',
];

const DETAILS_ID = 'assistant-query-details';

export function QAAssistant() {
  const [input, setInput] = useState('');
  const [showDetails, setShowDetails] = useState(false);
  const [lastQueryInfo, setLastQueryInfo] = useState<QueryResponse | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    chatHistory,
    addMessage,
    clearChat,
    neo4jConnected,
    selectedModel,
    retrievalMethod,
    embeddingModel,
  } = useAppStore();

  // Shared health query (polled by the app shell): tells "still connecting" apart from "offline".
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiService.getHealth(),
    staleTime: 30_000,
  });

  const scrollToBottom = () => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    messagesEndRef.current?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'end' });
  };

  const queryMutation = useMutation({
    mutationFn: (question: string) => {
      // Convert chat history to API format (exclude the current question)
      const historyForAPI = chatHistory.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      return apiService.queryFPL(
        question,
        selectedModel,
        retrievalMethod,
        embeddingModel,
        chatHistory.length === 0, // is_first_message
        historyForAPI
      );
    },
    onSuccess: (data) => {
      addMessage({
        role: 'assistant',
        content: data.answer,
        timestamp: Date.now(),
      });
      setLastQueryInfo(data);
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      addMessage({
        role: 'assistant',
        content: `Error: ${errorMessage}`,
        timestamp: Date.now(),
      });
    },
  });

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, queryMutation.isPending]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !neo4jConnected || queryMutation.isPending) return;

    addMessage({
      role: 'user',
      content: input,
      timestamp: Date.now(),
    });

    queryMutation.mutate(input);
    setInput('');
  };

  const handleSendQuestion = (question: string) => {
    if (!neo4jConnected || queryMutation.isPending) return;

    addMessage({
      role: 'user',
      content: question,
      timestamp: Date.now(),
    });

    queryMutation.mutate(question);
  };

  // Other pages can open the assistant with a question ready to send: navigate('/assistant', { state: { question } }).
  const location = useLocation();
  const navigate = useNavigate();
  const pendingQuestion = (location.state as { question?: string } | null)?.question;
  const sentRef = useRef<string | null>(null);
  useEffect(() => {
    if (!pendingQuestion || !neo4jConnected || sentRef.current === pendingQuestion) return;
    sentRef.current = pendingQuestion;
    handleSendQuestion(pendingQuestion);
    navigate(location.pathname, { replace: true, state: null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingQuestion, neo4jConnected]);

  const connecting = !neo4jConnected && !health;
  const isEmpty = chatHistory.length === 0 && !queryMutation.isPending;

  return (
    <div className="flex min-h-0 w-full flex-1 flex-col">
      {/* Page header */}
      <div className="shrink-0 border-b border-line bg-bg">
        <div className="mx-auto flex w-full max-w-page items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="min-w-0">
            <h1 className="font-display text-h2 font-extrabold text-fg">Assistant</h1>
            <p className="hidden truncate text-data text-fg-subtle sm:block">
              Ask about players, teams and seasons. Answers come from the FPL knowledge graph.
            </p>
          </div>
          <div className="flex shrink-0 gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setShowDetails((v) => !v)}
              aria-expanded={showDetails}
              aria-controls={DETAILS_ID}
              icon={
                showDetails ? (
                  <PanelRightClose className="h-4 w-4" aria-hidden="true" />
                ) : (
                  <PanelRightOpen className="h-4 w-4" aria-hidden="true" />
                )
              }
            >
              {showDetails ? 'Hide details' : 'Show details'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => clearChat()}
              disabled={chatHistory.length === 0}
              icon={<Trash2 className="h-4 w-4" aria-hidden="true" />}
            >
              <span className="hidden sm:inline">Clear chat</span>
              <span className="sr-only sm:hidden">Clear chat</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Chat (messages + composer) and the details panel. Below lg the panel sits between
          the messages and the composer; from lg it is a column on the right. */}
      <div
        className={cn(
          'grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,1fr)_auto_auto]',
          showDetails && 'lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:grid-rows-[minmax(0,1fr)_auto] xl:grid-cols-[minmax(0,1fr)_minmax(0,30rem)]',
        )}
      >
        {/* Messages */}
        <div className="row-start-1 min-h-0 overflow-y-auto lg:col-start-1">
          <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-6 sm:px-6">
            {isEmpty ? (
              <div className="my-auto flex flex-col items-center py-8 text-center">
                <span className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-brand text-accent ring-1 ring-brand-2">
                  <MessageCircle className="h-6 w-6" aria-hidden="true" />
                </span>
                <h2 className="font-display text-h2 font-bold text-fg">What do you want to know?</h2>
                <p className="mt-2 max-w-md text-body text-fg-muted">
                  Stats, comparisons and predictions for every Premier League season since 2020-21, plus the current one.
                </p>
                <ul className="mt-6 flex max-w-xl flex-wrap justify-center gap-2" aria-label="Example questions">
                  {EXAMPLES.map((example) => (
                    <li key={example}>
                      <button
                        type="button"
                        onClick={() => handleSendQuestion(example)}
                        disabled={!neo4jConnected || queryMutation.isPending}
                        className={cn(
                          'rounded-full border border-line-strong bg-surface px-3.5 py-1.5 text-data text-fg-muted transition-colors',
                          'hover:border-accent hover:text-fg disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-line-strong disabled:hover:text-fg-muted',
                        )}
                      >
                        {example}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <ol className="mt-auto flex flex-col gap-5" role="log" aria-live="polite" aria-label="Conversation">
                {chatHistory.map((message, index) => (
                  <ChatMessage key={index} message={message} />
                ))}
                {queryMutation.isPending && <ThinkingMessage />}
              </ol>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Composer */}
        <div className="row-start-3 shrink-0 border-t border-line bg-bg lg:col-start-1 lg:row-start-2">
          <div className="mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
            {!neo4jConnected && (
              <div
                role="status"
                className={cn(
                  'mb-2 flex items-start gap-2.5 rounded-md border px-3 py-2 text-data',
                  connecting ? 'border-line bg-surface text-fg-muted' : 'border-warning/40 bg-warning/10 text-fg',
                )}
              >
                {connecting ? (
                  <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-accent-2" aria-hidden="true" />
                ) : (
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
                )}
                <p>
                  {connecting ? (
                    'Connecting to the knowledge graph…'
                  ) : (
                    <>
                      <span className="font-semibold">Knowledge graph offline.</span> Questions are paused until the
                      database is reachable again. This page checks automatically.
                    </>
                  )}
                </p>
              </div>
            )}
            <Composer
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              disabled={!neo4jConnected}
              busy={queryMutation.isPending}
              placeholder={neo4jConnected ? 'Ask a question about FPL…' : 'Waiting for the knowledge graph…'}
              footer={
                <>
                  <span className="hidden sm:inline">Enter to send, Shift+Enter for a new line</span>
                  <span>
                    {selectedModel} · {retrievalMethod} retrieval ·{' '}
                    <Link to="/settings" className="font-semibold text-accent hover:underline">
                      Change
                    </Link>
                  </span>
                </>
              }
            />
          </div>
        </div>

        {/* Details panel */}
        {showDetails && (
          <QueryDetails
            id={DETAILS_ID}
            info={lastQueryInfo}
            onClose={() => setShowDetails(false)}
            className={cn(
              'row-start-2 max-h-[45dvh] border-t border-line',
              'lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:max-h-none lg:border-l lg:border-t-0',
            )}
          />
        )}
      </div>
    </div>
  );
}
