import { useState, useRef, useEffect, type CSSProperties } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Send, Loader2, AlertCircle, Info } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import type { QueryResponse } from '../services/api';
import ReactMarkdown from 'react-markdown';
import { GraphVisualization } from '../components/GraphVisualization';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../utils/cn';
import fplLogo from '../images/FPL_Logo.png';

const chatWallpaperStyle: CSSProperties = {
  backgroundImage: [
    'radial-gradient(circle at 50% 36%, rgba(168, 85, 247, 0.34) 0%, rgba(88, 28, 135, 0.18) 24%, rgba(15, 23, 42, 0) 63%)',
    'linear-gradient(180deg, rgba(5, 8, 20, 0.98) 0%, rgba(23, 8, 40, 0.94) 48%, rgba(72, 12, 92, 0.95) 100%)',
  ].join(', '),
  backgroundPosition: 'center center, center center',
  backgroundRepeat: 'no-repeat, no-repeat',
  backgroundSize: 'cover, cover',
  backgroundAttachment: 'scroll',
};

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
    theme,
  } = useAppStore();
  const isDark = theme === 'dark';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const queryMutation = useMutation({
    mutationFn: (question: string) =>
      apiService.queryFPL(question, selectedModel, retrievalMethod, embeddingModel),
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !neo4jConnected) return;

    addMessage({
      role: 'user',
      content: input,
      timestamp: Date.now(),
    });

    queryMutation.mutate(input);
    setInput('');
  };

  return (
    <div className="relative isolate flex flex-1 w-full min-h-0 flex-col overflow-hidden">
      {isDark && (
        <>
          <div aria-hidden="true" className="absolute inset-0 md:hidden" style={chatWallpaperStyle} />
          <div aria-hidden="true" className="absolute inset-0 flex md:hidden">
            <img
              src={fplLogo}
              alt=""
              className="w-full h-full object-cover opacity-15"
            />
          </div>
          <div aria-hidden="true" className="absolute inset-0 md:hidden bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.04),transparent_36%),radial-gradient(circle_at_bottom,rgba(168,85,247,0.18),transparent_42%)]" />
        </>
      )}

      {/* Header */}
      <div className={cn('relative z-10 flex flex-col gap-3 border-b px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6', isDark ? 'border-white/10 bg-slate-950/70 backdrop-blur-xl' : 'border-white/30 bg-white/80 backdrop-blur-xl')}>
        <div className="min-w-0">
          <h2 className={cn('text-xl font-bold sm:text-2xl', isDark ? 'text-slate-100' : 'text-gray-900')}>💬 FPL Q&A Assistant</h2>
          <p className={cn('text-sm', isDark ? 'text-slate-400' : 'text-gray-600')}>
            Ask questions about Fantasy Premier League players and statistics
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className={cn('rounded-lg px-4 py-2 text-sm transition-colors', isDark ? 'bg-white/5 text-slate-100 hover:bg-white/10' : 'bg-gray-100 text-gray-800 hover:bg-gray-200')}
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
          <button
            onClick={() => clearChat()}
            className={cn('rounded-lg px-4 py-2 text-sm transition-colors', isDark ? 'bg-red-500/10 text-red-200 hover:bg-red-500/20' : 'bg-red-50 text-red-600 hover:bg-red-100')}
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="relative z-10 flex-1 flex flex-col overflow-y-auto px-4 py-4 sm:px-6">
        <div className="flex-1" />
        <div className="flex flex-col space-y-4 mt-auto">
        {!neo4jConnected && (
          <div className={cn('flex items-start gap-3 rounded-lg border p-4 backdrop-blur-md', isDark ? 'border-amber-400/20 bg-amber-500/10' : 'border-yellow-200/90 bg-yellow-50/90')}>
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <p className={cn('font-medium', isDark ? 'text-amber-100' : 'text-yellow-900')}>Neo4j Not Connected</p>
              <p className={cn('mt-1 text-sm', isDark ? 'text-amber-200' : 'text-yellow-700')}>Connecting to Neo4j... Please wait.</p>
            </div>
          </div>
        )}

        {chatHistory.length === 0 && neo4jConnected && (
          <div className="text-center py-12">
            <div className={cn('mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full backdrop-blur-md', isDark ? 'bg-violet-500/15' : 'bg-purple-100/80')}>
              <MessageCircle className={cn('h-8 w-8', isDark ? 'text-violet-200' : 'text-purple-600')} />
            </div>
            <h3 className={cn('mb-2 text-lg font-semibold', isDark ? 'text-slate-100' : 'text-gray-900')}>
              Start a Conversation
            </h3>
            <p className={cn('mx-auto max-w-md', isDark ? 'text-slate-400' : 'text-gray-600')}>
              Ask me anything about FPL players, teams, and statistics. For example:
            </p>
            <div className="mx-auto mt-6 max-w-lg space-y-2">
              {[
                'Who scored the most goals in 2022-23?',
                'Which midfielder had the best points per game?',
                'Show me top defenders by clean sheets',
              ].map((example, i) => (
                <button
                  key={i}
                  onClick={() => setInput(example)}
                  className={cn('block w-full rounded-lg border px-4 py-3 text-left text-sm transition-colors backdrop-blur-md', isDark ? 'border-white/10 bg-slate-900/75 text-slate-100 hover:border-violet-400/40 hover:bg-slate-900/90' : 'border-white/50 bg-white/75 text-gray-700 hover:bg-white/90')}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence>
          {chatHistory.map((message, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={cn(
                  'max-w-3xl rounded-lg px-4 py-3 backdrop-blur-md',
                  message.role === 'user'
                    ? 'bg-purple-600 text-white'
                    : isDark
                      ? 'border border-white/10 bg-slate-900/78 text-slate-100'
                      : 'border border-white/70 bg-white/85 text-gray-900'
                )}
              >
                <ReactMarkdown className="prose prose-sm max-w-none">
                  {message.content}
                </ReactMarkdown>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {queryMutation.isPending && (
          <div className="flex justify-start">
            <div className={cn('flex items-center gap-2 rounded-lg border px-4 py-3 backdrop-blur-md', isDark ? 'border-white/10 bg-slate-900/78' : 'border-white/70 bg-white/85')}>
              <Loader2 className="w-4 h-4 animate-spin text-purple-600" />
              <span className={cn('text-sm', isDark ? 'text-slate-300' : 'text-gray-600')}>Analyzing your question...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Query Details Panel */}
      {showDetails && lastQueryInfo && (
        <div className={cn('relative z-10 max-h-96 overflow-y-auto border-t p-6 backdrop-blur-xl', isDark ? 'border-white/10 bg-slate-950/70' : 'border-white/30 bg-white/80')}>
          <h3 className={cn('mb-3 flex items-center gap-2 font-semibold', isDark ? 'text-slate-100' : 'text-gray-900')}>
            <Info className="w-4 h-4" />
            Query Details
          </h3>
          
          <div className="space-y-4">
            <div>
              <h4 className={cn('mb-1 text-sm font-medium', isDark ? 'text-slate-300' : 'text-gray-700')}>Intent</h4>
              <span className={cn('inline-block rounded px-2 py-1 text-sm', isDark ? 'bg-violet-500/15 text-violet-200' : 'bg-purple-100 text-purple-700')}>
                {lastQueryInfo.intent}
              </span>
            </div>

            <div>
              <h4 className={cn('mb-1 text-sm font-medium', isDark ? 'text-slate-300' : 'text-gray-700')}>Entities</h4>
              <pre className={cn('overflow-x-auto rounded p-2 text-xs', isDark ? 'border border-slate-800 bg-slate-900/80 text-slate-200' : 'border bg-white')}>
                {JSON.stringify(lastQueryInfo.entities, null, 2)}
              </pre>
            </div>

            <div>
              <h4 className={cn('mb-1 text-sm font-medium', isDark ? 'text-slate-300' : 'text-gray-700')}>Cypher Query</h4>
              <pre className={cn('overflow-x-auto rounded p-3 text-xs', isDark ? 'bg-slate-950 text-emerald-300' : 'bg-gray-900 text-green-400')}>
                {lastQueryInfo.cypher_query}
              </pre>
            </div>

            {lastQueryInfo.embedding_used && (
              <div>
                <h4 className={cn('mb-1 text-sm font-medium', isDark ? 'text-slate-300' : 'text-gray-700')}>
                  🔮 Embedding Search Active
                </h4>
                <p className={cn('text-sm', isDark ? 'text-slate-400' : 'text-gray-600')}>
                  Similar players found using semantic search
                </p>
              </div>
            )}

            {lastQueryInfo.graph_data && (
              <div>
                <h4 className={cn('mb-2 text-sm font-medium', isDark ? 'text-slate-300' : 'text-gray-700')}>
                  Knowledge Graph Visualization
                </h4>
                <GraphVisualization data={lastQueryInfo.graph_data} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className={cn('relative z-10 border-t px-4 py-4 sm:px-6 backdrop-blur-xl', isDark ? 'border-white/10 bg-slate-950/70' : 'border-white/30 bg-white/80')}>
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              neo4jConnected
                ? "Ask about FPL... (e.g., 'Who scored the most goals in 2022-23?')"
                : 'Connect to Neo4j first...'
            }
            disabled={!neo4jConnected || queryMutation.isPending}
            className={cn('flex-1 rounded-lg border px-4 py-3 text-base focus:outline-none focus:ring-2', isDark ? 'border-slate-700 bg-slate-900 text-slate-100 placeholder:text-slate-500 focus:ring-violet-500 disabled:bg-slate-900 disabled:text-slate-500' : 'border-gray-300 bg-white text-gray-900 placeholder:text-gray-400 focus:ring-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed')}
          />
          <button
            type="submit"
            disabled={!neo4jConnected || queryMutation.isPending || !input.trim()}
            className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 sm:w-auto"
          >
            <Send className="w-5 h-5" />
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

function MessageCircle({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}