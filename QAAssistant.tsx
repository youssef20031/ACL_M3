import { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Send, Loader2, AlertCircle, Info } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError, QueryResponse } from '../services/api';
import ReactMarkdown from 'react-markdown';
import { GraphVisualization } from '../components/GraphVisualization';
import { motion, AnimatePresence } from 'framer-motion';

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
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">💬 FPL Q&A Assistant</h2>
          <p className="text-sm text-gray-600">
            Ask questions about Fantasy Premier League players and statistics
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
          <button
            onClick={() => clearChat()}
            className="px-4 py-2 text-sm bg-red-50 hover:bg-red-100 text-red-600 rounded-lg transition-colors"
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {!neo4jConnected && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <p className="font-medium text-yellow-900">Neo4j Not Connected</p>
              <p className="text-sm text-yellow-700 mt-1">
                Please connect to Neo4j in the Settings page to start asking questions.
              </p>
            </div>
          </div>
        )}

        {chatHistory.length === 0 && neo4jConnected && (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 rounded-full mb-4">
              <MessageCircle className="w-8 h-8 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Start a Conversation
            </h3>
            <p className="text-gray-600 max-w-md mx-auto">
              Ask me anything about FPL players, teams, and statistics. For example:
            </p>
            <div className="mt-6 space-y-2 max-w-lg mx-auto">
              {[
                'Who scored the most goals in 2022-23?',
                'Which midfielder had the best points per game?',
                'Show me top defenders by clean sheets',
              ].map((example, i) => (
                <button
                  key={i}
                  onClick={() => setInput(example)}
                  className="block w-full text-left px-4 py-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors text-sm text-gray-700"
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
                className={`max-w-3xl rounded-lg px-4 py-3 ${
                  message.role === 'user'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white border border-gray-200 text-gray-900'
                }`}
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
            <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-purple-600" />
              <span className="text-sm text-gray-600">Analyzing your question...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Query Details Panel */}
      {showDetails && lastQueryInfo && (
        <div className="border-t bg-gray-50 p-6 max-h-96 overflow-y-auto">
          <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Info className="w-4 h-4" />
            Query Details
          </h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">Intent</h4>
              <span className="inline-block px-2 py-1 bg-purple-100 text-purple-700 rounded text-sm">
                {lastQueryInfo.intent}
              </span>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">Entities</h4>
              <pre className="bg-white border rounded p-2 text-xs overflow-x-auto">
                {JSON.stringify(lastQueryInfo.entities, null, 2)}
              </pre>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">Cypher Query</h4>
              <pre className="bg-gray-900 text-green-400 rounded p-3 text-xs overflow-x-auto">
                {lastQueryInfo.cypher_query}
              </pre>
            </div>

            {lastQueryInfo.embedding_used && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-1">
                  🔮 Embedding Search Active
                </h4>
                <p className="text-sm text-gray-600">
                  Similar players found using semantic search
                </p>
              </div>
            )}

            {lastQueryInfo.graph_data && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Knowledge Graph Visualization
                </h4>
                <GraphVisualization data={lastQueryInfo.graph_data} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Input Form */}
      <div className="bg-white border-t px-6 py-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
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
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!neo4jConnected || queryMutation.isPending || !input.trim()}
            className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {queryMutation.isPending ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
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