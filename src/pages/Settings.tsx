import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Database,
  Sparkles,
  Bot,
  Download,
  CheckCircle,
  XCircle,
  Loader2,
  Eye,
  EyeOff,
  RefreshCw,
  Server,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import { cn } from '../utils/cn';

const MODELS = [
  { key: 'qwen-2.5-coder', label: 'Qwen 2.5 Coder', description: 'Good for structured data queries' },
  { key: 'llama-3.2-3b', label: 'Llama 3.2 3B', description: 'Fast, lightweight responses' },
  { key: 'qwen-2.5-7b', label: 'Qwen 2.5 7B', description: 'Higher quality, slower' },
];

const RETRIEVAL_METHODS = [
  { key: 'Baseline', label: 'Baseline (Cypher)', description: 'Direct graph queries only' },
  { key: 'Embeddings', label: 'Embeddings', description: 'Semantic similarity search' },
  { key: 'Hybrid', label: 'Hybrid', description: 'Cypher + embeddings combined (recommended)' },
] as const;

const EMBEDDING_MODELS = [
  { key: 'minilm', label: 'MiniLM (Fast)', description: '384-dim, quick to build' },
  { key: 'mpnet', label: 'MPNet (Quality)', description: '768-dim, higher accuracy' },
] as const;

export function Settings() {
  const {
    neo4jConnected,
    neo4jStats,
    setNeo4jConnected,
    setNeo4jStats,
    embeddingsBuilt,
    embeddingCount,
    setEmbeddingsBuilt,
    setEmbeddingCount,
    selectedModel,
    retrievalMethod,
    embeddingModel,
    setSelectedModel,
    setRetrievalMethod,
    setEmbeddingModel,
    theme,
  } = useAppStore();
  const isDark = theme === 'dark';
  const pageText = isDark ? 'text-slate-100' : 'text-gray-900';
  const mutedText = isDark ? 'text-slate-400' : 'text-gray-600';
  const panelClass = isDark ? 'border-slate-800 bg-slate-900/80 text-slate-100' : 'border-gray-200 bg-white text-gray-900';
  const fieldClass = isDark ? 'border-slate-700 bg-slate-950/70 text-slate-100 placeholder:text-slate-500 focus:ring-violet-500' : 'border-gray-300 bg-white text-gray-900 placeholder:text-gray-400 focus:ring-purple-500';
  const optionSelectedClass = isDark ? 'border-violet-400/60 bg-violet-500/10' : 'border-purple-500 bg-purple-50';
  const optionIdleClass = isDark ? 'border-slate-700 bg-slate-950/60 hover:border-slate-500' : 'border-gray-200 hover:border-gray-300';

  const [uri, setUri] = useState('');
  const [username, setUsername] = useState('neo4j');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [connectionError, setConnectionError] = useState('');

  const { data: health, refetch: refetchHealth } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 30000,
  });

  const connectMutation = useMutation({
    mutationFn: () => apiService.connectNeo4j(uri, username, password),
    onSuccess: (data) => {
      if (data.success) {
        setNeo4jConnected(true);
        setNeo4jStats(data.stats ?? null);
        setConnectionError('');
      } else {
        setConnectionError(data.message);
      }
    },
    onError: (error) => {
      setConnectionError(handleApiError(error));
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: () => apiService.disconnectNeo4j(),
    onSuccess: () => {
      setNeo4jConnected(false);
      setNeo4jStats(null);
    },
  });

  const buildEmbeddingsMutation = useMutation({
    mutationFn: () => apiService.buildEmbeddings(embeddingModel),
    onSuccess: (data) => {
      void refetchHealth();
      if (data.success && !data.started && !data.building && data.count > 0) {
        setEmbeddingsBuilt(true);
        setEmbeddingCount(data.count);
      } else if (data.started) {
        setEmbeddingsBuilt(false);
      }
    },
  });

  const loadDataMutation = useMutation({
    mutationFn: () => apiService.loadFPLData(true),
  });

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className={cn('text-2xl font-bold', pageText)}>⚙️ Settings</h2>
        <p className={cn('text-sm mt-1', mutedText)}>Configure database, models, and retrieval options</p>
      </div>

      {/* API Status */}
      <div className={cn('rounded-xl border p-5 space-y-3', panelClass)}>
        <div className="flex items-center justify-between">
          <h3 className={cn('font-semibold flex items-center gap-2', pageText)}>
            <Server className="w-4 h-4" /> API Status
          </h3>
          <button
            onClick={() => refetchHealth()}
            className={cn('transition-colors', isDark ? 'text-slate-400 hover:text-slate-200' : 'text-gray-400 hover:text-gray-600')}
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        {health ? (
          <>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <StatusRow label="API" value={health.status === 'healthy' ? 'Online' : 'Error'} ok={health.status === 'healthy'} theme={theme} />
              <StatusRow label="Neo4j" value={health.neo4j} ok={health.neo4j === 'connected'} theme={theme} />
              <StatusRow label="LLM" value={health.llm_available ? 'Available' : 'Not configured'} ok={health.llm_available} theme={theme} />
              <StatusRow
                label="Embeddings"
                value={health.embeddings_building ? 'Building...' : health.embeddings_built ? `${health.embedding_count.toLocaleString()} vectors` : 'Not built'}
                ok={health.embeddings_built && !health.embeddings_building}
                theme={theme}
              />
            </div>
            {health.embedding_build_error && (
              <p className={cn('text-xs', isDark ? 'text-red-300' : 'text-red-600')}>
                Last build error: {health.embedding_build_error}
              </p>
            )}
          </>
        ) : (
          <p className={cn('text-sm', mutedText)}>Connecting to API...</p>
        )}
      </div>

      {/* Neo4j Connection */}
      <div className={cn('rounded-xl border p-5 space-y-4', panelClass)}>
        <h3 className={cn('font-semibold flex items-center gap-2', pageText)}>
          <Database className="w-4 h-4" /> Neo4j Connection
        </h3>

        <div className="space-y-3">
          <div>
            <label className={cn('block text-sm font-medium mb-1', isDark ? 'text-slate-300' : 'text-gray-700')}>URI</label>
            <input
              type="text"
              value={uri}
              onChange={(e) => setUri(e.target.value)}
              placeholder="neo4j+s://xxxxxxxx.databases.neo4j.io"
              className={cn('w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2', fieldClass)}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={cn('block text-sm font-medium mb-1', isDark ? 'text-slate-300' : 'text-gray-700')}>Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className={cn('w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2', fieldClass)}
              />
            </div>
            <div>
              <label className={cn('block text-sm font-medium mb-1', isDark ? 'text-slate-300' : 'text-gray-700')}>Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={cn('w-full rounded-lg border px-3 py-2 pr-9 text-sm focus:outline-none focus:ring-2', fieldClass)}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className={cn('absolute right-2 top-1/2 -translate-y-1/2 transition-colors', isDark ? 'text-slate-500 hover:text-slate-200' : 'text-gray-400 hover:text-gray-600')}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {connectionError && (
          <p className={cn('text-sm px-3 py-2 rounded-lg', isDark ? 'bg-red-500/10 text-red-200' : 'bg-red-50 text-red-600')}>{connectionError}</p>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          {neo4jConnected ? (
            <>
              <div className={cn('flex items-center gap-2 text-sm font-medium', isDark ? 'text-emerald-300' : 'text-green-600')}>
                <CheckCircle className="w-4 h-4" />
                Connected
                {neo4jStats && (
                  <span className={cn('font-normal', isDark ? 'text-slate-400' : 'text-gray-500')}>
                    — {neo4jStats.total_nodes.toLocaleString()} nodes, {neo4jStats.total_relationships.toLocaleString()} relationships
                  </span>
                )}
              </div>
              <button
                onClick={() => disconnectMutation.mutate()}
                disabled={disconnectMutation.isPending}
                className={cn('ml-auto rounded-lg border px-4 py-2 text-sm transition-colors', isDark ? 'border-red-500/30 text-red-200 hover:bg-red-500/10' : 'border-red-200 text-red-600 hover:bg-red-50')}
              >
                Disconnect
              </button>
            </>
          ) : (
            <button
              onClick={() => connectMutation.mutate()}
              disabled={connectMutation.isPending}
              className="px-5 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {connectMutation.isPending && <Loader2 className="w-4 h-4 animate-spin" />}
              Connect
            </button>
          )}
        </div>
      </div>

      {/* Data Management */}
      <div className={cn('rounded-xl border p-5 space-y-4', panelClass)}>
        <h3 className={cn('font-semibold flex items-center gap-2', pageText)}>
          <Download className="w-4 h-4" /> Data Management
        </h3>
        <p className={mutedText}>
          Load the FPL CSV data into Neo4j. This clears existing data and re-imports everything.
        </p>
        <div className="flex items-center gap-4 flex-wrap">
          <button
            onClick={() => loadDataMutation.mutate()}
            disabled={!neo4jConnected || loadDataMutation.isPending}
            className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loadDataMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Loading...
              </>
            ) : (
              '📥 Load FPL Data'
            )}
          </button>
          {loadDataMutation.isSuccess && (
            <span className={cn('text-sm flex items-center gap-1', isDark ? 'text-emerald-300' : 'text-green-600')}>
              <CheckCircle className="w-4 h-4" />
              Loaded {(loadDataMutation.data as any)?.stats?.total_nodes?.toLocaleString()} nodes
            </span>
          )}
          {loadDataMutation.isError && (
            <span className={cn('text-sm', isDark ? 'text-red-300' : 'text-red-600')}>{handleApiError(loadDataMutation.error)}</span>
          )}
        </div>
      </div>

      {/* LLM Model */}
      <div className={cn('rounded-xl border p-5 space-y-4', panelClass)}>
        <h3 className={cn('font-semibold flex items-center gap-2', pageText)}>
          <Bot className="w-4 h-4" /> LLM Model
        </h3>
        <div className="grid grid-cols-1 gap-2">
          {MODELS.map((m) => (
            <label
              key={m.key}
              className={cn('flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors', selectedModel === m.key ? optionSelectedClass : optionIdleClass)}
            >
              <input
                type="radio"
                name="model"
                value={m.key}
                checked={selectedModel === m.key}
                onChange={() => setSelectedModel(m.key)}
                className="mt-0.5 accent-purple-600"
              />
              <div>
                <p className={cn('text-sm font-medium', pageText)}>{m.label}</p>
                <p className={cn('text-xs', mutedText)}>{m.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Retrieval Method */}
      <div className={cn('rounded-xl border p-5 space-y-4', panelClass)}>
        <h3 className={cn('font-semibold', pageText)}>🔍 Retrieval Method</h3>
        <div className="grid grid-cols-1 gap-2">
          {RETRIEVAL_METHODS.map((r) => (
            <label
              key={r.key}
              className={cn('flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors', retrievalMethod === r.key ? optionSelectedClass : optionIdleClass)}
            >
              <input
                type="radio"
                name="retrieval"
                value={r.key}
                checked={retrievalMethod === r.key}
                onChange={() => setRetrievalMethod(r.key)}
                className="mt-0.5 accent-purple-600"
              />
              <div>
                <p className={cn('text-sm font-medium', pageText)}>{r.label}</p>
                <p className={cn('text-xs', mutedText)}>{r.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Embeddings */}
      <div className={cn('rounded-xl border p-5 space-y-4', panelClass)}>
        <h3 className={cn('font-semibold flex items-center gap-2', pageText)}>
          <Sparkles className="w-4 h-4" /> Embeddings
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {EMBEDDING_MODELS.map((m) => (
            <label
              key={m.key}
              className={cn('flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors', embeddingModel === m.key ? optionSelectedClass : optionIdleClass)}
            >
              <input
                type="radio"
                name="embedding"
                value={m.key}
                checked={embeddingModel === m.key}
                onChange={() => setEmbeddingModel(m.key)}
                className="mt-0.5 accent-purple-600"
              />
              <div>
                <p className={cn('text-sm font-medium', pageText)}>{m.label}</p>
                <p className={cn('text-xs', mutedText)}>{m.description}</p>
              </div>
            </label>
          ))}
        </div>
        <div className="space-y-3">
          <div className="flex items-center gap-4 flex-wrap">
            <button
              onClick={() => buildEmbeddingsMutation.mutate()}
              disabled={!neo4jConnected || buildEmbeddingsMutation.isPending || !!health?.embeddings_building}
              className="px-5 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {buildEmbeddingsMutation.isPending || health?.embeddings_building ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Building...
                </>
              ) : (
                '🔮 Build Embeddings'
              )}
            </button>
            {embeddingsBuilt && (
              <span className={cn('text-sm flex items-center gap-1', isDark ? 'text-emerald-300' : 'text-green-600')}>
                <CheckCircle className="w-4 h-4" />
                {embeddingCount.toLocaleString()} vectors ready
              </span>
            )}
            {buildEmbeddingsMutation.isError && (
              <span className={cn('text-sm', isDark ? 'text-red-300' : 'text-red-600')}>{handleApiError(buildEmbeddingsMutation.error)}</span>
            )}
          </div>

          {/* Clear, prominent in-progress message when building embeddings */}
          {(buildEmbeddingsMutation.isPending || health?.embeddings_building) && (
            <div className={cn('mt-3 flex items-start gap-3 rounded-lg border p-3 text-sm', isDark ? 'border-violet-400/20 bg-violet-500/10 text-violet-100' : 'border-purple-200 bg-purple-50 text-purple-900')}>
              <Loader2 className="mt-0.5 h-5 w-5 animate-spin" />
              <div>
                <div className="font-medium">Building embeddings in the background</div>
                <div className={cn('mt-1 text-xs', isDark ? 'text-violet-200' : 'text-purple-800')}>The request returns immediately now, so Railway does not drop the connection. Refresh in a minute or two to see the updated status.</div>
              </div>
            </div>
          )}

          {buildEmbeddingsMutation.isSuccess && buildEmbeddingsMutation.data && (
            <p className={mutedText}>{buildEmbeddingsMutation.data.message}</p>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusRow({ label, value, ok, theme }: { label: string; value: string; ok: boolean; theme: 'light' | 'dark' }) {
  const isDark = theme === 'dark';

  return (
    <div className={cn('flex items-center justify-between rounded-lg px-3 py-2', isDark ? 'bg-slate-950/70' : 'bg-gray-50')}>
      <span className={isDark ? 'text-slate-400' : 'text-gray-600'}>{label}</span>
      <span className={cn('flex items-center gap-1.5 font-medium', ok ? (isDark ? 'text-emerald-300' : 'text-green-600') : isDark ? 'text-red-300' : 'text-red-500')}>
        {ok ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
        {value}
      </span>
    </div>
  );
}
