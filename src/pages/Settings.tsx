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

const MODELS = [
  { key: 'qwen-2.5-coder', label: 'Qwen 2.5 Coder', description: 'Good for structured data queries' },
  { key: 'llama-3.2-3b', label: 'Llama 3.2 3B', description: 'Fast, lightweight responses' },
  { key: 'phi-3-mini', label: 'Phi-3 Mini', description: 'Microsoft compact model' },
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
  } = useAppStore();

  const [uri, setUri] = useState('bolt://localhost:7687');
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
      if (data.success) {
        setEmbeddingsBuilt(true);
        setEmbeddingCount(data.count);
      }
    },
  });

  const loadDataMutation = useMutation({
    mutationFn: () => apiService.loadFPLData(true),
  });

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">⚙️ Settings</h2>
        <p className="text-sm text-gray-600 mt-1">Configure database, models, and retrieval options</p>
      </div>

      {/* API Status */}
      <div className="bg-white rounded-xl border p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <Server className="w-4 h-4" /> API Status
          </h3>
          <button
            onClick={() => refetchHealth()}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        {health ? (
          <div className="grid grid-cols-2 gap-3 text-sm">
            <StatusRow label="API" value={health.status === 'healthy' ? 'Online' : 'Error'} ok={health.status === 'healthy'} />
            <StatusRow label="Neo4j" value={health.neo4j} ok={health.neo4j === 'connected'} />
            <StatusRow label="LLM" value={health.llm_available ? 'Available' : 'Not configured'} ok={health.llm_available} />
            <StatusRow
              label="Embeddings"
              value={health.embeddings_built ? `${health.embedding_count.toLocaleString()} vectors` : 'Not built'}
              ok={health.embeddings_built}
            />
          </div>
        ) : (
          <p className="text-sm text-gray-500">Connecting to API...</p>
        )}
      </div>

      {/* Neo4j Connection */}
      <div className="bg-white rounded-xl border p-5 space-y-4">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <Database className="w-4 h-4" /> Neo4j Connection
        </h3>

        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">URI</label>
            <input
              type="text"
              value={uri}
              onChange={(e) => setUri(e.target.value)}
              placeholder="bolt://localhost:7687"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 pr-9 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {connectionError && (
          <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{connectionError}</p>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          {neo4jConnected ? (
            <>
              <div className="flex items-center gap-2 text-green-600 text-sm font-medium">
                <CheckCircle className="w-4 h-4" />
                Connected
                {neo4jStats && (
                  <span className="text-gray-500 font-normal">
                    — {neo4jStats.total_nodes.toLocaleString()} nodes, {neo4jStats.total_relationships.toLocaleString()} relationships
                  </span>
                )}
              </div>
              <button
                onClick={() => disconnectMutation.mutate()}
                disabled={disconnectMutation.isPending}
                className="ml-auto px-4 py-2 text-sm border border-red-200 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
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
      <div className="bg-white rounded-xl border p-5 space-y-4">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <Download className="w-4 h-4" /> Data Management
        </h3>
        <p className="text-sm text-gray-600">
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
            <span className="text-sm text-green-600 flex items-center gap-1">
              <CheckCircle className="w-4 h-4" />
              Loaded {(loadDataMutation.data as any)?.stats?.total_nodes?.toLocaleString()} nodes
            </span>
          )}
          {loadDataMutation.isError && (
            <span className="text-sm text-red-600">{handleApiError(loadDataMutation.error)}</span>
          )}
        </div>
      </div>

      {/* LLM Model */}
      <div className="bg-white rounded-xl border p-5 space-y-4">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <Bot className="w-4 h-4" /> LLM Model
        </h3>
        <div className="grid grid-cols-1 gap-2">
          {MODELS.map((m) => (
            <label
              key={m.key}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                selectedModel === m.key
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
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
                <p className="text-sm font-medium text-gray-900">{m.label}</p>
                <p className="text-xs text-gray-500">{m.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Retrieval Method */}
      <div className="bg-white rounded-xl border p-5 space-y-4">
        <h3 className="font-semibold text-gray-900">🔍 Retrieval Method</h3>
        <div className="grid grid-cols-1 gap-2">
          {RETRIEVAL_METHODS.map((r) => (
            <label
              key={r.key}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                retrievalMethod === r.key
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
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
                <p className="text-sm font-medium text-gray-900">{r.label}</p>
                <p className="text-xs text-gray-500">{r.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Embeddings */}
      <div className="bg-white rounded-xl border p-5 space-y-4">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <Sparkles className="w-4 h-4" /> Embeddings
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {EMBEDDING_MODELS.map((m) => (
            <label
              key={m.key}
              className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                embeddingModel === m.key
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
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
                <p className="text-sm font-medium text-gray-900">{m.label}</p>
                <p className="text-xs text-gray-500">{m.description}</p>
              </div>
            </label>
          ))}
        </div>
        <div className="flex items-center gap-4 flex-wrap">
          <button
            onClick={() => buildEmbeddingsMutation.mutate()}
            disabled={!neo4jConnected || buildEmbeddingsMutation.isPending}
            className="px-5 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {buildEmbeddingsMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Building...
              </>
            ) : (
              '🔮 Build Embeddings'
            )}
          </button>
          {embeddingsBuilt && (
            <span className="text-sm text-green-600 flex items-center gap-1">
              <CheckCircle className="w-4 h-4" />
              {embeddingCount.toLocaleString()} vectors ready
            </span>
          )}
          {buildEmbeddingsMutation.isError && (
            <span className="text-sm text-red-600">{handleApiError(buildEmbeddingsMutation.error)}</span>
          )}
        </div>
        {buildEmbeddingsMutation.isSuccess && buildEmbeddingsMutation.data && (
          <p className="text-sm text-gray-600">{buildEmbeddingsMutation.data.message}</p>
        )}
      </div>
    </div>
  );
}

function StatusRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-2">
      <span className="text-gray-600">{label}</span>
      <span className={`flex items-center gap-1.5 font-medium ${ok ? 'text-green-600' : 'text-red-500'}`}>
        {ok ? <CheckCircle className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
        {value}
      </span>
    </div>
  );
}
