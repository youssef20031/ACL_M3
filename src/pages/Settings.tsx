import { useQuery } from '@tanstack/react-query';
import {
  Sparkles,
  Bot,
  CheckCircle,
  XCircle,
  RefreshCw,
  Server,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService } from '../services/api';
import { cn } from '../utils/cn';

const MODELS = [
  { key: 'llama-3.3-70b', label: 'Llama 3.3 70B Versatile', description: 'Highest performance model via Groq (recommended)' },
  { key: 'qwen-2.5-coder', label: 'Qwen 2.5 Coder', description: 'Light model, Good for structured data queries' },
  { key: 'llama-3.2-3b', label: 'Llama 3.2 3B', description: 'Fast, lightweight responses' },
  { key: 'qwen-2.5-7b', label: 'Qwen 2.5 7B', description: 'medium quality model' },
];

const RETRIEVAL_METHODS = [
  { key: 'Baseline', label: 'Baseline (Cypher)', description: 'Direct graph queries only' },
  { key: 'Embeddings', label: 'Embeddings', description: 'Semantic similarity search' },
  { key: 'Hybrid', label: 'Hybrid', description: 'Cypher + embeddings combined (recommended)' },
] as const;

const EMBEDDING_MODELS = [
  { key: 'minilm', label: 'MiniLM (Fast)', description: '384-dim · faster, lighter' },
  { key: 'mpnet', label: 'MPNet (Quality)', description: '768-dim · higher accuracy' },
] as const;

export function Settings() {
  const {
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
  const optionSelectedClass = isDark ? 'border-violet-400/60 bg-violet-500/10' : 'border-purple-500 bg-purple-50';
  const optionIdleClass = isDark ? 'border-slate-700 bg-slate-950/60 hover:border-slate-500' : 'border-gray-200 hover:border-gray-300';

  const { data: health, refetch: refetchHealth } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 30000,
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
              <StatusRow 
                label="ML Engine" 
                value={health.ml_available ? `Online (${health.ml_model_type || 'XGBoost'})` : 'Offline'} 
                ok={health.ml_available || false} 
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
        <div>
          <h3 className={cn('font-semibold flex items-center gap-2', pageText)}>
            <Sparkles className="w-4 h-4" /> Embedding Model
          </h3>
          <p className={cn('text-xs mt-1', mutedText)}>
            Both models are prebuilt and ready. Switching takes effect on the next query.
          </p>
        </div>
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
        {/* Status badge */}
        {health?.embeddings_built && (
          <div className={cn('flex items-center gap-2 text-sm', isDark ? 'text-emerald-300' : 'text-green-600')}>
            <CheckCircle className="w-4 h-4" />
            {health.embedding_count.toLocaleString()} vectors loaded
            <span className={cn('text-xs', mutedText)}>· active model: {health.embedding_count > 0 ? embeddingModel : '—'}</span>
          </div>
        )}
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
