import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export type ThemeMode = 'light' | 'dark';

export interface AppState {
  // Connection
  neo4jConnected: boolean;
  neo4jStats: {
    total_nodes: number;
    total_relationships: number;
  } | null;
  setNeo4jConnected: (connected: boolean) => void;
  setNeo4jStats: (stats: { total_nodes: number; total_relationships: number } | null) => void;

  // Embeddings
  embeddingsBuilt: boolean;
  embeddingCount: number;
  setEmbeddingsBuilt: (built: boolean) => void;
  setEmbeddingCount: (count: number) => void;

  // ML Predictions
  mlAvailable: boolean;
  mlModelType: string | null;
  setMlAvailable: (available: boolean) => void;
  setMlModelType: (modelType: string | null) => void;

  // Chat
  chatHistory: Message[];
  addMessage: (message: Message) => void;
  clearChat: () => void;

  // Settings
  selectedModel: string;
  retrievalMethod: 'Baseline' | 'Embeddings' | 'Hybrid';
  embeddingModel: 'minilm' | 'mpnet';
  setSelectedModel: (model: string) => void;
  setRetrievalMethod: (method: 'Baseline' | 'Embeddings' | 'Hybrid') => void;
  setEmbeddingModel: (model: 'minilm' | 'mpnet') => void;

  // Theme
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  toggleTheme: () => void;

  // Trivia
  triviaScore: number;
  triviaTotal: number;
  incrementScore: () => void;
  incrementTotal: () => void;
  resetTriviaScore: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Connection
      neo4jConnected: false,
      neo4jStats: null,
      setNeo4jConnected: (connected) => set({ neo4jConnected: connected }),
      setNeo4jStats: (stats) => set({ neo4jStats: stats }),

      // Embeddings
      embeddingsBuilt: false,
      embeddingCount: 0,
      setEmbeddingsBuilt: (built) => set({ embeddingsBuilt: built }),
      setEmbeddingCount: (count) => set({ embeddingCount: count }),

      // ML Predictions
      mlAvailable: false,
      mlModelType: null,
      setMlAvailable: (available) => set({ mlAvailable: available }),
      setMlModelType: (modelType) => set({ mlModelType: modelType }),

      // Chat
      chatHistory: [],
      addMessage: (message) =>
        set((state) => ({
          chatHistory: [...state.chatHistory, message],
        })),
      clearChat: () => set({ chatHistory: [] }),

      // Settings
      selectedModel: 'llama-3.3-70b',
      retrievalMethod: 'Hybrid',
      embeddingModel: 'minilm',
      setSelectedModel: (model) => set({ selectedModel: model }),
      setRetrievalMethod: (method) => set({ retrievalMethod: method }),
      setEmbeddingModel: (model) => set({ embeddingModel: model }),

      // Theme
      theme: 'light',
      setTheme: (theme) => set({ theme }),
      toggleTheme: () =>
        set((state) => ({ theme: state.theme === 'dark' ? 'light' : 'dark' })),

      // Trivia
      triviaScore: 0,
      triviaTotal: 0,
      incrementScore: () => set((state) => ({ triviaScore: state.triviaScore + 1 })),
      incrementTotal: () => set((state) => ({ triviaTotal: state.triviaTotal + 1 })),
      resetTriviaScore: () => set({ triviaScore: 0, triviaTotal: 0 }),
    }),
    {
      name: 'fpl-app-storage',
      version: 2, // bump version to force migration
      partialize: (state) => ({
        selectedModel: state.selectedModel,
        retrievalMethod: state.retrievalMethod,
        embeddingModel: state.embeddingModel,
        theme: state.theme,
        triviaScore: state.triviaScore,
        triviaTotal: state.triviaTotal,
      }),
      migrate: (persistedState: any, version: number) => {
        // v1 → v2: migrate legacy HuggingFace models to llama-3.3-70b
        if (version < 2) {
          const legacyModels = ['qwen-2.5-coder', 'qwen-2.5-7b', 'llama-3.2-3b', 'phi-3-mini', 'gemma-2-2b'];
          if (legacyModels.includes(persistedState?.selectedModel)) {
            persistedState.selectedModel = 'llama-3.3-70b';
          }
        }
        return persistedState;
      },
    }
  )
);