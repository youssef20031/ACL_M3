import axios, { AxiosError } from 'axios';

// In development: falls back to localhost (proxied by Vite).
// In production: set VITE_API_URL to your Railway backend URL.
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds
});

const EMBEDDINGS_BUILD_TIMEOUT_MS = 15 * 60 * 1000; // 15 minutes

// Response types
export interface ConnectionResponse {
  success: boolean;
  message: string;
  stats?: {
    total_nodes: number;
    total_relationships: number;
  };
}

export interface QueryResponse {
  answer: string;
  intent: string;
  entities: Record<string, any>;
  cypher_query: string;
  kg_context: string;
  embedding_context?: string;
  embedding_used: boolean;
  results: Array<Record<string, any>>;
  graph_data?: {
    nodes: Array<{
      id: string;
      label: string;
      type: string;
      data?: Record<string, any>;
    }>;
    edges: Array<{
      from: string;
      to: string;
      label: string;
    }>;
  };
}

export interface TriviaQuestion {
  question: string;
  options: string[];
  category: string;
  difficulty: string;
  question_id: string;
}

export interface TriviaAnswerResponse {
  correct: boolean;
  feedback: string;
  correct_answer?: string;
}

export interface HealthStatus {
  status: string;
  neo4j: string;
  neo4j_stats?: {
    total_nodes: number;
    total_relationships: number;
  };
  llm_available: boolean;
  embeddings_built: boolean;
  embedding_count: number;
  embeddings_building?: boolean;
  embedding_build_error?: string | null;
}

export interface EmbeddingBuildResponse {
  success: boolean;
  count: number;
  message: string;
  started?: boolean;
  building?: boolean;
  model?: string | null;
}

export interface ImageSearchResponse {
  query: string;
  image_url?: string | null;
  source?: string | null;
}

// API functions
export const apiService = {
  // Health & Connection
  async getHealth(): Promise<HealthStatus> {
    const { data } = await api.get<HealthStatus>('/health');
    return data;
  },

  async connectNeo4j(uri: string, username: string, password: string): Promise<ConnectionResponse> {
    const { data } = await api.post<ConnectionResponse>('/api/connection/connect', {
      uri,
      username,
      password,
    });
    return data;
  },

  async disconnectNeo4j(): Promise<{ success: boolean; message: string }> {
    const { data } = await api.post('/api/connection/disconnect');
    return data;
  },

  // Query
  async queryFPL(
    question: string,
    model: string = 'qwen-2.5-coder',
    retrievalMethod: string = 'Hybrid',
    embeddingModel: string = 'minilm',
    isFirstMessage: boolean = false
  ): Promise<QueryResponse> {
    const { data } = await api.post<QueryResponse>('/api/query', {
      question,
      model,
      retrieval_method: retrievalMethod,
      embedding_model: embeddingModel,
      is_first_message: isFirstMessage,
    });
    return data;
  },

  // Embeddings
  async buildEmbeddings(model: string = 'minilm'): Promise<EmbeddingBuildResponse> {
    const { data } = await api.post<EmbeddingBuildResponse>(
      '/api/embeddings/build',
      { model },
      { timeout: EMBEDDINGS_BUILD_TIMEOUT_MS }
    );
    return data;
  },

  // Trivia
  async getNewTriviaQuestion(): Promise<TriviaQuestion> {
    const { data } = await api.get<TriviaQuestion>('/api/trivia/new');
    return data;
  },

  async checkTriviaAnswer(questionId: string, answer: string): Promise<TriviaAnswerResponse> {
    const { data } = await api.post<TriviaAnswerResponse>('/api/trivia/answer', {
      question_id: questionId,
      answer,
    });
    return data;
  },

  // Players
  async searchPlayers(
    query: string,
    options?: { limit?: number; includeAvatars?: boolean }
  ): Promise<{ players: Array<Record<string, any>> }> {
    const { data } = await api.post('/api/players/search', {
      query,
      limit: options?.limit ?? 20,
      include_avatars: options?.includeAvatars ?? true,
    });
    return data;
  },

  async getPlayerStats(playerName: string, season?: string): Promise<{ stats: Record<string, any> }> {
    const { data } = await api.post('/api/players/stats', { player_name: playerName, season });
    return data;
  },

  async comparePlayers(
    player1: string,
    player2: string,
    season?: string
  ): Promise<{ comparison: Array<Record<string, any>> }> {
    const { data } = await api.post('/api/players/compare', {
      player1,
      player2,
      season,
    });
    return data;
  },

  async searchImage(query: string): Promise<ImageSearchResponse> {
    const { data } = await api.get<ImageSearchResponse>('/api/images/search', {
      params: { query },
    });
    return data;
  },

  // Data Management
  async loadFPLData(clearExisting: boolean = true): Promise<{ success: boolean; stats: any }> {
    const { data } = await api.post('/api/data/load', { clear_existing: clearExisting });
    return data;
  },

  async getDatabaseStats(): Promise<Record<string, any>> {
    const { data } = await api.get('/api/stats/database');
    return data;
  },
};

// Error handling
export const handleApiError = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail: string }>;
    if (axiosError.response?.data?.detail) {
      return axiosError.response.data.detail;
    }
    if (axiosError.message) {
      return axiosError.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unknown error occurred';
};

export default apiService;