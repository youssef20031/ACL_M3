import axios, { AxiosError } from 'axios';

// Production builds use relative URLs: the site and the API share an origin
// (Caddy serves dist/ and proxies /api, /health and /ws — see deploy/Caddyfile).
const API_BASE_URL = import.meta.env.MODE === 'production' ? '' : (import.meta.env.VITE_API_URL || '');

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
  ml_available?: boolean;
  ml_model_type?: string | null;
}

export interface MLPrediction {
  player_name: string;
  predicted_points: number;
  confidence_interval?: [number, number];
  features_used?: Record<string, any>;
  model_version: string;
  team_name?: string | null;
  fixture_context?: {
    opponent_team: string;
    home_away: "home" | "away";
    match_date: string;  // ISO datetime
    matchday: number;
    fixture_id: string;
  } | null;
  /** FPL availability: a available, d doubtful, i injured, s suspended, u unavailable, n not in squad */
  status?: 'a' | 'd' | 'i' | 's' | 'u' | 'n';
  /** 0-100, or null when FPL hasn't set one */
  chance_of_playing_next_round?: number | null;
  /** Official FPL team news, e.g. "Hamstring injury - Expected back 24 Oct" */
  news?: string;
}

export interface TopPerformersResponse {
  predictions: MLPrediction[];
  metadata: Record<string, any>;
}

export interface BestValueResponse {
  predictions: (MLPrediction & { value: number; cost: number })[];
  metadata: Record<string, any>;
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

// --- Season dashboard (Pitch Me redesign) ---------------------------------
export type FixtureStatusCode = 'scheduled' | 'in_progress' | 'finished';
export type PositionCode = 'GK' | 'DEF' | 'MID' | 'FWD';

export interface SeasonInfo {
  season: string;
  current_gameweek: number | null;
  next_gameweek: number | null;
  next_deadline: string | null;
  finished_gameweeks: number[];
  updated_at: string | null;
}

export interface SeasonFixture {
  id: string;
  gameweek: number;
  kickoff_time: string | null;
  status: FixtureStatusCode;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  home_difficulty: number | null;
  away_difficulty: number | null;
}

export interface FixturesResponse {
  season: string;
  gameweek: number;
  gameweeks: number[];
  fixtures: SeasonFixture[];
}

export interface TableRow {
  position: number;
  team: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
  form: Array<'W' | 'D' | 'L'>;
  next_fixture: { opponent: string; home: boolean; kickoff_time: string | null } | null;
}

export interface LeagueTableResponse {
  season: string;
  updated_at: string | null;
  rows: TableRow[];
}

export interface InjuredPlayer {
  name: string;
  web_name: string | null;
  team: string;
  position: PositionCode;
  status: 'd' | 'i' | 's' | 'u' | 'n';
  chance_of_playing_next_round: number | null;
  news: string | null;
  news_added: string | null;
  photo: string | null;
}

export interface InjuriesResponse {
  updated_at: string | null;
  players: InjuredPlayer[];
}

export interface TeamOfWeekPlayer {
  name: string;
  team: string;
  position: PositionCode;
  points: number;
  cost: number | null;
  captain: boolean;
  vice_captain: boolean;
  opponents: Array<{ team: string; home: boolean }>;
  status: string | null;
  chance_of_playing_next_round: number | null;
  news: string | null;
  photo: string | null;
}

export type TeamOfWeekMode = 'predicted' | 'actual';

export interface TeamOfWeekResponse {
  mode: TeamOfWeekMode;
  gameweek: number;
  formation: string;
  total_points: number;
  total_cost: number | null;
  budget: number | null;
  starters: TeamOfWeekPlayer[];
  bench: TeamOfWeekPlayer[];
}

export type PlayerLeaderSort =
  | 'total_points'
  | 'goals'
  | 'assists'
  | 'clean_sheets'
  | 'bonus'
  | 'points_per_game'
  | 'minutes';

export interface PlayerLeader {
  name: string;
  team: string | null;
  /** The player's position in that season. */
  position: PositionCode | null;
  games: number;
  minutes: number;
  goals: number;
  assists: number;
  clean_sheets: number;
  bonus: number;
  total_points: number;
  /** total_points / games, 2 dp. */
  points_per_game: number;
  /** FPL availability; current season only (null for past seasons). */
  status: string | null;
  chance_of_playing_next_round: number | null;
  news: string | null;
  photo: string | null;
}

export interface PlayerLeadersResponse {
  season: string;
  sort: PlayerLeaderSort;
  position: PositionCode | null;
  players: PlayerLeader[];
}

export interface PlayerLeadersParams {
  season?: string;
  position?: PositionCode;
  sort?: PlayerLeaderSort;
  /** 1–100, default 50. */
  limit?: number;
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

  // ML Predictions
  async predictPlayer(playerName: string): Promise<MLPrediction> {
    const { data } = await api.post<MLPrediction>('/api/ml/predict/player', {
      player_name: playerName,
    });
    return data;
  },

  async predictTopPerformers(position?: string, topK: number = 10): Promise<TopPerformersResponse> {
    // Try GET endpoint first (faster, uses CSV), fallback to POST (uses Neo4j + ML)
    try {
      const { data } = await api.get<TopPerformersResponse>('/api/ml/top-performers', {
        params: { position, limit: topK },
        timeout: 5000,
      });
      return data;
    } catch (error) {
      // Fallback to POST endpoint if GET fails
      console.warn('CSV endpoint failed, trying Neo4j-based endpoint:', error);
      const { data } = await api.post<TopPerformersResponse>('/api/ml/predict/top-performers', {
        position,
        top_k: topK,
      });
      return data;
    }
  },

  async predictBestValue(position?: string, topK: number = 10): Promise<BestValueResponse> {
    // Try GET endpoint first (faster, uses CSV), fallback to POST (uses Neo4j + ML)
    try {
      const { data } = await api.get<BestValueResponse>('/api/ml/best-value', {
        params: { position, limit: topK },
        timeout: 5000,
      });
      return data;
    } catch (error) {
      // Fallback to POST endpoint if GET fails
      console.warn('CSV endpoint failed, trying Neo4j-based endpoint:', error);
      const { data } = await api.post<BestValueResponse>('/api/ml/predict/best-value', {
        position,
        top_k: topK,
      });
      return data;
    }
  },

  async getMLStatus(): Promise<any> {
    const { data } = await api.get('/api/ml/status');
    return data;
  },

  // Query
  async queryFPL(
    question: string,
    model: string = 'gpt-oss-120b',
    retrievalMethod: string = 'Hybrid',
    embeddingModel: string = 'minilm',
    isFirstMessage: boolean = false,
    chatHistory: Array<{ role: string; content: string }> = []
  ): Promise<QueryResponse> {
    const { data } = await api.post<QueryResponse>('/api/query', {
      question,
      model,
      retrieval_method: retrievalMethod,
      embedding_model: embeddingModel,
      is_first_message: isFirstMessage,
      chat_history: chatHistory,
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
  async getNewTriviaQuestion(difficulty?: 'easy' | 'medium' | 'hard'): Promise<TriviaQuestion> {
    const params = difficulty ? { difficulty } : {};
    const { data } = await api.get<TriviaQuestion>('/api/trivia/new', { params });
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
    options?: { limit?: number; includeAvatars?: boolean; signal?: AbortSignal }
  ): Promise<{ players: Array<Record<string, any>> }> {
    const { data } = await api.post(
      '/api/players/search',
      {
        query,
        limit: options?.limit ?? 20,
        include_avatars: options?.includeAvatars ?? false, // Default to false for fast search
      },
      { signal: options?.signal }
    );
    return data;
  },

  async getPlayerStats(
    playerName: string,
    season?: string,
    signal?: AbortSignal
  ): Promise<{ stats: Record<string, any> }> {
    const { data } = await api.post('/api/players/stats', { player_name: playerName, season }, { signal });
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

  async getPlayerLeaders(params: PlayerLeadersParams = {}, signal?: AbortSignal): Promise<PlayerLeadersResponse> {
    const { data } = await api.get<PlayerLeadersResponse>('/api/players/leaders', { params, signal });
    return data;
  },

  async searchImage(query: string, signal?: AbortSignal): Promise<ImageSearchResponse> {
    const { data } = await api.get<ImageSearchResponse>('/api/images/search', {
      params: { query },
      signal,
    });
    return data;
  },

  // Season dashboard
  async getCurrentSeason(): Promise<SeasonInfo> {
    const { data } = await api.get<SeasonInfo>('/api/season/current');
    return data;
  },

  async getFixtures(gameweek?: number): Promise<FixturesResponse> {
    const { data } = await api.get<FixturesResponse>('/api/fixtures', { params: gameweek ? { gameweek } : undefined });
    return data;
  },

  async getLeagueTable(): Promise<LeagueTableResponse> {
    const { data } = await api.get<LeagueTableResponse>('/api/table');
    return data;
  },

  async getInjuries(team?: string): Promise<InjuriesResponse> {
    const { data } = await api.get<InjuriesResponse>('/api/injuries', { params: team ? { team } : undefined });
    return data;
  },

  async getTeamOfTheWeek(mode: TeamOfWeekMode = 'predicted', gameweek?: number): Promise<TeamOfWeekResponse> {
    const { data } = await api.get<TeamOfWeekResponse>('/api/team-of-the-week', {
      params: { mode, ...(gameweek ? { gameweek } : {}) },
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