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

/**
 * One row of player data from the search, stats and compare endpoints. Its keys depend on the
 * Cypher query that produced it, so the pages read the fields they expect and fall back when absent.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- keys and value types vary by query
export type PlayerRow = Record<string, any>;

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
  entities: Record<string, unknown>;
  cypher_query: string;
  kg_context: string;
  embedding_context?: string;
  embedding_used: boolean;
  results: Array<Record<string, unknown>>;
  graph_data?: {
    nodes: Array<{
      id: string;
      label: string;
      type: string;
      data?: Record<string, unknown>;
    }>;
    edges: Array<{
      from: string;
      to: string;
      label: string;
    }>;
  };
  /** Visual blocks drawn with the answer (services/answer_cards.py). Absent from older backends. */
  cards?: AnswerCard[];
  /** Suggested next questions, each routed by the backend's intent classifier. */
  follow_ups?: string[];
}

export interface QueryRequestBody {
  question: string;
  model: string;
  retrieval_method: string;
  embedding_model: string;
  is_first_message: boolean;
  chat_history: Array<{ role: string; content: string }>;
}

/** Events of POST /api/query/stream, in order: cards (as soon as they're built), the answer
 * in delta pieces, the final answer text, then the full response. */
export type AnswerStreamEvent =
  | { event: 'cards'; data: { cards: AnswerCard[]; follow_ups: string[] } }
  | { event: 'delta'; data: { text: string } }
  | { event: 'answer'; data: { text: string } }
  | { event: 'done'; data: QueryResponse }
  | { event: 'error'; data: { detail: string } };

/** The backend has no stream endpoint (an older deploy): use queryFPL instead. */
export class StreamUnavailableError extends Error {}

// --- Assistant answer cards -------------------------------------------------
// Built from structured data (query rows, the graph, predictions), never from the
// answer text. The backend may add card types later: unknown ones are skipped.

export type CardTone = 'accent' | 'info' | 'warning' | 'danger' | 'neutral';
export type CardIcon =
  | 'trophy'
  | 'flame'
  | 'medal'
  | 'star'
  | 'zap'
  | 'clock'
  | 'home'
  | 'trend-up'
  | 'trend-down'
  | 'shield'
  | 'target'
  | 'calendar'
  | 'alert';
/** int, dec1/dec2 (decimals), money (£m), compact (1.2M). */
export type CardValueFormat = 'int' | 'dec1' | 'dec2' | 'money' | 'compact';
export type MatchResult = 'W' | 'D' | 'L';

/** A "notable stat" computed from the data (streaks, ranks, records). */
export interface CardHighlight {
  label: string;
  value: string;
  detail: string | null;
  tone: CardTone;
  icon: CardIcon;
}

export interface CardPlayer {
  /** Exact player name in the graph (used for links). */
  name: string;
  team: string | null;
  position: PositionCode | null;
  photo: string | null;
  /** FPL availability (a/d/i/s/u/n); null where it doesn't apply (past seasons). */
  status: string | null;
  chance: number | null;
  news: string | null;
}

export interface CardStat {
  key: string;
  label: string;
  value: number | null;
  format: CardValueFormat;
}

export interface CardOpponent {
  team: string;
  home: boolean;
  /** FPL fixture difficulty 1–5 for the player's/team's side, when known. */
  difficulty: number | null;
  kickoff_time: string | null;
  gameweek: number | null;
}

export interface CardPrediction {
  gameweek: number;
  points: number;
  /** Several for a double gameweek, none for a blank one. */
  opponents: CardOpponent[];
  rank: number;
  pool: number;
  position_rank: number;
  position_pool: number;
}

export interface CardMatch {
  season: string;
  gameweek: number | null;
  points: number;
  minutes: number;
  goals: number;
  assists: number;
  opponent: string | null;
  home: boolean | null;
  /** From the player's side, e.g. "4-1". */
  score: string | null;
  result: MatchResult | null;
}

/** Price (£m) and ownership (managers) by gameweek over one season. */
export interface CardTrend {
  season: string;
  points: Array<{ gameweek: number; price: number; selected: number | null }>;
  price_change: number;
  selected_change: number | null;
}

/** Expected goals/assists over one season (FPL publishes them for recent seasons only). */
export interface CardUnderlying {
  season: string;
  minutes: number;
  xg: number;
  xa: number;
  xgi: number;
  goals: number;
  assists: number;
  xg_per90: number | null;
  xa_per90: number | null;
  xgi_per90: number | null;
  /** Expected goals conceded while on the pitch, next to the goals actually conceded. */
  xgc: number;
  goals_conceded: number;
}

export interface PlayerAnswerCard {
  type: 'player';
  id: string;
  emphasis: 'stats' | 'prediction';
  player: CardPlayer;
  /** null = all seasons. */
  season: string | null;
  season_label: string;
  stats: CardStat[];
  /** Points per season, oldest first (all-seasons cards only). */
  seasons: Array<{ season: string; team: string | null; points: number; goals: number; assists: number; games: number }>;
  /** Most recent matches in scope, oldest first. */
  form: CardMatch[];
  /** The gameweek the question asked about. */
  match: CardMatch | null;
  prediction: CardPrediction | null;
  /** The club's next matches with FPL difficulty (current players). */
  next_fixtures?: CardOpponent[];
  /** For the season asked about, else the player's latest season. */
  trend?: CardTrend | null;
  underlying?: CardUnderlying | null;
  highlights: CardHighlight[];
}

export interface ComparisonMetric {
  key: string;
  label: string;
  format: CardValueFormat;
  values: [number | null, number | null];
  /** Index of the higher value; null for ties, missing values and unscored rows. */
  better: 0 | 1 | null;
  scored: boolean;
}

export interface ComparisonAnswerCard {
  type: 'comparison';
  id: string;
  emphasis: 'stats' | 'prediction';
  season: string | null;
  season_label: string;
  players: [CardPlayer, CardPlayer];
  metrics: ComparisonMetric[];
  /** How many scored metrics each player leads. */
  leads: [number, number];
  predictions: [CardPrediction | null, CardPrediction | null];
  /** Each player's next three matches (current players). */
  next_fixtures?: [CardOpponent[], CardOpponent[]];
  form: [CardMatch[], CardMatch[]];
  highlights: CardHighlight[];
}

export interface LeaderboardRow {
  /** Shared by ties; null in grouped (per-position) lists. */
  rank: number | null;
  player: CardPlayer;
  value: number | null;
  values: Record<string, number | null>;
  detail: string | null;
  group: PositionCode | null;
}

export interface LeaderboardAnswerCard {
  type: 'leaderboard';
  id: string;
  title: string;
  subtitle: string;
  icon: CardIcon;
  metric: { key: string; label: string; format: CardValueFormat; unit: string };
  columns: Array<{ key: string; label: string; format: CardValueFormat }>;
  /** Rows grouped by position (the top players of each). */
  grouped: boolean;
  season: string | null;
  rows: LeaderboardRow[];
  highlights: CardHighlight[];
}

export interface PredictionRow {
  rank: number;
  player: CardPlayer;
  points: number;
  opponents: CardOpponent[];
  price: number | null;
}

export interface PredictionAnswerCard {
  type: 'prediction';
  id: string;
  title: string;
  gameweek: number;
  deadline: string | null;
  position: PositionCode | null;
  captain: boolean;
  model: string | null;
  rows: PredictionRow[];
  highlights: CardHighlight[];
}

export interface TeamAnswerCard {
  type: 'team';
  id: string;
  team: string;
  season: string;
  standing: {
    position: number;
    played: number;
    won: number;
    drawn: number;
    lost: number;
    goals_for: number;
    goals_against: number;
    goal_difference: number;
    points: number;
  } | null;
  teams_in_table: number;
  /** Last five results, oldest first. */
  form: Array<{ opponent: string; home: boolean; score: string; result: MatchResult; gameweek: number | null }>;
  next_fixtures: CardOpponent[];
  top_players: Array<{ player: CardPlayer; points: number; goals: number; assists: number; games: number }>;
  /** Injured, doubtful and suspended players (first four). */
  flagged: CardPlayer[];
  flagged_total: number;
  highlights: CardHighlight[];
}

export interface FixturesAnswerCard {
  type: 'fixtures';
  id: string;
  title: string;
  subtitle: string;
  /** One team (its results) or two (head to head); results are from teams[0]'s side. */
  teams: string[];
  record: { team: string; played: number; won: number; drawn: number; lost: number; goals_for: number; goals_against: number };
  /** Newest first. */
  matches: Array<{
    season: string | null;
    gameweek: number | null;
    home_team: string;
    away_team: string;
    home_score: number;
    away_score: number;
    kickoff_time: string | null;
    result: MatchResult | null;
  }>;
  total: number;
  highlights: CardHighlight[];
}

export interface SeasonAnswerCard {
  type: 'season';
  id: string;
  season: string;
  season_label: string;
  in_progress: boolean;
  stats: CardStat[];
  leaders: Array<{ player: CardPlayer; points: number; goals: number; assists: number }>;
  highlights: CardHighlight[];
}

export interface PitchSlot {
  player: CardPlayer;
  points: number | null;
  cost: number | null;
  captain: boolean;
  vice_captain: boolean;
  opponents: Array<{ team: string; home: boolean }>;
}

/** Best XI: the optimiser's predicted squad, or a finished gameweek's dream team. */
export interface PitchAnswerCard {
  type: 'pitch';
  id: string;
  mode: TeamOfWeekMode;
  gameweek: number;
  formation: string;
  /** Predicted mode counts the captain twice. */
  total_points: number;
  total_cost: number | null;
  budget: number | null;
  starters: PitchSlot[];
  /** Predicted mode only. */
  bench: PitchSlot[];
  highlights: CardHighlight[];
}

export type AnswerCard =
  | PlayerAnswerCard
  | ComparisonAnswerCard
  | LeaderboardAnswerCard
  | PredictionAnswerCard
  | TeamAnswerCard
  | FixturesAnswerCard
  | SeasonAnswerCard
  | PitchAnswerCard;

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
  features_used?: Record<string, unknown>;
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
  metadata: Record<string, unknown>;
}

export interface BestValueResponse {
  predictions: (MLPrediction & { value: number; cost: number })[];
  metadata: Record<string, unknown>;
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

/** One player of a saved model pick, scored with that gameweek's FPL points. */
export interface ModelReviewPlayer {
  name: string;
  /** FPL's short display name ("Raya"), when known. */
  web_name: string | null;
  team: string;
  position: PositionCode;
  /** £m. */
  cost: number | null;
  /** The model's prediction for this gameweek. */
  predicted_points: number | null;
  /** Raw FPL points this gameweek; null until one of his matches has finished. */
  points: number | null;
  minutes: number | null;
  /** 0 = on the bench after auto-subs, 1, 2 = captain whose points count double (the vice if the captain didn't play). */
  multiplier: number;
  /** As picked. */
  captain: boolean;
  /** As picked. */
  vice_captain: boolean;
  /** null while his matches are still to finish. */
  played: boolean | null;
  /** Came off the bench via an auto-sub. */
  subbed_in: boolean;
  /** Picked to start, didn't play, replaced. */
  subbed_out: boolean;
  opponents: Array<{ team: string; home: boolean }>;
  photo: string | null;
}

/** The squad the model picked before a past deadline, and how it actually scored. */
export interface ModelReviewResponse {
  season: string;
  gameweek: number;
  /** Every gameweek with a saved pick, ascending. */
  gameweeks: number[];
  /** finished = FPL has marked the gameweek finished. */
  status: 'live' | 'finished';
  deadline: string | null;
  /** When the pick was saved (ISO). */
  archived_at: string | null;
  /** manual = recorded by hand from the FPL app. */
  source: 'pipeline' | 'manual';
  note: string | null;
  /** Final XI after auto-subs, e.g. "4-4-2". */
  formation: string;
  /** The model's XI before auto-subs. */
  picked_formation: string;
  /** Model's XI + captain again, as predicted. */
  predicted_points: number;
  /** FPL score: final XI × multipliers. */
  points: number;
  /** Points left on the bench after auto-subs. */
  bench_points: number;
  fpl_average: number | null;
  fpl_highest: number | null;
  /** £m. */
  total_cost: number | null;
  /** £m. */
  budget: number | null;
  /** Final XI after auto-subs, GK, DEF, MID, FWD order. */
  starters: ModelReviewPlayer[];
  /** Final bench in priority order, goalkeeper first. */
  bench: ModelReviewPlayer[];
  auto_subs: Array<{ out: string; in: string }>;
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

  async getMLStatus(): Promise<unknown> {
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

  /**
   * Streams an answer from POST /api/query/stream, calling onEvent for each server-sent
   * event. Uses fetch, since axios can't read a response body as it arrives. Throws
   * StreamUnavailableError when the endpoint is missing, and an Error with the server's
   * detail for other HTTP errors (e.g. the rate limit).
   */
  async streamQuery(body: QueryRequestBody, onEvent: (event: AnswerStreamEvent) => void, signal?: AbortSignal): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal,
    });
    if (response.status === 404 || response.status === 405) throw new StreamUnavailableError();
    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const json = await response.json();
        if (typeof json?.detail === 'string') detail = json.detail;
      } catch {
        // not JSON: keep the status message
      }
      throw new Error(detail);
    }
    if (!response.body) throw new StreamUnavailableError();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end: number;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        let event = 'message';
        const data: string[] = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
        }
        if (data.length) onEvent({ event, data: JSON.parse(data.join('\n')) } as AnswerStreamEvent);
      }
    }
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
  ): Promise<{ players: PlayerRow[] }> {
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
  ): Promise<{ stats: PlayerRow }> {
    const { data } = await api.post('/api/players/stats', { player_name: playerName, season }, { signal });
    return data;
  },

  async comparePlayers(
    player1: string,
    player2: string,
    season?: string
  ): Promise<{ comparison: PlayerRow[] }> {
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

  /** The model's saved pick for a past gameweek (default: the latest whose deadline passed). 404 when none. */
  async getModelReview(gameweek?: number): Promise<ModelReviewResponse> {
    const { data } = await api.get<ModelReviewResponse>('/api/model-review', {
      params: gameweek ? { gameweek } : undefined,
    });
    return data;
  },

  // Data Management
  async loadFPLData(clearExisting: boolean = true): Promise<{ success: boolean; stats: unknown }> {
    const { data } = await api.post('/api/data/load', { clear_existing: clearExisting });
    return data;
  },

  async getDatabaseStats(): Promise<Record<string, unknown>> {
    const { data } = await api.get('/api/stats/database');
    return data;
  },

  // Cookie-free page-view counter (aggregate counts only; called only after consent).
  // Fire-and-forget: analytics must never surface an error to the visitor.
  trackPageview(path: string, referrer: string | null): void {
    api.post('/api/analytics/pageview', { path, referrer }, { timeout: 5000 }).catch(() => undefined);
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

/** True for an HTTP 404, e.g. a resource that doesn't exist yet (not a failure worth retrying). */
export const isNotFoundError = (error: unknown): boolean =>
  axios.isAxiosError(error) && error.response?.status === 404;

export default apiService;