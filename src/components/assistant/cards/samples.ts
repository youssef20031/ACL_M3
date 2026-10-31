import type { AnswerCard, CardMatch, CardPlayer, PitchSlot } from '../../../services/api';

/*
 * SAMPLE DATA for the hidden /design page only: made-up players and numbers that show
 * every answer card type. Nothing in the app reads these.
 */

const player = (name: string, team: string, position: CardPlayer['position'], extra: Partial<CardPlayer> = {}): CardPlayer => ({
  name,
  team,
  position,
  photo: null,
  status: 'a',
  chance: null,
  news: null,
  ...extra,
});

const match = (gameweek: number, opponent: string, home: boolean, points: number, goals = 0, assists = 0, minutes = 90): CardMatch => ({
  season: '2026-27',
  gameweek,
  points,
  minutes,
  goals,
  assists,
  opponent,
  home,
  score: minutes ? '2-1' : null,
  result: minutes ? 'W' : null,
});

const slot = (
  name: string,
  team: string,
  position: CardPlayer['position'],
  points: number,
  cost: number,
  opponent: string,
  home: boolean,
  extra: Partial<PitchSlot> = {},
): PitchSlot => ({
  player: player(name, team, position),
  points,
  cost,
  captain: false,
  vice_captain: false,
  opponents: [{ team: opponent, home }],
  ...extra,
});

const striker = player('Sample Striker', 'Arsenal', 'FWD');
const winger = player('Example Winger', 'Liverpool', 'MID', { status: 'd', chance: 75, news: 'Sample news: knock, 75% chance of playing' });
const defender = player('Demo Defender', 'Chelsea', 'DEF');
const midfielder = player('Mock Midfielder', 'Newcastle', 'MID');

const strikerForm = [
  match(1, 'Chelsea', true, 2),
  match(2, 'Everton', false, 13, 2),
  match(3, 'Fulham', true, 9, 1),
  match(4, 'Brighton', false, 0, 0, 0, 0),
  match(5, 'Newcastle', true, 8, 1),
  match(6, 'Leeds', false, 15, 2, 1),
];

export const SAMPLE_CARDS: AnswerCard[] = [
  {
    type: 'player',
    id: 'sample-player',
    emphasis: 'stats',
    player: striker,
    season: '2026-27',
    season_label: '2026-27 so far',
    stats: [
      { key: 'total_points', label: 'Points', value: 47, format: 'int' },
      { key: 'goals', label: 'Goals', value: 6, format: 'int' },
      { key: 'assists', label: 'Assists', value: 1, format: 'int' },
      { key: 'bonus', label: 'Bonus', value: 7, format: 'int' },
      { key: 'points_per_game', label: 'Pts/game', value: 9.4, format: 'dec1' },
      { key: 'minutes', label: 'Minutes', value: 450, format: 'int' },
    ],
    seasons: [],
    form: strikerForm,
    match: null,
    prediction: {
      gameweek: 7,
      points: 7.9,
      opponents: [{ team: 'Sunderland', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 }],
      rank: 2,
      pool: 667,
      position_rank: 1,
      position_pool: 79,
    },
    next_fixtures: [
      { team: 'Sunderland', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 },
      { team: 'Liverpool', home: false, difficulty: 4, kickoff_time: null, gameweek: 8 },
      { team: 'Newcastle', home: true, difficulty: 3, kickoff_time: null, gameweek: 9 },
      { team: 'Man City', home: false, difficulty: 5, kickoff_time: null, gameweek: 10 },
      { team: 'Fulham', home: true, difficulty: 2, kickoff_time: null, gameweek: 11 },
    ],
    trend: {
      season: '2026-27',
      points: [
        { gameweek: 1, price: 9.0, selected: 1_200_000 },
        { gameweek: 2, price: 9.0, selected: 1_450_000 },
        { gameweek: 3, price: 9.1, selected: 1_900_000 },
        { gameweek: 4, price: 9.2, selected: 2_300_000 },
        { gameweek: 5, price: 9.3, selected: 2_250_000 },
        { gameweek: 6, price: 9.5, selected: 2_800_000 },
      ],
      price_change: 0.5,
      selected_change: 1_600_000,
    },
    underlying: {
      season: '2026-27',
      minutes: 450,
      xg: 3.8,
      xa: 0.9,
      xgi: 4.7,
      goals: 6,
      assists: 1,
      xg_per90: 0.76,
      xa_per90: 0.18,
      xgi_per90: 0.94,
      xgc: 6.1,
      goals_conceded: 5,
    },
    highlights: [
      { label: 'Scoring streak', value: '3 games', detail: 'Scored in each of the last 3 appearances', tone: 'accent', icon: 'flame' },
      { label: 'FWD rank', value: '#1', detail: 'By points among 50 forwards, 2026-27 so far', tone: 'accent', icon: 'medal' },
      { label: 'Biggest haul', value: '15 pts', detail: 'v Leeds (A), GW 6 2026-27', tone: 'accent', icon: 'trophy' },
      { label: 'Minutes', value: '83%', detail: 'Of the available minutes, 2026-27 so far', tone: 'info', icon: 'clock' },
    ],
  },
  {
    type: 'comparison',
    id: 'sample-comparison',
    emphasis: 'stats',
    season: '2026-27',
    season_label: '2026-27 so far',
    players: [striker, winger],
    metrics: [
      { key: 'total_points', label: 'Points', format: 'int', values: [47, 41], better: 0, scored: true },
      { key: 'points_per_game', label: 'Pts/game', format: 'dec1', values: [9.4, 6.8], better: 0, scored: true },
      { key: 'goals', label: 'Goals', format: 'int', values: [6, 3], better: 0, scored: true },
      { key: 'assists', label: 'Assists', format: 'int', values: [1, 5], better: 1, scored: true },
      { key: 'bonus', label: 'Bonus', format: 'int', values: [7, 7], better: null, scored: true },
      { key: 'avg_price', label: 'Avg price', format: 'money', values: [9.5, 12.8], better: null, scored: false },
    ],
    leads: [3, 1],
    predictions: [
      { gameweek: 7, points: 7.9, opponents: [{ team: 'Sunderland', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 }], rank: 2, pool: 667, position_rank: 1, position_pool: 79 },
      { gameweek: 7, points: 5.2, opponents: [{ team: 'Man City', home: false, difficulty: 5, kickoff_time: null, gameweek: 7 }], rank: 9, pool: 667, position_rank: 4, position_pool: 298 },
    ],
    next_fixtures: [
      [
        { team: 'Sunderland', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 },
        { team: 'Liverpool', home: false, difficulty: 4, kickoff_time: null, gameweek: 8 },
        { team: 'Newcastle', home: true, difficulty: 3, kickoff_time: null, gameweek: 9 },
      ],
      [
        { team: 'Man City', home: false, difficulty: 5, kickoff_time: null, gameweek: 7 },
        { team: 'Arsenal', home: true, difficulty: 4, kickoff_time: null, gameweek: 8 },
        { team: 'Everton', home: false, difficulty: 2, kickoff_time: null, gameweek: 9 },
      ],
    ],
    form: [strikerForm.slice(-5), [match(2, 'Leeds', true, 6, 0, 1), match(3, 'Spurs', false, 3), match(4, 'Fulham', true, 10, 1, 1), match(5, 'Everton', false, 2), match(6, 'Brentford', true, 8, 1)]],
    highlights: [
      { label: 'Points edge', value: '+6', detail: 'Striker over Winger', tone: 'accent', icon: 'trophy' },
      { label: 'Next GW', value: '7.9 v 5.2', detail: 'The model favours Striker for GW 7', tone: 'accent', icon: 'star' },
    ],
  },
  {
    type: 'leaderboard',
    id: 'sample-leaderboard',
    title: 'Top scorers',
    subtitle: '2026-27 so far',
    icon: 'target',
    metric: { key: 'goals', label: 'Goals', format: 'int', unit: 'goals' },
    columns: [
      { key: 'total_points', label: 'Pts', format: 'int' },
      { key: 'assists', label: 'Ast', format: 'int' },
    ],
    grouped: false,
    season: '2026-27',
    rows: [
      { rank: 1, player: striker, value: 6, values: { total_points: 47, assists: 1 }, detail: null, group: null },
      { rank: 2, player: midfielder, value: 4, values: { total_points: 39, assists: 2 }, detail: null, group: null },
      { rank: 3, player: winger, value: 3, values: { total_points: 41, assists: 5 }, detail: null, group: null },
      { rank: 3, player: defender, value: 3, values: { total_points: 36, assists: 0 }, detail: null, group: null },
      { rank: 5, player: player('Another Forward', 'Spurs', 'FWD'), value: 2, values: { total_points: 22, assists: 1 }, detail: null, group: null },
    ],
    highlights: [
      { label: 'Clear at the top', value: '+2', detail: 'Striker leads Midfielder by 2 goals', tone: 'accent', icon: 'trophy' },
    ],
  },
  {
    type: 'fixture_picks',
    id: 'sample-fixture-picks',
    title: 'Top predicted players in each match',
    gameweek: 7,
    deadline: null,
    per_fixture: 2,
    fixtures: [
      {
        home: 'Man City',
        away: 'Chelsea',
        kickoff_time: '2026-10-17T11:30:00Z',
        players: [
          { player: player('Striker', 'Man City', 'FWD'), points: 8.1, double: false },
          { player: player('Playmaker', 'Chelsea', 'MID'), points: 6.4, double: false },
        ],
      },
      {
        home: 'Spurs',
        away: 'Arsenal',
        kickoff_time: '2026-10-17T14:00:00Z',
        players: [
          { player: player('Winger', 'Arsenal', 'MID'), points: 9.6, double: true },
          { player: player('Another Forward', 'Spurs', 'FWD'), points: 4.5, double: false },
        ],
      },
    ],
  },
  {
    type: 'prediction',
    id: 'sample-prediction',
    title: 'Captain picks',
    gameweek: 7,
    deadline: null,
    position: null,
    captain: true,
    model: 'XGBoost v3',
    rows: [
      { rank: 1, player: striker, points: 7.9, opponents: [{ team: 'Sunderland', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 }], price: 9.5 },
      {
        rank: 2,
        player: midfielder,
        points: 6.4,
        opponents: [
          { team: 'Leeds', home: false, difficulty: 3, kickoff_time: null, gameweek: 7 },
          { team: 'Fulham', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 },
        ],
        price: 7.1,
      },
      { rank: 3, player: winger, points: 5.2, opponents: [{ team: 'Man City', home: false, difficulty: 5, kickoff_time: null, gameweek: 7 }], price: 12.8 },
    ],
    highlights: [
      { label: 'Top pick', value: '+1.5 pts', detail: 'Striker ahead of Midfielder', tone: 'accent', icon: 'star' },
      { label: 'Double gameweek', value: 'Midfielder', detail: 'Two fixtures this gameweek', tone: 'info', icon: 'calendar' },
      { label: 'Fitness doubt', value: 'Winger (75%)', detail: 'Sample news: knock', tone: 'warning', icon: 'alert' },
    ],
  },
  {
    type: 'team',
    id: 'sample-team',
    team: 'Arsenal',
    season: '2026-27',
    standing: { position: 2, played: 6, won: 4, drawn: 1, lost: 1, goals_for: 11, goals_against: 4, goal_difference: 7, points: 13 },
    teams_in_table: 20,
    form: [
      { opponent: 'Chelsea', home: true, score: '2-1', result: 'W', gameweek: 2 },
      { opponent: 'Everton', home: false, score: '1-1', result: 'D', gameweek: 3 },
      { opponent: 'Fulham', home: true, score: '3-0', result: 'W', gameweek: 4 },
      { opponent: 'Brighton', home: false, score: '0-1', result: 'L', gameweek: 5 },
      { opponent: 'Leeds', home: true, score: '2-0', result: 'W', gameweek: 6 },
    ],
    next_fixtures: [
      { team: 'Sunderland', home: true, difficulty: 2, kickoff_time: null, gameweek: 7 },
      { team: 'Liverpool', home: false, difficulty: 4, kickoff_time: null, gameweek: 8 },
      { team: 'Newcastle', home: true, difficulty: 3, kickoff_time: null, gameweek: 9 },
    ],
    top_players: [
      { player: striker, points: 47, goals: 6, assists: 1, games: 6 },
      { player: player('Sample Keeper', 'Arsenal', 'GK'), points: 33, goals: 0, assists: 0, games: 6 },
      { player: player('Sample Full-back', 'Arsenal', 'DEF'), points: 31, goals: 1, assists: 2, games: 6 },
    ],
    flagged: [player('Sample Centre-back', 'Arsenal', 'DEF', { status: 'i', chance: 0, news: 'Sample news: hamstring injury' })],
    flagged_total: 1,
    highlights: [
      { label: 'Clean sheets', value: '3 in 6', detail: 'This season', tone: 'info', icon: 'shield' },
      { label: 'Next 3 fixtures', value: 'FDR 3.0', detail: 'A mixed run', tone: 'neutral', icon: 'calendar' },
    ],
  },
  {
    type: 'fixtures',
    id: 'sample-fixtures',
    title: 'Arsenal v Chelsea',
    subtitle: 'Head to head · 2020-21 to 2026-27',
    teams: ['Arsenal', 'Chelsea'],
    record: { team: 'Arsenal', played: 8, won: 5, drawn: 2, lost: 1, goals_for: 14, goals_against: 7 },
    matches: [
      { season: '2026-27', gameweek: 2, home_team: 'Arsenal', away_team: 'Chelsea', home_score: 2, away_score: 1, kickoff_time: null, result: 'W' },
      { season: '2025-26', gameweek: 30, home_team: 'Chelsea', away_team: 'Arsenal', home_score: 1, away_score: 1, kickoff_time: null, result: 'D' },
    ],
    total: 8,
    highlights: [
      { label: 'Biggest win', value: 'Arsenal 4-0 Chelsea', detail: 'GW 12, 2023-24', tone: 'accent', icon: 'trophy' },
      { label: 'Goals per game', value: '2.6', detail: '21 goals in 8 matches', tone: 'info', icon: 'target' },
    ],
  },
  {
    type: 'season',
    id: 'sample-season',
    season: '2025-26',
    season_label: '2025-26',
    in_progress: false,
    stats: [
      { key: 'total_gameweeks', label: 'Gameweeks', value: 38, format: 'int' },
      { key: 'total_fixtures', label: 'Matches', value: 380, format: 'int' },
      { key: 'total_goals', label: 'Goals', value: 1050, format: 'int' },
      { key: 'goals_per_game', label: 'Goals/match', value: 2.76, format: 'dec2' },
      { key: 'total_players', label: 'Players', value: 690, format: 'int' },
    ],
    leaders: [
      { player: striker, points: 231, goals: 25, assists: 6 },
      { player: winger, points: 214, goals: 14, assists: 13 },
      { player: midfielder, points: 198, goals: 11, assists: 9 },
    ],
    highlights: [{ label: 'Top scorer', value: 'Striker', detail: '25 goals for Arsenal', tone: 'accent', icon: 'target' }],
  },
  {
    type: 'pitch',
    id: 'sample-pitch',
    mode: 'predicted',
    gameweek: 7,
    formation: '3-5-2',
    total_points: 63.4,
    total_cost: 99.6,
    budget: 100,
    starters: [
      slot('Placeholder Keeper', 'Brighton', 'GK', 4.1, 4.9, 'Everton', true),
      slot('Demo Defender', 'Chelsea', 'DEF', 4.6, 5.6, 'Fulham', false),
      slot('Another Defender', 'Arsenal', 'DEF', 4.4, 6.0, 'Sunderland', true),
      slot('Third Defender', 'Liverpool', 'DEF', 4.2, 7.1, 'Ipswich', true),
      slot('Example Winger', 'Liverpool', 'MID', 6.2, 12.8, 'Ipswich', true, { vice_captain: true }),
      slot('Mock Midfielder', 'Newcastle', 'MID', 5.9, 7.1, 'Leeds', false),
      slot('Sample Playmaker', 'Aston Villa', 'MID', 5.1, 8.0, 'Bournemouth', true),
      slot('Test Midfielder', 'Brighton', 'MID', 4.8, 6.4, 'Everton', true),
      slot('Fifth Midfielder', 'Fulham', 'MID', 4.5, 5.5, 'Chelsea', true),
      slot('Sample Striker', 'Arsenal', 'FWD', 7.9, 9.5, 'Sunderland', true, { captain: true }),
      slot('Second Striker', 'Man City', 'FWD', 5.8, 14.5, 'Spurs', false),
    ],
    bench: [
      slot('Bench Keeper', 'Everton', 'GK', 3.0, 4.0, 'Brighton', false),
      slot('Bench Defender', 'Fulham', 'DEF', 2.6, 4.0, 'Chelsea', true),
      slot('Bench Midfielder', 'Leeds', 'MID', 2.2, 4.5, 'Newcastle', true),
      slot('Bench Forward', 'Burnley', 'FWD', 1.9, 4.5, 'Wolves', false),
    ],
    highlights: [
      { label: 'Captain', value: 'Striker', detail: '7.9 pts, doubled to 15.8', tone: 'accent', icon: 'star' },
      { label: 'Most picked club', value: 'Liverpool', detail: '2 starters', tone: 'info', icon: 'shield' },
      { label: 'In the bank', value: '£0.4m', detail: 'Squad cost £99.6m of £100.0m', tone: 'neutral', icon: 'zap' },
    ],
  },
];

export const SAMPLE_FOLLOW_UPS = [
  'Predicted points for Sample Striker next gameweek',
  'Compare Sample Striker and Example Winger this season',
  'Top 5 forwards by points this season',
];
