import type {
  MlManagerChip,
  MlManagerChipState,
  MlManagerGameweek,
  MlManagerHistoryRow,
  MlManagerResultPlayer,
  MlManagerSquadPlayer,
  MlManagerView,
  ModelReviewPlayer,
  ModelReviewResponse,
} from '../../services/api';

/*
 * SAMPLE DATA for the hidden /design page only: the model's real GW7 2026-27 pick (recorded from
 * the FPL app), with that gameweek's FPL points, minutes, prices and opponents and the model's
 * predictions. Photos are left out, so the silhouettes show. Nothing in the app reads these.
 */

const WEB_NAMES: Record<string, string> = {
  'David Raya Martín': 'Raya',
  'Maxim De Cuyper': 'De Cuyper',
  'Jan Paul van Hecke': 'Van Hecke',
  'Gabriel dos Santos Magalhães': 'Gabriel',
  'Jacob Greaves': 'Greaves',
  'Antoine Semenyo': 'Semenyo',
  'Pascal Groß': 'Groß',
  'Sandro Tonali': 'Tonali',
  'Kiernan Dewsbury-Hall': 'Dewsbury-Hall',
  'Erling Haaland': 'Haaland',
  'Igor Thiago Nascimento Rodrigues': 'Thiago',
  'Martin Dubravka': 'Dubravka',
  'Rayan Cherki': 'Cherki',
  'Milan van Ewijk': 'van Ewijk',
  'Jonah Kusi-Asare': 'Kusi-Asare',
};

const player = (
  name: string,
  team: string,
  position: ModelReviewPlayer['position'],
  cost: number,
  predicted: number,
  points: number,
  minutes: number,
  opponent: { team: string; home: boolean },
  extra: Partial<ModelReviewPlayer> = {},
): ModelReviewPlayer => ({
  name,
  web_name: WEB_NAMES[name] ?? null,
  team,
  position,
  cost,
  predicted_points: predicted,
  points,
  minutes,
  multiplier: 1,
  captain: false,
  vice_captain: false,
  played: minutes > 0,
  subbed_in: false,
  subbed_out: false,
  opponents: [opponent],
  photo: null,
  ...extra,
});

const AT_FOREST = { team: "Nott'm Forest", home: false };
const VS_PALACE = { team: 'Crystal Palace', home: true };
const VS_COVENTRY = { team: 'Coventry City', home: true };
const VS_IPSWICH = { team: 'Ipswich', home: true };

export const SAMPLE_MODEL_REVIEW: ModelReviewResponse = {
  season: '2026-27',
  gameweek: 7,
  gameweeks: [7],
  status: 'finished',
  // 90 minutes before the first kick-off (11:30 UTC on 17 October).
  deadline: '2026-10-17T10:00:00Z',
  archived_at: null,
  source: 'manual',
  note:
    'Recorded from the FPL app: the squad, captaincy and bench order the model’s pick had at the deadline, before picks were saved automatically. Predicted points were recomputed afterwards with the same model from data before the GW7 deadline, without that day’s team news.',
  formation: '4-4-2',
  picked_formation: '3-5-2',
  predicted_points: 59.13,
  points: 70,
  bench_points: 1,
  fpl_average: 59,
  fpl_highest: 122,
  total_cost: 98.0,
  budget: 100.0,
  starters: [
    player('David Raya Martín', 'Arsenal', 'GK', 6.1, 3.68, 9, 90, AT_FOREST, { vice_captain: true }),
    player('Maxim De Cuyper', 'Brighton', 'DEF', 5.2, 4.62, 6, 90, VS_PALACE),
    player('Jan Paul van Hecke', 'Spurs', 'DEF', 4.9, 4.12, 4, 90, VS_COVENTRY),
    player('Gabriel dos Santos Magalhães', 'Arsenal', 'DEF', 7.9, 3.81, 8, 90, AT_FOREST),
    player('Jacob Greaves', 'Ipswich', 'DEF', 3.9, 0.37, 1, 90, { team: 'Man City', home: false }, { subbed_in: true }),
    player('Antoine Semenyo', 'Man City', 'MID', 8.3, 4.61, 1, 13, VS_IPSWICH),
    player('Pascal Groß', 'Brighton', 'MID', 6.1, 4.41, 5, 70, VS_PALACE),
    player('Sandro Tonali', 'Spurs', 'MID', 5.3, 4.31, 4, 90, VS_COVENTRY),
    player('Kiernan Dewsbury-Hall', 'Everton', 'MID', 6.6, 3.81, 2, 90, { team: 'Chelsea', home: true }),
    player('Erling Haaland', 'Man City', 'FWD', 15.7, 8.57, 13, 90, VS_IPSWICH, { captain: true, multiplier: 2 }),
    player('Igor Thiago Nascimento Rodrigues', 'Brentford', 'FWD', 7.8, 3.42, 4, 90, { team: 'Liverpool', home: true }),
  ],
  bench: [
    player('Martin Dubravka', 'Spurs', 'GK', 4.0, 0.05, 0, 0, VS_COVENTRY, { multiplier: 0 }),
    player('Rayan Cherki', 'Man City', 'MID', 7.9, 5.2, 0, 0, VS_IPSWICH, { multiplier: 0, subbed_out: true }),
    player('Milan van Ewijk', 'Coventry City', 'DEF', 3.9, 2.07, 1, 90, { team: 'Spurs', home: false }, { multiplier: 0 }),
    player('Jonah Kusi-Asare', 'Fulham', 'FWD', 4.4, 0.04, 0, 0, { team: 'Hull City', home: true }, { multiplier: 0 }),
  ],
  auto_subs: [{ out: 'Rayan Cherki', in: 'Jacob Greaves' }],
};

/*
 * ML MANAGER SAMPLES (made-up players, real clubs) for the /design page: the team planned for GW10
 * (one transfer, no hit) a day or two before its deadline, and the same team scored in GW12 after a
 * Triple Captain, a 4-point hit and an automatic substitution. Prices are in tenths of £1m.
 */

type Opp = { team: string; home: boolean };

const H = (team: string): Opp => ({ team, home: true });
const A = (team: string): Opp => ({ team, home: false });

const squadPlayer = (
  element: number,
  name: string,
  web_name: string,
  team: string,
  position: MlManagerSquadPlayer['position'],
  prices: [purchase: number, selling: number, now: number],
  predicted: number,
  slot: 'XI' | 1 | 2 | 3 | 4,
  opponents: Opp[],
  extra: Partial<MlManagerSquadPlayer> = {},
): MlManagerSquadPlayer => ({
  element,
  name,
  web_name,
  team,
  position,
  purchase_price: prices[0],
  selling_price: prices[1],
  now_cost: prices[2],
  predicted_points: predicted,
  role: slot === 'XI' ? 'starter' : 'bench',
  bench_order: slot === 'XI' ? null : slot,
  captain: false,
  vice_captain: false,
  photo: null,
  opponents,
  ...extra,
});

const CHIP_WINDOWS: Array<[MlManagerChip, string, [number, number]]> = [
  ['wildcard', 'Wildcard', [2, 19]],
  ['freehit', 'Free Hit', [2, 19]],
  ['bboost', 'Bench Boost', [1, 19]],
  ['3xc', 'Triple Captain', [1, 19]],
];

/** Every chip for both halves; `used` maps a first-half chip to the gameweek it was played. */
const chipStates = (used: Partial<Record<MlManagerChip, number>>): MlManagerChipState[] => [
  ...CHIP_WINDOWS.map(([chip, label, window]): MlManagerChipState => ({
    chip,
    label,
    window,
    used_gameweek: used[chip] ?? null,
    available: used[chip] == null,
  })),
  ...CHIP_WINDOWS.map(([chip, label]): MlManagerChipState => ({ chip, label, window: [20, 38], used_gameweek: null, available: false })),
];

const RULES = { budget: 1000, max_free_transfers: 5, hit_cost: 4 };
const LEAGUE_SIZE = 182400;
const league = (rank: number) => ({
  id: 284,
  name: 'Gameweek 9',
  rank,
  size: LEAGUE_SIZE,
  percentile: Math.round((1 - rank / LEAGUE_SIZE) * 1000) / 10,
});

const historyRow = (
  gameweek: number,
  points: number,
  average: number,
  highest: number,
  captain: string,
  transfers: number,
  total: number,
  value: [team: number, bank: number],
  rank: number,
  extra: Partial<MlManagerHistoryRow> = {},
): MlManagerHistoryRow => ({
  gameweek,
  points,
  hit_cost: 0,
  chip: null,
  captain,
  transfers,
  total,
  team_value: value[0],
  bank: value[1],
  gw_average: average,
  gw_highest: highest,
  league_rank: rank,
  league_size: LEAGUE_SIZE,
  percentile: league(rank).percentile,
  ...extra,
});

const GW9_ROW = historyRow(9, 58, 49, 121, 'Sample Striker', 0, 58, [993, 8], 41230);

// --- Planned: GW10, deadline a couple of days away ---------------------------------------------

// Fixed per page load, so the sample always reads as "before the deadline".
const PLANNED_DEADLINE = new Date(Date.now() + 2 * 86400000 + 5 * 3600000).toISOString();
const PLANNED_AT = new Date(Date.now() - 6 * 3600000).toISOString();

const GW10: MlManagerGameweek = {
  season: '2026-27',
  gameweek: 10,
  deadline: PLANNED_DEADLINE,
  status: 'planned',
  planned_at: PLANNED_AT,
  model: 'v4',
  chip: null,
  transfers: [
    {
      out: { element: 116, name: 'Injured Winger', web_name: 'Injured', position: 'MID', team: 'Brighton', selling_price: 74 },
      in: { element: 111, name: 'Transfer Target', web_name: 'Target', position: 'MID', team: 'Chelsea', price: 80 },
    },
  ],
  free_transfers: 1,
  transfers_made: 1,
  hits: 0,
  hit_cost: 0,
  bank_before: 8,
  bank_after: 2,
  squad: [
    squadPlayer(101, 'Placeholder Keeper', 'Keeper', 'Arsenal', 'GK', [56, 56, 56], 4.1, 'XI', [H('Everton')]),
    squadPlayer(103, 'Demo Defender', 'Demo', 'Arsenal', 'DEF', [60, 60, 61], 4.4, 'XI', [H('Everton')]),
    squadPlayer(104, 'Sample Fullback', 'Fullback', 'Liverpool', 'DEF', [62, 62, 62], 4.0, 'XI', [A('Ipswich')]),
    squadPlayer(105, 'Example Centreback', 'Centreback', 'Chelsea', 'DEF', [50, 50, 50], 3.6, 'XI', [H('Wolves')]),
    squadPlayer(108, 'Example Winger', 'Winger', 'Liverpool', 'MID', [105, 105, 105], 7.6, 'XI', [A('Ipswich')], { vice_captain: true }),
    squadPlayer(109, 'Sample Playmaker', 'Playmaker', 'Aston Villa', 'MID', [78, 78, 78], 5.4, 'XI', [H('Bournemouth')]),
    squadPlayer(110, 'Mock Midfielder', 'Midfielder', 'Man City', 'MID', [70, 70, 70], 5.1, 'XI', [H('Sunderland')]),
    squadPlayer(111, 'Transfer Target', 'Target', 'Chelsea', 'MID', [80, 80, 80], 5.6, 'XI', [H('Wolves')]),
    squadPlayer(113, 'Sample Striker', 'Striker', 'Man City', 'FWD', [140, 141, 142], 8.9, 'XI', [H('Sunderland')], { captain: true }),
    squadPlayer(114, 'Second Striker', 'Second', 'Brentford', 'FWD', [66, 66, 66], 4.6, 'XI', [A('Fulham')]),
    squadPlayer(115, 'Budget Forward', 'Budget', 'Everton', 'FWD', [55, 55, 55], 3.3, 'XI', [A('Arsenal')]),
    squadPlayer(102, 'Reserve Keeper', 'Reserve', 'Brentford', 'GK', [40, 40, 40], 2.0, 1, [A('Fulham')]),
    squadPlayer(112, 'Squad Runner', 'Runner', 'Fulham', 'MID', [48, 48, 48], 2.6, 2, [H('Brentford')]),
    squadPlayer(106, 'Mock Wingback', 'Wingback', 'Newcastle', 'DEF', [47, 47, 48], 3.1, 3, [A('Spurs')]),
    squadPlayer(107, 'Test Stopper', 'Stopper', 'Ipswich', 'DEF', [40, 40, 40], 1.8, 4, [H('Liverpool')]),
  ],
  projected_points: 65.5,
  horizon: [
    { gameweek: 10, projected: 65.5 },
    { gameweek: 11, projected: 58.2 },
    { gameweek: 12, projected: 61.0 },
    { gameweek: 13, projected: 57.4 },
  ],
  reason: 'Injured Winger (injured, 0% chance of playing) out for Transfer Target: +9.8 projected over the next 4 gameweeks, with the free transfer.',
  result: null,
};

export const SAMPLE_ML_MANAGER_PLANNED: MlManagerView = {
  season: '2026-27',
  start_gameweek: 9,
  model: 'v4',
  rules: RULES,
  current: GW10,
  chips: chipStates({}),
  history: [GW9_ROW],
  totals: { points: 58, gameweeks: 1, vs_average: 9, league: league(41230) },
  next_deadline: PLANNED_DEADLINE,
};

// --- Scored: GW12, Triple Captain, a 4-point hit and an auto-sub ---------------------------------

const scored = (element: number, points: number, minutes: number, multiplier = 1): MlManagerResultPlayer => ({
  element,
  points,
  minutes,
  multiplier,
  counted: multiplier > 0,
});

const GW12: MlManagerGameweek = {
  season: '2026-27',
  gameweek: 12,
  deadline: '2026-11-21T11:00:00Z',
  status: 'scored',
  planned_at: '2026-11-21T06:00:00Z',
  model: 'v4',
  chip: '3xc',
  transfers: [
    {
      out: { element: 115, name: 'Budget Forward', web_name: 'Budget', position: 'FWD', team: 'Everton', selling_price: 54 },
      in: { element: 119, name: 'Form Forward', web_name: 'Form', position: 'FWD', team: 'Bournemouth', price: 58 },
    },
    {
      out: { element: 106, name: 'Mock Wingback', web_name: 'Wingback', position: 'DEF', team: 'Newcastle', selling_price: 47 },
      in: { element: 117, name: 'Attacking Fullback', web_name: 'Attacker', position: 'DEF', team: 'Spurs', price: 45 },
    },
  ],
  free_transfers: 1,
  transfers_made: 2,
  hits: 1,
  hit_cost: 4,
  bank_before: 5,
  bank_after: 3,
  squad: [
    squadPlayer(101, 'Placeholder Keeper', 'Keeper', 'Arsenal', 'GK', [56, 56, 56], 4.3, 'XI', [A('Leeds')]),
    squadPlayer(103, 'Demo Defender', 'Demo', 'Arsenal', 'DEF', [60, 60, 61], 3.9, 'XI', [A('Leeds')]),
    squadPlayer(104, 'Sample Fullback', 'Fullback', 'Liverpool', 'DEF', [62, 62, 63], 4.6, 'XI', [H('Aston Villa')]),
    squadPlayer(105, 'Example Centreback', 'Centreback', 'Chelsea', 'DEF', [50, 50, 50], 3.4, 'XI', [A('Newcastle')]),
    squadPlayer(117, 'Attacking Fullback', 'Attacker', 'Spurs', 'DEF', [45, 45, 45], 4.1, 'XI', [H('Bournemouth')]),
    squadPlayer(108, 'Example Winger', 'Winger', 'Liverpool', 'MID', [105, 105, 106], 6.8, 'XI', [H('Aston Villa')], { vice_captain: true }),
    squadPlayer(109, 'Sample Playmaker', 'Playmaker', 'Aston Villa', 'MID', [78, 78, 78], 5.0, 'XI', [A('Liverpool')]),
    squadPlayer(111, 'Transfer Target', 'Target', 'Chelsea', 'MID', [80, 80, 81], 5.2, 'XI', [A('Newcastle')]),
    squadPlayer(110, 'Mock Midfielder', 'Midfielder', 'Man City', 'MID', [70, 69, 69], 4.9, 'XI', [H('Ipswich')]),
    squadPlayer(113, 'Sample Striker', 'Striker', 'Man City', 'FWD', [140, 142, 144], 9.6, 'XI', [H('Ipswich')], { captain: true }),
    squadPlayer(119, 'Form Forward', 'Form', 'Bournemouth', 'FWD', [58, 58, 58], 5.4, 'XI', [A('Spurs')]),
    squadPlayer(102, 'Reserve Keeper', 'Reserve', 'Brentford', 'GK', [40, 40, 40], 1.9, 1, [H('Wolves')]),
    squadPlayer(114, 'Second Striker', 'Second', 'Brentford', 'FWD', [66, 66, 66], 4.2, 2, [H('Wolves')]),
    squadPlayer(118, 'Cheap Midfielder', 'Cheap', 'Wolves', 'MID', [45, 45, 45], 2.4, 3, [A('Brentford')]),
    squadPlayer(107, 'Test Stopper', 'Stopper', 'Ipswich', 'DEF', [40, 39, 39], 1.5, 4, [A('Man City')]),
  ],
  projected_points: 72.4,
  horizon: [
    { gameweek: 12, projected: 72.4 },
    { gameweek: 13, projected: 57.9 },
    { gameweek: 14, projected: 60.3 },
    { gameweek: 15, projected: 58.8 },
  ],
  reason:
    'Triple Captain on Sample Striker (9.6 projected at home); Budget Forward and Mock Wingback out for Form Forward and Attacking Fullback: +7.1 over the next 4 gameweeks, worth the 4-point hit.',
  result: {
    points: 82,
    gross: 86,
    hit_cost: 4,
    captain: 'Sample Striker',
    auto_subs: [['Mock Midfielder', 'Second Striker']],
    players: [
      scored(101, 6, 90),
      scored(103, 2, 90),
      scored(104, 8, 90),
      scored(105, 1, 90),
      scored(117, 6, 90),
      scored(108, 5, 90),
      scored(109, 3, 78),
      scored(111, 2, 90),
      scored(110, 0, 0, 0),
      scored(113, 13, 90, 3),
      scored(119, 9, 90),
      scored(102, 0, 0, 0),
      scored(114, 5, 72),
      scored(118, 2, 65, 0),
      scored(107, 1, 90, 0),
    ],
    gw_average: 52,
    gw_highest: 134,
    total: 255,
    league: league(11920),
  },
};

export const SAMPLE_ML_MANAGER_SCORED: MlManagerView = {
  season: '2026-27',
  start_gameweek: 9,
  model: 'v4',
  rules: RULES,
  current: GW12,
  chips: chipStates({ '3xc': 12 }),
  history: [
    GW9_ROW,
    historyRow(10, 71, 55, 128, 'Sample Striker', 1, 129, [1001, 2], 23410),
    historyRow(11, 44, 47, 112, 'Example Winger', 1, 173, [1002, 5], 26870),
    historyRow(12, 82, 52, 134, 'Sample Striker', 2, 255, [1001, 3], 11920, { hit_cost: 4, chip: '3xc' }),
  ],
  totals: { points: 255, gameweeks: 4, vs_average: 52, league: league(11920) },
  next_deadline: '2026-11-28T13:30:00Z',
};
