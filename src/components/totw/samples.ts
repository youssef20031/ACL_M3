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
 * ML MANAGER SAMPLES for the /design page: the model's real GW9 2026-27 squad (players, FPL short
 * names, photos and prices from that pick) with real GW10 and GW12 fixtures, planned for GW10 (one
 * transfer, no hit) and scored in GW12 after a Triple Captain, a 4-point hit and an automatic
 * substitution. The points, ranks, later price changes and transfers are made up. Prices are in
 * tenths of £1m.
 */

type Opp = { team: string; home: boolean };

const H = (team: string): Opp => ({ team, home: true });
const A = (team: string): Opp => ({ team, home: false });
const photo = (code: number) => `https://resources.premierleague.com/premierleague25/photos/players/110x140/${code}.png`;

/** element -> [full name, FPL short name, club, position, photo code (null: FPL has no photo)] */
const PEOPLE: Record<number, [string, string, string, MlManagerSquadPlayer['position'], number | null]> = {
  496: ['Antonín Kinský', 'Kinsky', 'Spurs', 'GK', 485055],
  572: ['Konstantinos Tzolakis', 'Tzolakis', 'Hull City', 'GK', 473284],
  229: ['James Tarkowski', 'Tarkowski', 'Everton', 'DEF', 17761],
  566: ['António João Pereira de Albuquerque Tavares da Silva', 'Silva', 'Bournemouth', 'DEF', 483067],
  257: ['Calvin Bassey', 'Bassey', 'Fulham', 'DEF', 232892],
  277: ['John Egan', 'Egan', 'Hull City', 'DEF', 108416],
  32: ['Matty Cash', 'Cash', 'Aston Villa', 'DEF', 199796],
  258: ['Timothy Castagne', 'Castagne', 'Fulham', 'DEF', 166477],
  304: ["Dara O'Shea", "O'Shea", 'Ipswich', 'DEF', 216616],
  68: ['Marcus Tavernier', 'Tavernier', 'Bournemouth', 'MID', 201658],
  154: ['Cole Palmer', 'Palmer', 'Chelsea', 'MID', 244851],
  427: ['Bryan Mbeumo', 'Mbeumo', 'Man Utd', 'MID', 446008],
  286: ['Mohamed Belloumi', 'Belloumi', 'Hull City', 'MID', 540324],
  236: ['Kiernan Dewsbury-Hall', 'Dewsbury-Hall', 'Everton', 'MID', 215413],
  129: ['Yasin Ayari', 'Ayari', 'Brighton', 'MID', 509416],
  411: ['Erling Haaland', 'Haaland', 'Man City', 'FWD', 223094],
  106: ['Igor Thiago Nascimento Rodrigues', 'Thiago', 'Brentford', 'FWD', 502500],
  166: ['Nicolas Jackson', 'N.Jackson', 'Aston Villa', 'FWD', null],
  401: ['Omar Marmoush', 'Marmoush', 'Spurs', 'FWD', 438234],
};

const ref = (element: number) => {
  const [name, web_name, team, position, code] = PEOPLE[element];
  return { element, name, web_name, team, position, photo: code == null ? null : photo(code) };
};

const squadPlayer = (
  element: number,
  prices: [purchase: number, selling: number, now: number],
  predicted: number,
  slot: 'XI' | 1 | 2 | 3 | 4,
  opponents: Opp[],
  extra: Partial<MlManagerSquadPlayer> = {},
): MlManagerSquadPlayer => ({
  ...ref(element),
  purchase_price: prices[0],
  selling_price: prices[1],
  now_cost: prices[2],
  predicted_points: predicted,
  role: slot === 'XI' ? 'starter' : 'bench',
  bench_order: slot === 'XI' ? null : slot,
  captain: false,
  vice_captain: false,
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

const GW9_ROW = historyRow(9, 58, 49, 121, 'Erling Haaland', 0, 58, [995, 5], 41230);

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
  transfers: [{ out: { ...ref(166), selling_price: 68 }, in: { ...ref(401), price: 70 } }],
  free_transfers: 1,
  transfers_made: 1,
  hits: 0,
  hit_cost: 0,
  bank_before: 5,
  bank_after: 3,
  squad: [
    squadPlayer(496, [44, 44, 44], 3.9, 'XI', [A('Leeds')]),
    squadPlayer(229, [62, 62, 62], 4.3, 'XI', [H('Coventry City')]),
    squadPlayer(566, [50, 50, 50], 3.8, 'XI', [A('Ipswich')]),
    squadPlayer(257, [45, 45, 45], 3.6, 'XI', [H('Newcastle')]),
    squadPlayer(68, [64, 64, 64], 4.6, 'XI', [A('Ipswich')], { vice_captain: true }),
    squadPlayer(154, [98, 98, 99], 6.1, 'XI', [A('Sunderland')]),
    squadPlayer(427, [79, 79, 79], 4.9, 'XI', [H('Aston Villa')]),
    squadPlayer(286, [52, 52, 52], 3.7, 'XI', [A('Arsenal')]),
    squadPlayer(411, [158, 159, 160], 7.4, 'XI', [A("Nott'm Forest")], { captain: true }),
    squadPlayer(106, [78, 78, 78], 4.4, 'XI', [A('Brighton')]),
    squadPlayer(401, [70, 70, 70], 4.1, 'XI', [A('Leeds')]),
    squadPlayer(572, [47, 47, 47], 2.1, 1, [A('Arsenal')]),
    squadPlayer(236, [66, 66, 66], 3.9, 2, [H('Coventry City')]),
    squadPlayer(277, [41, 41, 41], 2.2, 3, [A('Arsenal')]),
    squadPlayer(32, [43, 43, 43], 2.0, 4, [A('Man Utd')]),
  ],
  projected_points: 57.6,
  horizon: [
    { gameweek: 10, projected: 57.6 },
    { gameweek: 11, projected: 54.2 },
    { gameweek: 12, projected: 56.0 },
    { gameweek: 13, projected: 53.4 },
  ],
  reason: 'N.Jackson (injured, 0% chance of playing) out for Marmoush: +9.8 projected over the next 4 gameweeks, with the free transfer.',
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

const scored = (
  element: number,
  points: number,
  minutes: number,
  multiplier = 1,
  extra: Partial<MlManagerResultPlayer> = {},
): MlManagerResultPlayer => ({
  element,
  name: PEOPLE[element][0],
  points,
  minutes,
  multiplier,
  counted: multiplier > 0,
  ...extra,
});

const GW12: MlManagerGameweek = {
  season: '2026-27',
  gameweek: 12,
  deadline: '2026-12-05T11:00:00Z',
  status: 'scored',
  planned_at: '2026-12-05T06:00:00Z',
  model: 'v4',
  chip: '3xc',
  transfers: [
    { out: { ...ref(277), selling_price: 41 }, in: { ...ref(304), price: 40 } },
    { out: { ...ref(286), selling_price: 52 }, in: { ...ref(129), price: 54 } },
  ],
  free_transfers: 1,
  transfers_made: 2,
  hits: 1,
  hit_cost: 4,
  bank_before: 1,
  bank_after: 0,
  squad: [
    squadPlayer(496, [44, 44, 44], 4.0, 'XI', [A('Sunderland')]),
    squadPlayer(229, [62, 62, 62], 3.4, 'XI', [H('Liverpool')]),
    squadPlayer(566, [50, 50, 50], 3.9, 'XI', [A('Fulham')]),
    squadPlayer(257, [45, 45, 45], 3.7, 'XI', [H('Bournemouth')]),
    squadPlayer(68, [64, 64, 64], 4.8, 'XI', [A('Fulham')], { vice_captain: true }),
    squadPlayer(154, [98, 98, 99], 6.3, 'XI', [A("Nott'm Forest")]),
    squadPlayer(427, [79, 79, 79], 4.6, 'XI', [H('Brentford')]),
    squadPlayer(129, [54, 54, 54], 4.4, 'XI', [H('Newcastle')]),
    squadPlayer(411, [158, 159, 160], 8.1, 'XI', [A('Arsenal')], { captain: true }),
    squadPlayer(106, [78, 78, 78], 4.5, 'XI', [A('Man Utd')]),
    squadPlayer(401, [70, 70, 70], 4.2, 'XI', [A('Sunderland')]),
    squadPlayer(572, [47, 47, 47], 2.0, 1, [A('Crystal Palace')]),
    squadPlayer(236, [66, 66, 66], 3.8, 2, [H('Liverpool')]),
    squadPlayer(258, [45, 45, 45], 3.2, 3, [H('Bournemouth')]),
    squadPlayer(304, [40, 40, 40], 2.6, 4, [H('Aston Villa')]),
  ],
  projected_points: 70.6,
  horizon: [
    { gameweek: 12, projected: 70.6 },
    { gameweek: 13, projected: 55.9 },
    { gameweek: 14, projected: 58.3 },
    { gameweek: 15, projected: 56.8 },
  ],
  reason:
    "Triple Captain on Haaland (8.1 projected); Egan and Belloumi out for O'Shea and Ayari: +7.1 over the next 4 gameweeks, worth the 4-point hit.",
  result: {
    points: 82,
    gross: 86,
    hit_cost: 4,
    captain: 'Erling Haaland',
    auto_subs: [['Bryan Mbeumo', 'Kiernan Dewsbury-Hall']],
    players: [
      scored(496, 6, 90),
      scored(229, 2, 90),
      scored(566, 8, 90),
      scored(257, 1, 90),
      scored(68, 5, 90),
      scored(154, 9, 90),
      scored(427, 0, 0, 0, { subbed_out: true }),
      scored(129, 3, 78),
      scored(411, 13, 90, 3),
      scored(106, 6, 90),
      scored(401, 2, 72),
      scored(572, 0, 0, 0),
      scored(236, 5, 85, 1, { subbed_in: true }),
      scored(258, 2, 90, 0),
      scored(304, 1, 90, 0),
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
    historyRow(10, 71, 55, 128, 'Erling Haaland', 1, 129, [997, 3], 23410),
    historyRow(11, 44, 47, 112, 'Cole Palmer', 1, 173, [1000, 1], 26870),
    historyRow(12, 82, 52, 134, 'Erling Haaland', 2, 255, [1003, 0], 11920, { hit_cost: 4, chip: '3xc' }),
  ],
  totals: { points: 255, gameweeks: 4, vs_average: 52, league: league(11920) },
  next_deadline: '2026-12-12T13:30:00Z',
};
