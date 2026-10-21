import type { ModelReviewPlayer, ModelReviewResponse } from '../../services/api';

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
