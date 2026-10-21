import type { PositionCode } from '../../services/api';

/** Pitch rows of a starting XI, goalkeeper first, with each row's accessible name. */
export const PITCH_LINES: { position: PositionCode; label: string }[] = [
  { position: 'GK', label: 'Goalkeeper' },
  { position: 'DEF', label: 'Defenders' },
  { position: 'MID', label: 'Midfielders' },
  { position: 'FWD', label: 'Forwards' },
];
