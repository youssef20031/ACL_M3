import { useEffect, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  ArrowLeftRight,
  Loader2,
  AlertCircle,
  Search,
  Target,
  Trophy,
  Shield,
  Flame,
  DollarSign,
  Gauge,
  RotateCcw,
} from 'lucide-react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from 'recharts';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import { cn } from '../utils/cn';

const SEASONS = ['', '2020-21', '2021-22', '2022-23'];

const EXAMPLE_PAIRS = [
  { p1: 'Mohamed Salah', p2: 'Son Heung-min' },
  { p1: 'Erling Haaland', p2: 'Harry Kane' },
  { p1: 'Kevin De Bruyne', p2: 'Bruno Fernandes' },
];

type PlayerStats = Record<string, any>;
type Suggestion = { name: string; position?: string; season?: string; avatar?: string };
type CompareRouteState = {
  player1?: string;
  player2?: string;
  season?: string;
};

function formatMetricValue(value: number, options?: { prefix?: string; suffix?: string; decimals?: number }) {
  const decimals = options?.decimals;
  const body = decimals != null ? value.toFixed(decimals) : `${Math.round(value)}`;
  return `${options?.prefix ?? ''}${body}${options?.suffix ?? ''}`;
}

export function PlayerComparison() {
  const { neo4jConnected, theme } = useAppStore();
  const isDark = theme === 'dark';
  const location = useLocation();
  const navigate = useNavigate();

  const [player1, setPlayer1] = useState('');
  const [player2, setPlayer2] = useState('');
  const [p1Suggestions, setP1Suggestions] = useState<Suggestion[]>([]);
  const [p2Suggestions, setP2Suggestions] = useState<Suggestion[]>([]);
  const [p1Index, setP1Index] = useState(-1);
  const [p2Index, setP2Index] = useState(-1);
  const p1Timer = useRef<number | null>(null);
  const p2Timer = useRef<number | null>(null);
  const p1SuppressSearch = useRef(false);
  const p2SuppressSearch = useRef(false);
  const p1AutocompleteActive = useRef(false);
  const p2AutocompleteActive = useRef(false);
  const [avatarByName, setAvatarByName] = useState<Record<string, string | null>>({});
  const avatarHydrationVersion = useRef(0);
  const p1InputRef = useRef<HTMLInputElement | null>(null);
  const p2InputRef = useRef<HTMLInputElement | null>(null);
  const p1ListId = 'p1-suggestions-list';
  const p2ListId = 'p2-suggestions-list';
  const [season, setSeason] = useState('');
  const [comparisonData, setComparisonData] = useState<PlayerStats[]>([]);
  const restoreComparisonRef = useRef(false);

  const compareMutation = useMutation({
    mutationFn: ({ p1, p2, s }: { p1: string; p2: string; s?: string }) =>
      apiService.comparePlayers(p1, p2, s || undefined),
    onSuccess: (data) => {
      setComparisonData(data.comparison);
    },
  });

  const handleCompare = (e: React.FormEvent) => {
    e.preventDefault();
    if (!player1.trim() || !player2.trim() || !neo4jConnected) return;
    compareMutation.mutate({ p1: player1.trim(), p2: player2.trim(), s: season });
  };

  const handleExample = (p1: string, p2: string) => {
    p1AutocompleteActive.current = false;
    p2AutocompleteActive.current = false;
    setPlayer1(p1);
    setPlayer2(p2);
    setP1Suggestions([]);
    setP2Suggestions([]);
    compareMutation.mutate({ p1, p2, s: season });
  };

  const runSearch = async (q: string, setSuggestions: (s: Suggestion[]) => void) => {
    if (!q || q.trim().length < 2) {
      setSuggestions([]);
      return;
    }

    try {
      const res = await apiService.searchPlayers(q.trim(), { limit: 10, includeAvatars: false });
      const players = (res.players || [])
        .map((p: any) => ({
          name: p.player_name || p.name || '',
          position: p.position || p.pos || undefined,
          season: p.season || undefined,
          avatar: p.avatar || p.image || p.photo || undefined,
        }))
        .filter((x: Suggestion) => x.name);
      setSuggestions(players.slice(0, 10));
    } catch {
      setSuggestions([]);
    }
  };

  useEffect(() => {
    const visibleNames = Array.from(new Set([...p1Suggestions, ...p2Suggestions].map((s) => s.name).filter(Boolean)));
    const namesToHydrate = visibleNames.filter((name) => avatarByName[name] === undefined).slice(0, 4);

    if (namesToHydrate.length === 0) {
      return;
    }

    const version = ++avatarHydrationVersion.current;

    void Promise.allSettled(
      namesToHydrate.map(async (name) => {
        const result = await apiService.searchImage(name);
        return { name, avatar: result.image_url ?? null };
      })
    ).then((settled) => {
      if (version !== avatarHydrationVersion.current) return;

      setAvatarByName((current) => {
        const next = { ...current };
        for (const item of settled) {
          if (item.status === 'fulfilled') {
            next[item.value.name] = item.value.avatar;
          }
        }
        return next;
      });
    });
  }, [avatarByName, p1Suggestions, p2Suggestions]);

  useEffect(() => {
    if (p1Timer.current) window.clearTimeout(p1Timer.current);
    p1Timer.current = window.setTimeout(() => {
      if (!p1AutocompleteActive.current) {
        setP1Suggestions([]);
        setP1Index(-1);
        return;
      }
      if (p1SuppressSearch.current) {
        p1SuppressSearch.current = false;
        setP1Index(-1);
        return;
      }
      runSearch(player1, setP1Suggestions);
      setP1Index(-1);
    }, 150) as unknown as number;

    return () => {
      if (p1Timer.current) window.clearTimeout(p1Timer.current);
    };
  }, [player1]);

  useEffect(() => {
    if (p2Timer.current) window.clearTimeout(p2Timer.current);
    p2Timer.current = window.setTimeout(() => {
      if (!p2AutocompleteActive.current) {
        setP2Suggestions([]);
        setP2Index(-1);
        return;
      }
      if (p2SuppressSearch.current) {
        p2SuppressSearch.current = false;
        setP2Index(-1);
        return;
      }
      runSearch(player2, setP2Suggestions);
      setP2Index(-1);
    }, 150) as unknown as number;

    return () => {
      if (p2Timer.current) window.clearTimeout(p2Timer.current);
    };
  }, [player2]);

  useEffect(() => {
    const q = player1.trim();
    if (q && !q.includes(' ') && p1Suggestions.length === 1) {
      p1SuppressSearch.current = true;
      setPlayer1(p1Suggestions[0].name);
      setP1Suggestions([]);
    }
  }, [p1Suggestions, player1]);

  useEffect(() => {
    const q = player2.trim();
    if (q && !q.includes(' ') && p2Suggestions.length === 1) {
      p2SuppressSearch.current = true;
      setPlayer2(p2Suggestions[0].name);
      setP2Suggestions([]);
    }
  }, [p2Suggestions, player2]);

  useEffect(() => {
    if (player1.trim() && player2.trim() && neo4jConnected) {
      compareMutation.mutate({ p1: player1.trim(), p2: player2.trim(), s: season });
    }
  }, [season]);

  useEffect(() => {
    const state = location.state as CompareRouteState | null;
    if (!state?.player1 || !state?.player2) {
      return;
    }

    restoreComparisonRef.current = true;
    p1AutocompleteActive.current = false;
    p2AutocompleteActive.current = false;
    setPlayer1(state.player1);
    setPlayer2(state.player2);
    setSeason(state.season ?? '');
    setP1Suggestions([]);
    setP2Suggestions([]);

    if (neo4jConnected) {
      compareMutation.mutate({ p1: state.player1, p2: state.player2, s: state.season });
    }
  }, [location.state, neo4jConnected]);

  const onP1KeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (p1Suggestions.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setP1Index((i) => Math.min(i + 1, p1Suggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setP1Index((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      if (p1Index >= 0 && p1Index < p1Suggestions.length) {
        e.preventDefault();
        p1SuppressSearch.current = true;
        setPlayer1(p1Suggestions[p1Index].name);
        setP1Suggestions([]);
        setP1Index(-1);
        p1InputRef.current?.focus();
      }
    } else if (e.key === 'Escape') {
      setP1Suggestions([]);
      setP1Index(-1);
    }
  };

  const onP2KeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (p2Suggestions.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setP2Index((i) => Math.min(i + 1, p2Suggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setP2Index((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      if (p2Index >= 0 && p2Index < p2Suggestions.length) {
        e.preventDefault();
        p2SuppressSearch.current = true;
        setPlayer2(p2Suggestions[p2Index].name);
        setP2Suggestions([]);
        setP2Index(-1);
        p2InputRef.current?.focus();
      }
    } else if (e.key === 'Escape') {
      setP2Suggestions([]);
      setP2Index(-1);
    }
  };

  const p1Data = comparisonData[0];
  const p2Data = comparisonData[1];

  const pageClass = cn(
    'min-h-full transition-colors',
    isDark
      ? 'bg-[radial-gradient(circle_at_top,_rgba(139,92,246,0.16),_transparent_32%),linear-gradient(180deg,_#020617_0%,_#0f172a_100%)] text-slate-100'
      : 'bg-slate-50 text-slate-900'
  );
  const panelClass = isDark ? 'border-slate-800 bg-slate-900/80 text-slate-100 shadow-[0_24px_80px_rgba(2,6,23,0.45)]' : 'border-slate-200 bg-white text-slate-900 shadow-sm';
  const softPanelClass = isDark ? 'border-slate-800 bg-slate-900/65 text-slate-100' : 'border-slate-200 bg-white text-slate-900';
  const inputClass = cn(
    'w-full rounded-xl border px-3 py-3 text-sm outline-none transition focus:ring-2',
    isDark
      ? 'border-slate-700 bg-slate-950/70 text-slate-100 placeholder:text-slate-500 focus:border-violet-500 focus:ring-violet-500/30'
      : 'border-slate-300 bg-white text-slate-900 placeholder:text-slate-400 focus:border-purple-500 focus:ring-purple-500/20'
  );
  const selectClass = cn(
    'rounded-xl border px-3 py-3 text-sm outline-none transition focus:ring-2',
    isDark
      ? 'border-slate-700 bg-slate-950/70 text-slate-100 focus:border-violet-500 focus:ring-violet-500/30'
      : 'border-slate-300 bg-white text-slate-900 focus:border-purple-500 focus:ring-purple-500/20'
  );
  const labelClass = cn('mb-2 block text-sm font-medium', isDark ? 'text-slate-200' : 'text-slate-700');
  const mutedText = isDark ? 'text-slate-400' : 'text-slate-600';

  return (
    <div className={pageClass}>
      <div className={cn('border-b backdrop-blur-xl', isDark ? 'border-white/10 bg-slate-950/70' : 'border-slate-200 bg-white/85')}>
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-4 py-5 sm:px-6 lg:px-8">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className={cn('text-xs font-semibold uppercase tracking-[0.28em]', isDark ? 'text-violet-300' : 'text-purple-600')}>
                Head-to-head analysis
              </p>
              <h2 className="mt-2 text-2xl font-black tracking-tight sm:text-3xl">Enhanced Player Comparison V2</h2>
              <p className={cn('mt-2 max-w-2xl text-sm sm:text-base', mutedText)}>
                Compare two FPL players with a dramatic match-up view, then dive into the stat table and radar chart for the fine detail.
              </p>
            </div>

            <div className={cn('rounded-2xl border px-4 py-3 text-sm', isDark ? 'border-white/10 bg-white/5' : 'border-slate-200 bg-slate-50')}>
              <div className={cn('flex items-center gap-2 font-medium', neo4jConnected ? (isDark ? 'text-emerald-300' : 'text-emerald-700') : 'text-amber-600')}>
                <span className={cn('h-2.5 w-2.5 rounded-full', neo4jConnected ? 'bg-emerald-500' : 'bg-amber-500')} />
                {neo4jConnected ? 'Neo4j connected' : 'Neo4j offline'}
              </div>
              <p className={cn('mt-1 text-xs', mutedText)}>Use Settings to connect if the compare view is empty.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        {!neo4jConnected && (
          <div className={cn('flex items-start gap-3 rounded-2xl border p-4', isDark ? 'border-amber-400/20 bg-amber-500/10' : 'border-amber-200 bg-amber-50')}>
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-yellow-600" />
            <p className={cn('text-sm', isDark ? 'text-amber-200' : 'text-yellow-800')}>Connect to Neo4j in Settings to compare players.</p>
          </div>
        )}

        <form id="comparison-form" onSubmit={handleCompare} className={cn('space-y-5 rounded-3xl border p-4 sm:p-6', panelClass)} autoComplete="off">
          <input aria-hidden="true" tabIndex={-1} style={{ position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden' }} type="text" name="__fake_user" autoComplete="username" />
          <input aria-hidden="true" tabIndex={-1} style={{ position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden' }} type="password" name="__fake_pass" autoComplete="current-password" />

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="relative">
              <label className={labelClass}>Player 1</label>
              <input
                ref={(el) => (p1InputRef.current = el)}
                id="p1-input"
                name="player1_search"
                type="text"
                value={player1}
                onChange={(e) => {
                  p1AutocompleteActive.current = true;
                  setPlayer1(e.target.value);
                }}
                onKeyDown={onP1KeyDown}
                placeholder="e.g. Mohamed Salah"
                disabled={!neo4jConnected}
                aria-autocomplete="list"
                aria-controls={p1ListId}
                aria-expanded={p1Suggestions.length > 0}
                aria-activedescendant={p1Index >= 0 ? `p1-suggestion-${p1Index}` : undefined}
                role="combobox"
                autoComplete="off"
                className={inputClass}
              />
              {p1Suggestions.length > 0 && (
                <ul id={p1ListId} role="listbox" className={cn('absolute z-20 mt-2 w-full max-h-56 overflow-auto rounded-2xl border shadow-xl', isDark ? 'border-slate-700 bg-slate-950' : 'border-slate-200 bg-white')}>
                  {p1Suggestions.map((s, i) => (
                    <li
                      id={`p1-suggestion-${i}`}
                      role="option"
                      aria-selected={i === p1Index}
                      key={s.name + i}
                      onClick={() => {
                        p1AutocompleteActive.current = false;
                        p1SuppressSearch.current = true;
                        setPlayer1(s.name);
                        setP1Suggestions([]);
                        p1InputRef.current?.focus();
                      }}
                      className={cn(
                        'flex cursor-pointer items-center justify-between gap-2 px-3 py-2 text-sm transition-colors',
                        i === p1Index ? (isDark ? 'bg-violet-500/20' : 'bg-purple-50') : isDark ? 'hover:bg-white/5' : 'hover:bg-purple-50'
                      )}
                    >
                      <div className="flex items-center gap-3">
                        {(() => {
                          const avatarUrl = s.avatar ?? avatarByName[s.name];
                          return avatarUrl ? <img src={avatarUrl} alt={`${s.name} avatar`} className="h-6 w-6 rounded-full" /> : null;
                        })()}
                        <div>
                          <div className="font-medium">{s.name}</div>
                          {(s.position || s.season) && <div className={cn('text-xs', mutedText)}>{[s.position, s.season].filter(Boolean).join(' · ')}</div>}
                        </div>
                      </div>
                      <div className="text-xs text-slate-400">{i === p1Index ? '↵' : ''}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="relative">
              <label className={labelClass}>Player 2</label>
              <input
                ref={(el) => (p2InputRef.current = el)}
                id="p2-input"
                name="player2_search"
                type="text"
                value={player2}
                onChange={(e) => {
                  p2AutocompleteActive.current = true;
                  setPlayer2(e.target.value);
                }}
                onKeyDown={onP2KeyDown}
                placeholder="e.g. Son Heung-min"
                disabled={!neo4jConnected}
                aria-autocomplete="list"
                aria-controls={p2ListId}
                aria-expanded={p2Suggestions.length > 0}
                aria-activedescendant={p2Index >= 0 ? `p2-suggestion-${p2Index}` : undefined}
                role="combobox"
                autoComplete="off"
                className={inputClass}
              />
              {p2Suggestions.length > 0 && (
                <ul id={p2ListId} role="listbox" className={cn('absolute z-20 mt-2 w-full max-h-56 overflow-auto rounded-2xl border shadow-xl', isDark ? 'border-slate-700 bg-slate-950' : 'border-slate-200 bg-white')}>
                  {p2Suggestions.map((s, i) => (
                    <li
                      id={`p2-suggestion-${i}`}
                      role="option"
                      aria-selected={i === p2Index}
                      key={s.name + i}
                      onClick={() => {
                        p2AutocompleteActive.current = false;
                        p2SuppressSearch.current = true;
                        setPlayer2(s.name);
                        setP2Suggestions([]);
                        p2InputRef.current?.focus();
                      }}
                      className={cn(
                        'flex cursor-pointer items-center justify-between gap-2 px-3 py-2 text-sm transition-colors',
                        i === p2Index ? (isDark ? 'bg-violet-500/20' : 'bg-purple-50') : isDark ? 'hover:bg-white/5' : 'hover:bg-purple-50'
                      )}
                    >
                      <div className="flex items-center gap-3">
                        {(() => {
                          const avatarUrl = s.avatar ?? avatarByName[s.name];
                          return avatarUrl ? <img src={avatarUrl} alt={`${s.name} avatar`} className="h-6 w-6 rounded-full" /> : null;
                        })()}
                        <div>
                          <div className="font-medium">{s.name}</div>
                          {(s.position || s.season) && <div className={cn('text-xs', mutedText)}>{[s.position, s.season].filter(Boolean).join(' · ')}</div>}
                        </div>
                      </div>
                      <div className="text-xs text-slate-400">{i === p2Index ? '↵' : ''}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:flex-wrap sm:items-end">
            <div>
              <label className={labelClass}>Season (optional)</label>
              <select value={season} onChange={(e) => setSeason(e.target.value)} disabled={!neo4jConnected} className={selectClass}>
                {SEASONS.map((s) => (
                  <option key={s} value={s}>
                    {s || 'All seasons'}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="submit"
              disabled={!neo4jConnected || compareMutation.isPending || !player1.trim() || !player2.trim()}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-purple-600 px-6 py-3 text-sm text-white transition-colors hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
            >
              {compareMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowLeftRight className="h-4 w-4" />}
              Compare
            </button>
          </div>
        </form>

        {comparisonData.length === 0 && neo4jConnected && (
          <div className={cn('rounded-3xl border p-4 sm:p-5', softPanelClass)}>
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Target className="h-4 w-4 text-violet-500" />
              Try these matchups
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {EXAMPLE_PAIRS.map((pair) => (
                <button
                  key={pair.p1}
                  onClick={() => handleExample(pair.p1, pair.p2)}
                  className={cn('rounded-full px-4 py-2 text-sm transition-colors', isDark ? 'bg-white/8 hover:bg-violet-500/20 hover:text-violet-200' : 'bg-slate-100 hover:bg-purple-100 hover:text-purple-700')}
                >
                  {pair.p1} vs {pair.p2}
                </button>
              ))}
            </div>
          </div>
        )}

        {compareMutation.isError && (
          <div className={cn('rounded-2xl border p-4 text-sm', isDark ? 'border-red-400/20 bg-red-500/10 text-red-200' : 'border-red-200 bg-red-50 text-red-700')}>
            {handleApiError(compareMutation.error)}
          </div>
        )}

        {compareMutation.isSuccess && comparisonData.length === 0 && (
          <div className={cn('rounded-3xl border px-6 py-14 text-center', softPanelClass)}>
            <Search className="mx-auto mb-3 h-12 w-12 opacity-30" />
            <p>No data found for these players.</p>
            <p className="mt-1 text-sm">Check the spelling or try different names.</p>
          </div>
        )}

        {p1Data && p2Data && (
          <ComparisonResults
            p1={p1Data}
            p2={p2Data}
            isDark={isDark}
            onOpenDetails={(player) =>
              navigate('/search', {
                state: {
                  player,
                  season: season || 'All seasons',
                  returnTo: 'history',
                  compare: {
                    player1,
                    player2,
                    season,
                  },
                },
              })
            }
            onChangePlayers={() => p1InputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
          />
        )}

        {comparisonData.length === 1 && (
          <div className={cn('rounded-2xl border p-4 text-sm', isDark ? 'border-amber-400/20 bg-amber-500/10 text-amber-200' : 'border-yellow-200 bg-yellow-50 text-yellow-800')}>
            Only found data for <strong>{comparisonData[0].player_name}</strong>. The second player may not exist or have no data for the selected season.
          </div>
        )}
      </div>
    </div>
  );
}

function ComparisonResults({
  p1,
  p2,
  isDark,
  onOpenDetails,
  onChangePlayers,
}: {
  p1: PlayerStats;
  p2: PlayerStats;
  isDark: boolean;
  onOpenDetails: (player: PlayerStats) => void;
  onChangePlayers: () => void;
}) {
  const p1Name = p1.player_name ?? 'Player 1';
  const p2Name = p2.player_name ?? 'Player 2';

  const heroMetrics = [
    { label: 'Goals Scored', p1: p1.goals ?? 0, p2: p2.goals ?? 0, icon: Trophy },
    { label: 'Assists', p1: p1.assists ?? 0, p2: p2.assists ?? 0, icon: Target },
    { label: 'Bonus Points', p1: p1.bonus ?? 0, p2: p2.bonus ?? 0, icon: Shield },
    { label: 'Form', p1: p1.avg_form ?? 0, p2: p2.avg_form ?? 0, icon: Flame, decimals: 2 },
    { label: 'Price', p1: p1.avg_value_millions ?? 0, p2: p2.avg_value_millions ?? 0, icon: DollarSign, prefix: '£', suffix: 'm', decimals: 1 },
    { label: 'ICT Index', p1: p1.avg_ict_index ?? 0, p2: p2.avg_ict_index ?? 0, icon: Gauge, decimals: 2 },
  ];

  const barStats = [
    { label: 'Points', p1: p1.total_points ?? 0, p2: p2.total_points ?? 0 },
    { label: 'Goals', p1: p1.goals ?? 0, p2: p2.goals ?? 0 },
    { label: 'Assists', p1: p1.assists ?? 0, p2: p2.assists ?? 0 },
    { label: 'Bonus', p1: p1.bonus ?? 0, p2: p2.bonus ?? 0 },
    { label: 'CS', p1: p1.clean_sheets ?? 0, p2: p2.clean_sheets ?? 0 },
  ];

  const radarMetrics = [
    { key: 'total_points', label: 'Points', max: 300 },
    { key: 'goals', label: 'Goals', max: 30 },
    { key: 'assists', label: 'Assists', max: 20 },
    { key: 'bonus', label: 'Bonus', max: 50 },
    { key: 'avg_ict_index', label: 'ICT', max: 10 },
    { key: 'games', label: 'Games', max: 38 },
  ];

  const radarData = radarMetrics.map(({ key, label, max }) => ({
    metric: label,
    [p1Name]: Math.round(((p1[key] ?? 0) / max) * 100),
    [p2Name]: Math.round(((p2[key] ?? 0) / max) * 100),
  }));

  const statRows = [
    { label: 'Position', p1v: p1.position, p2v: p2.position, compare: false },
    { label: 'Season', p1v: p1.season, p2v: p2.season, compare: false },
    { label: 'Total Points', p1v: p1.total_points, p2v: p2.total_points, compare: true },
    { label: 'Goals', p1v: p1.goals, p2v: p2.goals, compare: true },
    { label: 'Assists', p1v: p1.assists, p2v: p2.assists, compare: true },
    { label: 'Clean Sheets', p1v: p1.clean_sheets, p2v: p2.clean_sheets, compare: true },
    { label: 'Bonus', p1v: p1.bonus, p2v: p2.bonus, compare: true },
    { label: 'Minutes', p1v: p1.minutes?.toLocaleString(), p2v: p2.minutes?.toLocaleString(), compare: false },
    { label: 'Games', p1v: p1.games, p2v: p2.games, compare: true },
    { label: 'Form', p1v: p1.avg_form, p2v: p2.avg_form, compare: true },
    { label: 'Avg ICT', p1v: p1.avg_ict_index, p2v: p2.avg_ict_index, compare: true },
    { label: 'Value (£m)', p1v: p1.avg_value_millions, p2v: p2.avg_value_millions, compare: false },
  ].filter((r) => r.p1v != null || r.p2v != null);

  const panelClass = isDark ? 'border-slate-800 bg-slate-900/80 text-slate-100 shadow-[0_24px_80px_rgba(2,6,23,0.45)]' : 'border-slate-200 bg-white text-slate-900 shadow-sm';
  const statMuted = isDark ? 'text-slate-400' : 'text-slate-500';
  const statBorder = isDark ? 'border-slate-800' : 'border-slate-200';

  return (
    <div className="space-y-6">
      <div className="grid gap-5 xl:grid-cols-[minmax(0,0.95fr)_minmax(320px,1.15fr)_minmax(0,0.95fr)]">
        {[p1, p2].map((p, i) => {
          const name = p.player_name ?? `Player ${i + 1}`;
          const accent = i === 0 ? 'from-violet-600 to-indigo-700' : 'from-sky-500 to-cyan-600';

          return (
            <div key={i} className={cn('relative overflow-hidden rounded-[2rem] border', panelClass)}>
              <div className={cn('absolute inset-x-0 top-0 h-1.5 bg-gradient-to-r', accent)} />
              <div className="relative min-h-[420px]">
                {p.avatar ? (
                  <img src={p.avatar} alt={`${name} avatar`} className="absolute inset-0 h-full w-full object-cover" />
                ) : (
                  <div className={cn('absolute inset-0 bg-gradient-to-br', i === 0 ? 'from-violet-900 via-slate-900 to-slate-950' : 'from-sky-900 via-slate-900 to-slate-950')} />
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-slate-950/40 to-slate-950/10" />
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.08),_transparent_35%)]" />
                <div className="relative flex h-full flex-col justify-end p-6 text-white">
                  <div className="mb-4 flex items-center gap-2">
                    <span className="rounded-full bg-white/15 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] backdrop-blur">{p.position ?? 'Player'}</span>
                    {p.season && <span className="rounded-full bg-white/10 px-3 py-1 text-xs backdrop-blur">{p.season}</span>}
                  </div>
                  <h3 className="max-w-[12ch] text-4xl font-black uppercase leading-[0.9] tracking-tight sm:text-5xl">{name}</h3>
                  <p className="mt-2 text-sm text-white/75">{p.team_name ?? p.team ?? 'Premier League'}</p>
                  <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-2xl border border-white/10 bg-white/10 p-3 backdrop-blur">
                      <div className="text-xs uppercase tracking-wide text-white/60">Points</div>
                      <div className="mt-1 text-2xl font-bold">{p.total_points ?? 0}</div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-white/10 p-3 backdrop-blur">
                      <div className="text-xs uppercase tracking-wide text-white/60">Games</div>
                      <div className="mt-1 text-2xl font-bold">{p.games ?? 0}</div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onOpenDetails(p)}
                    className={cn(
                      'relative z-20 mt-5 inline-flex self-start items-center rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] shadow-lg transition',
                      isDark
                        ? 'border-white/15 bg-white/10 text-white hover:bg-white/20'
                        : 'border-white/20 bg-white/15 text-white hover:bg-white/25'
                    )}
                  >
                    View details
                  </button>
                </div>
              </div>
            </div>
          );
        })}

        <div className={cn('rounded-[2rem] border p-5 sm:p-6 xl:py-7', panelClass)}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className={cn('text-xs font-semibold uppercase tracking-[0.25em]', isDark ? 'text-violet-300' : 'text-purple-600')}>Matchup</p>
              <h4 className="mt-1 text-2xl font-black tracking-tight">Player comparison</h4>
            </div>
            <div className={cn('rounded-full border px-3 py-1 text-xs font-medium', isDark ? 'border-white/10 bg-white/5 text-slate-300' : 'border-slate-200 bg-slate-50 text-slate-500')}>
              {p1Name} vs {p2Name}
            </div>
          </div>

          <div className="mt-6 space-y-4">
            {heroMetrics.map((metric) => {
              const left = metric.p1 as number;
              const right = metric.p2 as number;
              const total = Math.max(left + right, 1);
              const leftPct = Math.max(8, Math.round((left / total) * 100));
              const rightPct = Math.max(8, 100 - leftPct);
              const Icon = metric.icon;
              const winnerLeft = left > right;
              const winnerRight = right > left;

              return (
                <div key={metric.label} className="space-y-2">
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <div className={cn('flex items-center gap-2 font-medium', winnerLeft ? (isDark ? 'text-violet-300' : 'text-purple-700') : isDark ? 'text-slate-100' : 'text-slate-700')}>
                      <Icon className="h-4 w-4 opacity-80" />
                      <span>{formatMetricValue(left, metric)}</span>
                    </div>
                    <div className={cn('text-center text-xs uppercase tracking-[0.28em]', statMuted)}>{metric.label}</div>
                    <div className={cn('flex items-center gap-2 font-medium', winnerRight ? (isDark ? 'text-sky-300' : 'text-cyan-700') : isDark ? 'text-slate-100' : 'text-slate-700')}>
                      <span>{formatMetricValue(right, metric)}</span>
                    </div>
                  </div>
                  <div className={cn('relative h-2 overflow-hidden rounded-full border', statBorder)}>
                    <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-violet-500 to-fuchsia-500" style={{ width: `${leftPct}%` }} />
                    <div className="absolute inset-y-0 right-0 bg-gradient-to-l from-sky-500 to-cyan-400" style={{ width: `${rightPct}%` }} />
                    <div className="absolute inset-0 flex items-center justify-center text-[10px] font-semibold uppercase tracking-[0.3em] text-white/80">VS</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <div className={cn('rounded-3xl border p-4 sm:p-5', panelClass)}>
          <div className="grid grid-cols-3 border-b px-4 py-2 text-xs font-semibold uppercase tracking-wide" style={{ borderColor: isDark ? 'rgba(148,163,184,0.12)' : 'rgba(226,232,240,1)' }}>
            <span className={isDark ? 'text-violet-300' : 'text-purple-600'}>{p1Name}</span>
            <span className="text-center">Stat</span>
            <span className="text-right text-pink-600">{p2Name}</span>
          </div>
          {statRows.map((row, i) => {
            const p1Wins = row.compare && row.p1v != null && row.p2v != null && Number(row.p1v) > Number(row.p2v);
            const p2Wins = row.compare && row.p1v != null && row.p2v != null && Number(row.p2v) > Number(row.p1v);

            return (
              <div
                key={i}
                className={cn('grid grid-cols-3 px-4 py-3 text-sm', i % 2 === 0 ? (isDark ? 'bg-white/0' : 'bg-white') : isDark ? 'bg-white/[0.03]' : 'bg-slate-50')}
              >
                <span className={cn('font-medium', p1Wins ? (isDark ? 'text-violet-300' : 'text-purple-700') : isDark ? 'text-slate-200' : 'text-slate-700')}>
                  {row.p1v ?? '—'}
                </span>
                <span className={cn('self-center text-center text-xs', statMuted)}>{row.label}</span>
                <span className={cn('text-right font-medium', p2Wins ? (isDark ? 'text-sky-300' : 'text-pink-600') : isDark ? 'text-slate-200' : 'text-slate-700')}>
                  {row.p2v ?? '—'}
                </span>
              </div>
            );
          })}
        </div>

        <div className={cn('rounded-3xl border p-4 sm:p-5', panelClass)}>
          <h4 className="mb-4 text-sm font-semibold">Side-by-Side Comparison</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barStats} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDark ? 'rgba(148,163,184,0.12)' : '#f0f0f0'} />
              <XAxis dataKey="label" tick={{ fontSize: 11, fill: isDark ? '#cbd5e1' : '#475569' }} />
              <YAxis tick={{ fontSize: 11, fill: isDark ? '#cbd5e1' : '#475569' }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, background: isDark ? '#020617' : '#ffffff', borderColor: isDark ? 'rgba(148,163,184,0.18)' : '#e2e8f0', color: isDark ? '#e2e8f0' : '#0f172a' }} />
              <Legend wrapperStyle={{ fontSize: 12, color: isDark ? '#e2e8f0' : '#0f172a' }} />
              <Bar dataKey="p1" name={p1Name} fill="#7c3aed" radius={[4, 4, 0, 0]} />
              <Bar dataKey="p2" name={p2Name} fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 text-xs uppercase tracking-[0.25em] text-slate-400">Core output</div>
        </div>

        <div className={cn('rounded-3xl border p-4 sm:p-5', panelClass)}>
          <h4 className="mb-4 text-sm font-semibold">Performance Radar (normalised)</h4>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={isDark ? 'rgba(148,163,184,0.18)' : '#e5e7eb'} />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: isDark ? '#cbd5e1' : '#475569' }} />
              <Radar name={p1Name} dataKey={p1Name} stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.25} />
              <Radar name={p2Name} dataKey={p2Name} stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.22} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, background: isDark ? '#020617' : '#ffffff', borderColor: isDark ? 'rgba(148,163,184,0.18)' : '#e2e8f0', color: isDark ? '#e2e8f0' : '#0f172a' }} />
              <Legend wrapperStyle={{ fontSize: 12, color: isDark ? '#e2e8f0' : '#0f172a' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="flex justify-center">
        <button
          type="button"
          onClick={onChangePlayers}
          className={cn('inline-flex items-center gap-2 rounded-full border px-5 py-3 text-sm font-medium transition-colors', isDark ? 'border-white/10 bg-white/5 text-slate-100 hover:bg-white/10' : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50')}
        >
          <RotateCcw className="h-4 w-4" />
          Change Players
        </button>
      </div>
    </div>
  );
}