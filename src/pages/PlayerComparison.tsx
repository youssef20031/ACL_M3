import { useState, useEffect, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { ArrowLeftRight, Loader2, AlertCircle, Search } from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from 'recharts';


const SEASONS = ['', '2020-21', '2021-22', '2022-23'];

const EXAMPLE_PAIRS = [
  { p1: 'Mohamed Salah', p2: 'Son Heung-min' },
  { p1: 'Erling Haaland', p2: 'Harry Kane' },
  { p1: 'Kevin De Bruyne', p2: 'Bruno Fernandes' },
];

type PlayerStats = Record<string, any>;

export function PlayerComparison() {
  const { neo4jConnected } = useAppStore();
  const [player1, setPlayer1] = useState('');
  const [player2, setPlayer2] = useState('');
  type Suggestion = { name: string; position?: string; season?: string; avatar?: string };
  const [p1Suggestions, setP1Suggestions] = useState<Suggestion[]>([]);
  const [p2Suggestions, setP2Suggestions] = useState<Suggestion[]>([]);
  const [p1Index, setP1Index] = useState(-1);
  const [p2Index, setP2Index] = useState(-1);
  const p1Timer = useRef<number | null>(null);
  const p2Timer = useRef<number | null>(null);
  const p1SuppressSearch = useRef(false);
  const p2SuppressSearch = useRef(false);
  const p1InputRef = useRef<HTMLInputElement | null>(null);
  const p2InputRef = useRef<HTMLInputElement | null>(null);
  const p1ListId = 'p1-suggestions-list';
  const p2ListId = 'p2-suggestions-list';
  const [season, setSeason] = useState('');
  const [comparisonData, setComparisonData] = useState<PlayerStats[]>([]);

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
    setPlayer1(p1);
    setPlayer2(p2);
    compareMutation.mutate({ p1, p2, s: season });
  };

  // helper to run search and update suggestions
  const runSearch = async (q: string, setSuggestions: (s: Suggestion[]) => void) => {
    if (!q || q.trim().length < 2) {
      setSuggestions([]);
      return;
    }

    try {
      const res = await apiService.searchPlayers(q.trim());
      const players = (res.players || []).map((p: any) => ({
        name: p.player_name || p.name || '',
        position: p.position || p.pos || undefined,
        season: p.season || undefined,
        avatar: p.avatar || p.image || p.photo || undefined,
      })).filter((x: Suggestion) => x.name);
      setSuggestions(players.slice(0, 10));
    } catch (e) {
      setSuggestions([]);
    }
  };

  // debounce effects for player1
  useEffect(() => {
    if (p1Timer.current) window.clearTimeout(p1Timer.current);
    p1Timer.current = window.setTimeout(() => {
      if (p1SuppressSearch.current) {
        p1SuppressSearch.current = false;
        setP1Index(-1);
        return;
      }
      runSearch(player1, setP1Suggestions);
      setP1Index(-1);
    }, 250) as unknown as number;

    return () => {
      if (p1Timer.current) window.clearTimeout(p1Timer.current);
    };
  }, [player1]);

  // debounce effects for player2
  useEffect(() => {
    if (p2Timer.current) window.clearTimeout(p2Timer.current);
    p2Timer.current = window.setTimeout(() => {
      if (p2SuppressSearch.current) {
        p2SuppressSearch.current = false;
        setP2Index(-1);
        return;
      }
      runSearch(player2, setP2Suggestions);
      setP2Index(-1);
    }, 250) as unknown as number;

    return () => {
      if (p2Timer.current) window.clearTimeout(p2Timer.current);
    };
  }, [player2]);

  // auto-default when single last-name match
  useEffect(() => {
    const q = player1.trim();
    if (q && !q.includes(' ') && p1Suggestions.length === 1) {
      p1SuppressSearch.current = true;
      setPlayer1(p1Suggestions[0].name);
      setP1Suggestions([]);
    }
  }, [p1Suggestions]);

  useEffect(() => {
    const q = player2.trim();
    if (q && !q.includes(' ') && p2Suggestions.length === 1) {
      p2SuppressSearch.current = true;
      setPlayer2(p2Suggestions[0].name);
      setP2Suggestions([]);
    }
  }, [p2Suggestions]);

  // keyboard handlers
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
        // return focus to input
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

  return (
    <div className="flex flex-col h-full">
      <div className="bg-white border-b px-4 py-4 sm:px-6">
        <h2 className="text-xl font-bold text-gray-900 sm:text-2xl">⚖️ Compare Players</h2>
        <p className="text-sm text-gray-600 mt-1">Head-to-head stats comparison between two players</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6 max-w-4xl mx-auto w-full sm:px-6">
        {!neo4jConnected && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 shrink-0" />
            <p className="text-sm text-yellow-800">Connect to Neo4j in Settings to compare players.</p>
          </div>
        )}

        <form onSubmit={handleCompare} className="bg-white border rounded-xl p-4 space-y-4 sm:p-5" autoComplete="off">
          {/* Hidden fields to deter browser autofill/credentials suggestions */}
          <input aria-hidden="true" tabIndex={-1} style={{position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden'}} type="text" name="__fake_user" autoComplete="username" />
          <input aria-hidden="true" tabIndex={-1} style={{position: 'absolute', left: '-9999px', width: '1px', height: '1px', overflow: 'hidden'}} type="password" name="__fake_pass" autoComplete="current-password" />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="relative">
              <label className="block text-sm font-medium text-gray-700 mb-1">Player 1</label>
              <input
                ref={(el) => (p1InputRef.current = el)}
                id="p1-input"
                name="player1_search"
                type="text"
                value={player1}
                onChange={(e) => setPlayer1(e.target.value)}
                onKeyDown={onP1KeyDown}
                placeholder="e.g. Mohamed Salah"
                disabled={!neo4jConnected}
                aria-autocomplete="list"
                aria-controls={p1ListId}
                aria-expanded={p1Suggestions.length > 0}
                aria-activedescendant={p1Index >= 0 ? `p1-suggestion-${p1Index}` : undefined}
                role="combobox"
                autoComplete="off"
                className="w-full px-3 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100"
              />
              {p1Suggestions.length > 0 && (
                <ul id={p1ListId} role="listbox" className="absolute z-20 bg-white border rounded shadow mt-1 w-full max-h-56 overflow-auto">
                  {p1Suggestions.map((s, i) => (
                    <li
                      id={`p1-suggestion-${i}`}
                      role="option"
                      aria-selected={i === p1Index}
                      key={s.name + i}
                      title=""
                      onClick={() => {
                        p1SuppressSearch.current = true;
                        setPlayer1(s.name);
                        setP1Suggestions([]);
                        p1InputRef.current?.focus();
                      }}
                      className={`px-3 py-2 cursor-pointer text-sm flex items-center justify-between gap-2 ${
                        i === p1Index ? 'bg-purple-50' : 'hover:bg-purple-50'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {s.avatar && (
                          <img src={s.avatar} alt={`${s.name} avatar`} title="" role="img" className="w-6 h-6 rounded-full" />
                        )}
                        <div>
                          <div className="font-medium">{s.name}</div>
                          {(s.position || s.season) && (
                            <div className="text-xs text-gray-500">{[s.position, s.season].filter(Boolean).join(' · ')}</div>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">{i === p1Index ? '↵' : ''}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="relative">
              <label className="block text-sm font-medium text-gray-700 mb-1">Player 2</label>
              <input
                ref={(el) => (p2InputRef.current = el)}
                id="p2-input"
                name="player2_search"
                type="text"
                value={player2}
                onChange={(e) => setPlayer2(e.target.value)}
                onKeyDown={onP2KeyDown}
                placeholder="e.g. Son Heung-min"
                disabled={!neo4jConnected}
                aria-autocomplete="list"
                aria-controls={p2ListId}
                aria-expanded={p2Suggestions.length > 0}
                aria-activedescendant={p2Index >= 0 ? `p2-suggestion-${p2Index}` : undefined}
                role="combobox"
                autoComplete="off"
                className="w-full px-3 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100"
              />
              {p2Suggestions.length > 0 && (
                <ul id={p2ListId} role="listbox" className="absolute z-20 bg-white border rounded shadow mt-1 w-full max-h-56 overflow-auto">
                  {p2Suggestions.map((s, i) => (
                    <li
                      id={`p2-suggestion-${i}`}
                      role="option"
                      aria-selected={i === p2Index}
                      key={s.name + i}
                      title=""
                      onClick={() => {
                        p2SuppressSearch.current = true;
                        setPlayer2(s.name);
                        setP2Suggestions([]);
                        p2InputRef.current?.focus();
                      }}
                      className={`px-3 py-2 cursor-pointer text-sm flex items-center justify-between gap-2 ${
                        i === p2Index ? 'bg-purple-50' : 'hover:bg-purple-50'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {s.avatar && (
                          <img src={s.avatar} alt={`${s.name} avatar`} title="" role="img" className="w-6 h-6 rounded-full" />
                        )}
                        <div>
                          <div className="font-medium">{s.name}</div>
                          {(s.position || s.season) && (
                            <div className="text-xs text-gray-500">{[s.position, s.season].filter(Boolean).join(' · ')}</div>
                          )}
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">{i === p2Index ? '↵' : ''}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:flex-wrap sm:items-end">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Season (optional)</label>
              <select
                value={season}
                onChange={(e) => setSeason(e.target.value)}
                disabled={!neo4jConnected}
                className="px-3 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100"
              >
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
              className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 sm:w-auto"
            >
              {compareMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <ArrowLeftRight className="w-4 h-4" />
              )}
              Compare
            </button>
          </div>
        </form>

        {comparisonData.length === 0 && neo4jConnected && (
          <div>
            <p className="text-sm text-gray-500 mb-2">Try these matchups:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_PAIRS.map((pair) => (
                <button
                  key={pair.p1}
                  onClick={() => handleExample(pair.p1, pair.p2)}
                  className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-purple-100 hover:text-purple-700 rounded-lg transition-colors"
                >
                  {pair.p1} vs {pair.p2}
                </button>
              ))}
            </div>
          </div>
        )}

        {compareMutation.isError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
            {handleApiError(compareMutation.error)}
          </div>
        )}

        {compareMutation.isSuccess && comparisonData.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <Search className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>No data found for these players.</p>
            <p className="text-sm mt-1">Check the spelling or try different names.</p>
          </div>
        )}

        {p1Data && p2Data && <ComparisonResults p1={p1Data} p2={p2Data} />}

        {comparisonData.length === 1 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
            Only found data for <strong>{comparisonData[0].player_name}</strong>. The second player may not exist or have no data for the selected season.
          </div>
        )}
      </div>
    </div>
  );
}

function ComparisonResults({ p1, p2 }: { p1: PlayerStats; p2: PlayerStats }) {
  const p1Name = p1.player_name ?? 'Player 1';
  const p2Name = p2.player_name ?? 'Player 2';

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
    { label: 'Avg ICT', p1v: p1.avg_ict_index, p2v: p2.avg_ict_index, compare: true },
    { label: 'Value (£m)', p1v: p1.avg_value_millions, p2v: p2.avg_value_millions, compare: false },
  ].filter((r) => r.p1v != null || r.p2v != null);

  return (
    <div className="space-y-6">
      {/* Player headers */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {[p1, p2].map((p, i) => {
          const name = p.player_name ?? `Player ${i + 1}`;
          const color = i === 0 ? 'from-purple-600 to-indigo-700' : 'from-pink-500 to-rose-600';
          return (
            <div key={i} className={`bg-gradient-to-br ${color} text-white rounded-xl p-5 text-center`}>
              {p.avatar ? (
                <img
                  src={p.avatar}
                  alt={`${name} avatar`}
                  className="w-14 h-14 rounded-full object-cover mx-auto mb-2 border-2 border-white/30"
                />
              ) : (
                <div className="w-14 h-14 bg-white/20 rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-2">
                  {name[0].toUpperCase()}
                </div>
              )}
              <h3 className="font-bold text-lg leading-tight">{name}</h3>
              <div className="flex items-center justify-center gap-2 mt-1">
                {p.position && (
                  <span className="text-xs bg-white/20 px-2 py-0.5 rounded-full">{p.position}</span>
                )}
                {p.season && <span className="text-xs opacity-75">{p.season}</span>}
              </div>
              {p.total_points != null && (
                <p className="text-3xl font-bold mt-3">
                  {p.total_points} <span className="text-sm font-normal opacity-75">pts</span>
                </p>
              )}
            </div>
          );
        })}
      </div>

      {/* Stat table */}
        <div className="bg-white border rounded-xl overflow-hidden overflow-x-auto">
        <div className="grid grid-cols-3 bg-gray-50 border-b px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wide">
          <span className="text-purple-600">{p1Name}</span>
          <span className="text-center">Stat</span>
          <span className="text-right text-pink-600">{p2Name}</span>
        </div>
        {statRows.map((row, i) => {
          const p1Wins = row.compare && row.p1v != null && row.p2v != null && Number(row.p1v) > Number(row.p2v);
          const p2Wins = row.compare && row.p1v != null && row.p2v != null && Number(row.p2v) > Number(row.p1v);
          return (
            <div
              key={i}
              className={`grid grid-cols-3 px-4 py-3 text-sm ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}`}
            >
              <span className={`font-medium ${p1Wins ? 'text-purple-700' : 'text-gray-700'}`}>
                {row.p1v ?? '—'}
              </span>
              <span className="text-center text-xs text-gray-400 self-center">{row.label}</span>
              <span className={`text-right font-medium ${p2Wins ? 'text-pink-600' : 'text-gray-700'}`}>
                {row.p2v ?? '—'}
              </span>
            </div>
          );
        })}
      </div>

      {/* Bar chart */}
      <div className="bg-white border rounded-xl p-4 sm:p-5">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">Side-by-Side Comparison</h4>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={barStats} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="p1" name={p1Name} fill="#7c3aed" radius={[4, 4, 0, 0]} />
            <Bar dataKey="p2" name={p2Name} fill="#ec4899" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Radar chart */}
      <div className="bg-white border rounded-xl p-4 sm:p-5">
        <h4 className="text-sm font-semibold text-gray-700 mb-4">Performance Radar (normalised)</h4>
        <ResponsiveContainer width="100%" height={280}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#e5e7eb" />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
            <Radar name={p1Name} dataKey={p1Name} stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.25} />
            <Radar name={p2Name} dataKey={p2Name} stroke="#ec4899" fill="#ec4899" fillOpacity={0.25} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
}
