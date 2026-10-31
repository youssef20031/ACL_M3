import { useEffect, useRef, useState, type FormEvent } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { Lock, Unlock } from 'lucide-react';
import { apiService, handleApiError, isForbiddenError, isNotFoundError, type SiteStats as Stats } from '../services/api';
import {
  Button,
  Card,
  Container,
  EmptyState,
  ErrorState,
  Field,
  Input,
  PageHeader,
  SectionHeading,
  SkeletonRows,
  StatCard,
  TabPanel,
  Tabs,
} from '../components/ui';
import { DailyViewsChart } from '../components/stats/DailyViewsChart';
import { RankedList, type RankedItem } from '../components/stats/RankedList';
import { isOwnerBrowser, setOwnerBrowser } from '../lib/visitor';
import { usePageMeta } from '../hooks/usePageMeta';

/*
 * The owner's view of the page-view counts (GET /api/admin/analytics). Unlocked with the server's
 * ADMIN_TOKEN, kept in this tab's session storage only. Not linked anywhere and not indexed.
 */

const TOKEN_KEY = 'pm-admin-token';
const PERIODS = [
  { value: '7', label: '7 days' },
  { value: '30', label: '30 days' },
  { value: '90', label: '90 days' },
] as const;
type Period = (typeof PERIODS)[number]['value'];

function readToken(): string {
  try {
    return window.sessionStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

function storeToken(token: string) {
  try {
    if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
    else window.sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // kept in memory for this visit
  }
}

let regionNames: Intl.DisplayNames | null | undefined;
function countryName(code: string | null, fallback: string): string {
  if (!code) return 'Not reported';
  if (regionNames === undefined) {
    try {
      regionNames = new Intl.DisplayNames(['en'], { type: 'region' });
    } catch {
      regionNames = null;
    }
  }
  try {
    return regionNames?.of(code) ?? fallback;
  } catch {
    return fallback;
  }
}

/** "America/Argentina/Buenos_Aires" -> "Buenos Aires" with "America / Argentina" as detail. */
function zoneLabel(zone: string): { label: string; detail?: string } {
  if (zone === 'unknown') return { label: 'Not reported' };
  const parts = zone.split('/');
  const city = parts.pop()!.replace(/_/g, ' ');
  return { label: city, detail: parts.length ? parts.join(' / ').replace(/_/g, ' ') : undefined };
}

const percent = (part: number, whole: number) => (whole > 0 ? Math.round((part / whole) * 100) : 0);

function UnlockForm({ error, onUnlock }: { error?: string; onUnlock: (token: string) => void }) {
  const [value, setValue] = useState('');
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (value.trim()) onUnlock(value.trim());
  };
  return (
    <Card className="max-w-md">
      <form onSubmit={submit} className="flex flex-col gap-4">
        <Field
          label="Admin token"
          hint="The server's ADMIN_TOKEN. It stays in this tab only and is forgotten when you close it."
          error={error}
        >
          {(id, a11y) => (
            <Input
              id={id}
              type="password"
              autoComplete="current-password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              {...a11y}
            />
          )}
        </Field>
        <Button type="submit" icon={<Unlock className="h-4 w-4" aria-hidden="true" />} className="self-start">
          Show stats
        </Button>
      </form>
    </Card>
  );
}

function OwnerToggle({ owner, onChange }: { owner: boolean; onChange: (on: boolean) => void }) {
  return (
    <Card>
      <label className="flex cursor-pointer items-start gap-3">
        <input
          type="checkbox"
          checked={owner}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-1 h-4 w-4 shrink-0 accent-accent"
        />
        <span>
          <span className="block text-data font-semibold text-fg">This browser is me</span>
          <span className="block text-data text-fg-muted">
            Views from this browser are counted as yours, apart from everyone else's, so the numbers here are other
            people. Unlocking this page turns it on; do the same on each device you use.
          </span>
        </span>
      </label>
    </Card>
  );
}

function StatsView({ stats }: { stats: Stats }) {
  const known = stats.top_countries.filter((c) => c.code);
  const countries: RankedItem[] = stats.top_countries.map((c) => ({
    key: c.code ?? 'unknown',
    label: countryName(c.code, c.country),
    value: c.visitors,
  }));
  const zones: RankedItem[] = stats.top_zones.map((z) => ({ key: z.zone, ...zoneLabel(z.zone), value: z.visitors }));
  const pages: RankedItem[] = stats.top_paths.map((p) => ({ key: p.path, label: p.path, value: p.views }));
  const referrers: RankedItem[] = stats.top_referrers.map((r) => ({
    key: r.host,
    label: r.host === 'other' ? 'Other sites' : r.host,
    value: r.views,
  }));
  const share = percent(stats.consented_views, stats.total_views);

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard label="Views by others" value={stats.total_views.toLocaleString()} highlight={stats.total_views > 0} />
        <StatCard
          label="Visitors"
          value={stats.visitors.toLocaleString()}
          detail={`${stats.new_visitors.toLocaleString()} new`}
        />
        <StatCard
          label="Countries"
          value={known.length.toLocaleString()}
          detail={known.length ? countryName(known[0].code, known[0].country) : undefined}
        />
        <StatCard label="Your own views" value={stats.owner_views.toLocaleString()} detail="Not in the other numbers" />
      </div>
      <p className="-mt-3 text-data text-fg-muted">
        Views count everyone. Visitors, countries and timezones count only people who accepted the cookie banner:{' '}
        {stats.consented_views.toLocaleString()} of {stats.total_views.toLocaleString()} views ({share}%). A visitor
        is counted once a day, so the same person on two days counts twice.
      </p>

      <Card>
        <SectionHeading title="Views per day" subtitle="Other people's page views" size="md" />
        {stats.total_views > 0 || stats.owner_views > 0 ? (
          <DailyViewsChart days={stats.by_day} className="mt-3" />
        ) : (
          <EmptyState title="No views yet" description={`Nobody else has opened the site in the last ${stats.days} days.`} />
        )}
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <SectionHeading title="Countries" subtitle="Visitors, from the browser's timezone" size="md" />
          <RankedList items={countries} unit="visitor" empty="No visitors who accepted the banner yet." className="mt-3" />
        </Card>
        <Card>
          <SectionHeading title="Timezones" subtitle="Visitors, as their browsers report them" size="md" />
          <RankedList items={zones} unit="visitor" empty="No visitors who accepted the banner yet." className="mt-3" />
        </Card>
        <Card>
          <SectionHeading title="Pages" subtitle="Views by others" size="md" />
          <RankedList items={pages} unit="view" empty="No views yet." className="mt-3" />
        </Card>
        <Card>
          <SectionHeading title="Referring sites" subtitle="Where visits came from, when another site linked here" size="md" />
          <RankedList items={referrers} unit="view" empty="No visits from other sites yet." className="mt-3" />
        </Card>
      </div>
    </div>
  );
}

export function SiteStats() {
  usePageMeta({ title: 'Site stats', description: 'Visitor counts for the site owner.', noindex: true });

  const [token, setToken] = useState(readToken);
  const [period, setPeriod] = useState<Period>('30');
  const [formError, setFormError] = useState<string>();
  const [owner, setOwner] = useState(isOwnerBrowser);
  const justUnlocked = useRef(false);

  const query = useQuery({
    queryKey: ['site-stats', period, token],
    queryFn: () => apiService.getSiteStats(Number(period), token),
    enabled: token !== '',
    retry: false,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (query.isSuccess && justUnlocked.current) {
      justUnlocked.current = false;
      setOwnerBrowser(true);
      setOwner(true);
    }
  }, [query.isSuccess]);

  useEffect(() => {
    if (query.error && isForbiddenError(query.error)) {
      justUnlocked.current = false;
      storeToken('');
      setToken('');
      setFormError("That token isn't right.");
    }
  }, [query.error]);

  const unlock = (value: string) => {
    justUnlocked.current = true;
    setFormError(undefined);
    storeToken(value);
    setToken(value);
  };

  const lock = () => {
    storeToken('');
    setToken('');
  };

  const toggleOwner = (on: boolean) => {
    setOwnerBrowser(on);
    setOwner(on);
  };

  let body;
  if (!token) body = <UnlockForm error={formError} onUnlock={unlock} />;
  else if (query.error && isNotFoundError(query.error))
    body = (
      <ErrorState
        title="Stats are switched off"
        description="The server has no ADMIN_TOKEN. Set it in the server's environment (deploy/.env on the VM) and restart the backend."
      />
    );
  else if (query.error && !isForbiddenError(query.error))
    body = <ErrorState description={handleApiError(query.error)} onRetry={() => query.refetch()} />;
  else
    body = (
      <div className="flex flex-col gap-6">
        <OwnerToggle owner={owner} onChange={toggleOwner} />
        <div>
          <Tabs items={[...PERIODS]} value={period} onChange={setPeriod} label="Period" idPrefix="site-stats" />
          <TabPanel idPrefix="site-stats" value={period} className="pt-6">
            {query.data ? <StatsView stats={query.data} /> : <SkeletonRows rows={6} />}
          </TabPanel>
        </div>
      </div>
    );

  return (
    <Container className="pb-12">
      <PageHeader
        eyebrow="Owner only"
        title="Site stats"
        description="Who has been visiting. Counts are anonymous: no IP addresses, cookies or identifiers are stored."
        actions={
          token ? (
            <Button variant="secondary" icon={<Lock className="h-4 w-4" aria-hidden="true" />} onClick={lock}>
              Lock
            </Button>
          ) : undefined
        }
      />
      {body}
    </Container>
  );
}
