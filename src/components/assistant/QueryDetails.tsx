import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Check, Copy, Info, Network, X } from 'lucide-react';
import type { QueryResponse } from '../../services/api';
import { GraphVisualization } from '../GraphVisualization';
import { Button, Chip, EmptyState, TabPanel, Tabs, type TabItem } from '../ui';
import { cn } from '../../utils/cn';

type DetailTab = 'query' | 'cypher' | 'context' | 'graph';

const TABS: TabItem<DetailTab>[] = [
  { value: 'query', label: 'Query' },
  { value: 'cypher', label: 'Cypher' },
  { value: 'context', label: 'Context' },
  { value: 'graph', label: 'Graph' },
];

const ID_PREFIX = 'assistant-details';

function humanize(key: string) {
  return key.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

function isEmptyValue(v: unknown) {
  return v == null || v === '' || (Array.isArray(v) && v.length === 0) || (typeof v === 'object' && !Array.isArray(v) && Object.keys(v as object).length === 0);
}

function valueText(v: unknown): string {
  if (v == null) return '';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** Strip the common indentation the backend's triple-quoted queries carry. */
function dedent(text: string) {
  const lines = text.replace(/\t/g, '  ').split('\n');
  while (lines.length && !lines[0].trim()) lines.shift();
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  const indents = lines.filter((l) => l.trim()).map((l) => l.match(/^ */)![0].length);
  const min = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => l.slice(min)).join('\n');
}

function Label({ children }: { children: ReactNode }) {
  return <h3 className="mb-1.5 text-micro font-semibold uppercase text-fg-subtle">{children}</h3>;
}

function CodeBlock({ children, className }: { children: string; className?: string }) {
  return (
    <pre
      className={cn(
        'overflow-x-auto whitespace-pre-wrap break-words rounded-md border border-line bg-bg p-3 font-mono text-data leading-relaxed text-fg-muted',
        className,
      )}
    >
      {children}
    </pre>
  );
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number>();
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={copy}
      aria-label={copied ? `${label} copied` : `Copy ${label}`}
      icon={copied ? <Check className="h-3.5 w-3.5 text-accent" aria-hidden="true" /> : <Copy className="h-3.5 w-3.5" aria-hidden="true" />}
    >
      <span aria-live="polite">{copied ? 'Copied' : 'Copy'}</span>
    </Button>
  );
}

function QueryTab({ info }: { info: QueryResponse }) {
  const entities = Object.entries(info.entities ?? {}).filter(([, v]) => !isEmptyValue(v));
  return (
    <div className="space-y-5">
      <div>
        <Label>Intent</Label>
        <Chip tone="accent" className="text-data">{info.intent}</Chip>
      </div>

      <div>
        <Label>Retrieval</Label>
        <ul className="flex flex-wrap gap-1.5">
          <li>
            <Chip tone="neutral">{info.results?.length ?? 0} graph results</Chip>
          </li>
          <li>
            <Chip tone={info.embedding_used ? 'info' : 'neutral'}>
              {info.embedding_used ? 'Embedding search used' : 'No embedding search'}
            </Chip>
          </li>
        </ul>
        {info.embedding_used && (
          <p className="mt-1.5 text-data text-fg-subtle">Similar players were found with semantic search.</p>
        )}
      </div>

      <div>
        <Label>Entities</Label>
        {entities.length === 0 ? (
          <p className="text-data text-fg-subtle">No entities detected in the question.</p>
        ) : (
          <dl className="divide-y divide-line rounded-md border border-line">
            {entities.map(([key, value]) => (
              <div key={key} className="grid grid-cols-[7rem_minmax(0,1fr)] gap-3 px-3 py-2">
                <dt className="text-data text-fg-subtle">{humanize(key)}</dt>
                <dd className="flex min-w-0 flex-wrap gap-1">
                  {(Array.isArray(value) ? value : [value]).map((v, i) => (
                    <Chip key={i} tone="brand" className="whitespace-normal break-all text-data font-medium">
                      {valueText(v)}
                    </Chip>
                  ))}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </div>
  );
}

interface QueryDetailsProps {
  info: QueryResponse | null;
  onClose: () => void;
  id?: string;
  className?: string;
}

/** Explains how the last answer was produced: intent, Cypher, retrieved context, graph. */
export function QueryDetails({ info, onClose, id, className }: QueryDetailsProps) {
  const [tab, setTab] = useState<DetailTab>('query');
  const hasGraph = !!info?.graph_data && info.graph_data.nodes.length > 0;

  return (
    <aside id={id} aria-labelledby={`${ID_PREFIX}-title`} className={cn('flex min-h-0 flex-col bg-surface', className)}>
      <div className="flex items-center justify-between gap-2 border-b border-line px-4 pt-3">
        <h2 id={`${ID_PREFIX}-title`} className="flex items-center gap-2 font-display text-h3 font-bold text-fg">
          <Info className="h-4 w-4 text-accent" aria-hidden="true" />
          Query details
        </h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Hide details"
          className="rounded-sm p-1.5 text-fg-subtle transition-colors hover:bg-surface-2 hover:text-fg"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      {!info ? (
        <EmptyState
          className="flex-1"
          icon={<Network className="h-8 w-8" aria-hidden="true" />}
          title="Nothing to show yet"
          description="Ask a question to see the detected intent, the Cypher query, the context retrieved from the graph and a graph view."
        />
      ) : (
        <>
          <Tabs items={TABS} value={tab} onChange={setTab} label="Query details" idPrefix={ID_PREFIX} className="px-2" />
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === 'query' && (
              <TabPanel idPrefix={ID_PREFIX} value="query">
                <QueryTab info={info} />
              </TabPanel>
            )}

            {tab === 'cypher' && (
              <TabPanel idPrefix={ID_PREFIX} value="cypher">
                {info.cypher_query ? (
                  <>
                    <div className="mb-1.5 flex items-center justify-between">
                      <Label>Generated Cypher</Label>
                      <CopyButton text={info.cypher_query} label="Cypher query" />
                    </div>
                    <CodeBlock className="text-accent">{dedent(info.cypher_query)}</CodeBlock>
                  </>
                ) : (
                  <EmptyState title="No Cypher query" description="This answer did not run a graph query." />
                )}
              </TabPanel>
            )}

            {tab === 'context' && (
              <TabPanel idPrefix={ID_PREFIX} value="context" className="space-y-5">
                <div>
                  <Label>Knowledge graph context</Label>
                  {info.kg_context ? (
                    <CodeBlock>{info.kg_context}</CodeBlock>
                  ) : (
                    <p className="text-data text-fg-subtle">No graph context was retrieved.</p>
                  )}
                </div>
                {info.embedding_context && (
                  <div>
                    <Label>Embedding context</Label>
                    <CodeBlock>{info.embedding_context}</CodeBlock>
                  </div>
                )}
              </TabPanel>
            )}

            {tab === 'graph' && (
              <TabPanel idPrefix={ID_PREFIX} value="graph">
                {hasGraph ? (
                  <GraphVisualization data={info.graph_data!} />
                ) : (
                  <EmptyState
                    icon={<Network className="h-8 w-8" aria-hidden="true" />}
                    title="No graph for this answer"
                    description="The query did not return nodes that can be drawn."
                  />
                )}
              </TabPanel>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
