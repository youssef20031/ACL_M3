import { Component, type ErrorInfo, type ReactNode } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { CornerDownRight } from 'lucide-react';
import type { AnswerCard } from '../../../services/api';
import { Skeleton } from '../../ui';
import { cn } from '../../../utils/cn';
import { PhotoLookupProvider } from './shared';
import { PlayerAnswerCard } from './PlayerAnswerCard';
import { ComparisonAnswerCard } from './ComparisonAnswerCard';
import { LeaderboardAnswerCard } from './LeaderboardAnswerCard';
import { PredictionAnswerCard } from './PredictionAnswerCard';
import { TeamAnswerCard } from './TeamAnswerCard';
import { FixturesAnswerCard } from './FixturesAnswerCard';
import { SeasonAnswerCard } from './SeasonAnswerCard';
import { PitchAnswerCard } from './PitchAnswerCard';

/** A card that fails to render is left out; the answer text and other cards stay. */
class CardBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Answer card failed to render', error, info.componentStack);
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function CardView({ card }: { card: AnswerCard }) {
  switch (card.type) {
    case 'player':
      return <PlayerAnswerCard card={card} />;
    case 'comparison':
      return <ComparisonAnswerCard card={card} />;
    case 'leaderboard':
      return <LeaderboardAnswerCard card={card} />;
    case 'prediction':
      return <PredictionAnswerCard card={card} />;
    case 'team':
      return <TeamAnswerCard card={card} />;
    case 'fixtures':
      return <FixturesAnswerCard card={card} />;
    case 'season':
      return <SeasonAnswerCard card={card} />;
    case 'pitch':
      return <PitchAnswerCard card={card} />;
    default:
      return null; // a card type this build doesn't know yet
  }
}

interface AnswerCardsProps {
  cards: AnswerCard[];
  /** Look up photos the backend didn't send (off for /design's made-up players). */
  lookupPhotos?: boolean;
  className?: string;
}

/** The visual blocks under an assistant answer, revealed one after another. */
export function AnswerCards({ cards, lookupPhotos = true, className }: AnswerCardsProps) {
  const reduceMotion = useReducedMotion();
  if (!cards.length) return null;
  return (
    <PhotoLookupProvider enabled={lookupPhotos}>
      <div className={cn('mt-3 space-y-3', className)}>
        {cards.map((card, i) => (
          <motion.div
            key={card.id ?? i}
            initial={reduceMotion ? false : { opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25, delay: reduceMotion ? 0 : 0.1 + i * 0.08, ease: 'easeOut' }}
          >
            <CardBoundary>
              <CardView card={card} />
            </CardBoundary>
          </motion.div>
        ))}
      </div>
    </PhotoLookupProvider>
  );
}

interface FollowUpsProps {
  questions: string[];
  onAsk: (question: string) => void;
  disabled?: boolean;
  className?: string;
}

/** Suggested next questions; each one sends straight away. */
export function FollowUps({ questions, onAsk, disabled, className }: FollowUpsProps) {
  if (!questions.length) return null;
  return (
    <nav aria-label="Suggested follow-up questions" className={cn('mt-3', className)}>
      <p className="mb-1.5 text-micro font-semibold uppercase text-fg-subtle">Ask next</p>
      <ul className="flex flex-wrap gap-2">
        {questions.map((q) => (
          <li key={q} className="min-w-0 max-w-full">
            <button
              type="button"
              onClick={() => onAsk(q)}
              disabled={disabled}
              className={cn(
                'inline-flex max-w-full items-center gap-1.5 rounded-full border border-line-strong bg-surface px-3 py-1.5 text-left text-data text-fg-muted transition-colors',
                'hover:border-accent hover:text-fg disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-line-strong disabled:hover:text-fg-muted',
              )}
            >
              <CornerDownRight className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden="true" />
              <span className="min-w-0 break-words">{q}</span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** Placeholder while an answer is on its way: the shape of a card, not its data. */
export function AnswerCardSkeleton({ className }: { className?: string }) {
  return (
    <div aria-hidden="true" className={cn('overflow-hidden rounded-lg border border-line bg-surface', className)}>
      <div className="flex items-center gap-3 p-4">
        <Skeleton className="h-14 w-14 rounded-md" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-5 w-40 max-w-full" />
          <Skeleton className="h-3 w-32" />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-px border-t border-line bg-line sm:grid-cols-6">
        {Array.from({ length: 6 }, (_, i) => (
          <div key={i} className={cn('space-y-1.5 bg-surface px-3 py-2.5', i >= 3 && 'hidden sm:block')}>
            <Skeleton className="h-2.5 w-10" />
            <Skeleton className="h-4 w-8" />
          </div>
        ))}
      </div>
      <div className="flex h-16 items-end gap-[2px] border-t border-line px-4 py-3">
        {[40, 75, 55, 90, 30, 65, 50, 80].map((h, i) => (
          <div key={i} className="max-w-6 flex-1 animate-pulse rounded-t-[4px] bg-surface-2" style={{ height: `${h}%` }} />
        ))}
      </div>
    </div>
  );
}
