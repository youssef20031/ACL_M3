import { motion, useReducedMotion } from 'framer-motion';
import { AlertTriangle, Loader2 } from 'lucide-react';
import type { Message } from '../../store/appStore';
import { Markdown } from './Markdown';
import { AnswerCards, AnswerCardSkeleton, FollowUps } from './cards/AnswerCards';
import { Logo } from '../ui';
import { cn } from '../../utils/cn';

function AssistantMark({ tone = 'accent' }: { tone?: 'accent' | 'danger' }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        'mt-0.5 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full',
        tone === 'danger' ? 'bg-danger/15 text-danger' : 'bg-brand ring-1 ring-accent/40',
      )}
    >
      {tone === 'danger' ? <AlertTriangle className="h-3.5 w-3.5" /> : <Logo iconOnly size={12} />}
    </span>
  );
}

function timeLabel(ts: number) {
  return new Date(ts).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

interface ChatMessageProps {
  message: Message;
  /** Sends a suggested follow-up question; follow-ups show only when set. */
  onAsk?: (question: string) => void;
  /** Disables the follow-up buttons (a question is in flight or the graph is offline). */
  askDisabled?: boolean;
}

export function ChatMessage({ message, onAsk, askDisabled }: ChatMessageProps) {
  const reduceMotion = useReducedMotion();
  const isUser = message.role === 'user';
  // The page stores failed requests as assistant messages prefixed with "Error:".
  const isError = !isUser && message.content.startsWith('Error:');

  return (
    <motion.li
      initial={reduceMotion ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      className={cn('flex gap-3', isUser ? 'justify-end' : 'justify-start')}
    >
      {!isUser && <AssistantMark tone={isError ? 'danger' : 'accent'} />}
      <div className={cn('min-w-0', isUser ? 'max-w-[85%] sm:max-w-[75%]' : 'flex-1')}>
        <p className="sr-only">
          {isUser ? 'You' : 'Assistant'} at {timeLabel(message.timestamp)}:
        </p>
        {isUser ? (
          <div className="whitespace-pre-wrap break-words rounded-lg rounded-br-sm bg-brand-2 px-4 py-2.5 text-body text-fg">
            {message.content}
          </div>
        ) : (
          <>
            <div
              className={cn(
                'rounded-lg rounded-tl-sm border px-4 py-3',
                isError ? 'border-danger/40 bg-danger/10' : 'border-line bg-surface',
              )}
            >
              <Markdown>{message.content}</Markdown>
            </div>
            {!isError && message.cards && message.cards.length > 0 && <AnswerCards cards={message.cards} />}
            {!isError && onAsk && message.followUps && message.followUps.length > 0 && (
              <FollowUps questions={message.followUps} onAsk={onAsk} disabled={askDisabled} />
            )}
          </>
        )}
      </div>
    </motion.li>
  );
}

export function ThinkingMessage() {
  return (
    <li className="flex gap-3" role="status">
      <AssistantMark />
      <div className="min-w-0 flex-1">
        <div className="inline-flex items-center gap-2 rounded-lg rounded-tl-sm border border-line bg-surface px-4 py-3 text-data text-fg-muted">
          <Loader2 className="h-4 w-4 animate-spin text-accent" aria-hidden="true" />
          Searching the knowledge graph…
        </div>
        <AnswerCardSkeleton className="mt-3 max-w-xl" />
      </div>
    </li>
  );
}
