import { useEffect, useRef, type FormEvent, type KeyboardEvent, type ReactNode } from 'react';
import { ArrowUp } from 'lucide-react';
import { cn } from '../../utils/cn';

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: FormEvent) => void;
  /** No input at all (e.g. the graph is offline). */
  disabled: boolean;
  /** A request is in flight: typing is allowed, sending is not. */
  busy: boolean;
  placeholder: string;
  /** Small line under the field (shortcuts, active model). */
  footer?: ReactNode;
}

const MAX_HEIGHT = 168;
/** Matches the API's QueryRequest limit, so a long question is stopped here, not rejected after sending. */
export const MAX_QUESTION_CHARS = 1000;
const COUNTER_FROM = 850;

export function Composer({ value, onChange, onSubmit, disabled, busy, placeholder, footer }: ComposerProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to MAX_HEIGHT, then scroll inside.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  const canSend = !disabled && !busy && value.trim().length > 0;

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter adds a line. Ignore Enter while an IME is composing.
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (canSend) e.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <form onSubmit={onSubmit} aria-label="Ask the assistant">
      <div
        className={cn(
          'flex items-end gap-2 rounded-lg border border-line-strong bg-surface-2 p-1.5 pl-3.5 transition-colors',
          'focus-within:border-accent-2 focus-within:ring-1 focus-within:ring-accent-2',
          disabled && 'opacity-60',
        )}
      >
        <label htmlFor="assistant-input" className="sr-only">
          Your question
        </label>
        <textarea
          ref={ref}
          id="assistant-input"
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          maxLength={MAX_QUESTION_CHARS}
          aria-describedby={footer ? 'assistant-input-hint' : undefined}
          className={cn(
            'min-h-[2.25rem] flex-1 resize-none bg-transparent py-1.5 text-body text-fg placeholder:text-fg-subtle',
            'focus:outline-none focus-visible:outline-none disabled:cursor-not-allowed',
          )}
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label="Send question"
          className={cn(
            'inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent text-accent-on transition-colors',
            'hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-surface disabled:text-fg-subtle',
          )}
        >
          <ArrowUp className="h-5 w-5" aria-hidden="true" />
        </button>
      </div>
      {value.length >= COUNTER_FROM && (
        <p
          aria-live="polite"
          className={cn('mt-1 px-1 text-right text-micro', value.length >= MAX_QUESTION_CHARS ? 'font-semibold text-danger' : 'text-warning')}
        >
          {value.length >= MAX_QUESTION_CHARS
            ? `That's the limit: ${MAX_QUESTION_CHARS} characters. Try a shorter question.`
            : `${MAX_QUESTION_CHARS - value.length} characters left`}
        </p>
      )}
      {footer && (
        <div id="assistant-input-hint" className="mt-1.5 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1 text-micro text-fg-subtle">
          {footer}
        </div>
      )}
    </form>
  );
}
