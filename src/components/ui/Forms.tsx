import { forwardRef, useId, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from 'react';
import { ChevronDown, Search, X } from 'lucide-react';
import { cn } from '../../utils/cn';

// Any control with aria-invalid="true" gets the error border (see Field's `error`).
export const CONTROL =
  'h-10 w-full rounded-md border border-line-strong bg-surface-2 text-sm text-fg placeholder:text-fg-subtle ' +
  'transition-colors hover:border-fg-subtle disabled:cursor-not-allowed disabled:opacity-50 ' +
  'aria-[invalid=true]:border-danger aria-[invalid=true]:hover:border-danger';

/** Props a control needs to announce Field's hint/error to assistive tech. */
export interface FieldA11y {
  'aria-invalid'?: true;
  'aria-describedby'?: string;
}

interface FieldProps {
  label: ReactNode;
  /** Visually hide the label but keep it for screen readers. */
  hideLabel?: boolean;
  hint?: ReactNode;
  /** Validation message: shown in red under the control, which is marked invalid. */
  error?: ReactNode;
  /** Render the control; spread `a11y` onto it so the hint/error are announced. */
  children: (id: string, a11y: FieldA11y) => ReactNode;
  className?: string;
}

/** Label + control + hint/error wiring with a generated id. */
export function Field({ label, hideLabel, hint, error, children, className }: FieldProps) {
  const id = useId();
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [errorId, hintId].filter(Boolean).join(' ') || undefined;
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <label htmlFor={id} className={cn('text-micro font-semibold uppercase text-fg-muted', hideLabel && 'sr-only')}>
        {label}
      </label>
      {children(id, { 'aria-invalid': error ? true : undefined, 'aria-describedby': describedBy })}
      {error && (
        <p id={errorId} role="alert" className="text-micro font-semibold text-danger">
          {error}
        </p>
      )}
      {hint && (
        <p id={hintId} className="text-micro text-fg-subtle">
          {hint}
        </p>
      )}
    </div>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...rest },
  ref,
) {
  return <input ref={ref} className={cn(CONTROL, 'px-3', className)} {...rest} />;
});

/** Styled native select: keeps the platform picker and keyboard behaviour. */
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select(
  { className, children, ...rest },
  ref,
) {
  return (
    <div className={cn('relative', className)}>
      <select ref={ref} className={cn(CONTROL, 'appearance-none pl-3 pr-9')} {...rest}>
        {children}
      </select>
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle"
      />
    </div>
  );
});

interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'value' | 'type'> {
  value: string;
  onChange: (value: string) => void;
  /** Accessible name when there is no visible label. */
  label?: string;
}

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(function SearchInput(
  { value, onChange, label = 'Search', className, placeholder = 'Search…', ...rest },
  ref,
) {
  return (
    <div className={cn('relative', className)}>
      <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
      <input
        ref={ref}
        type="search"
        aria-label={label}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className={cn(CONTROL, 'pl-9 pr-9 [&::-webkit-search-cancel-button]:hidden')}
        {...rest}
      />
      {value && (
        <button
          type="button"
          aria-label="Clear search"
          onClick={() => onChange('')}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded-sm p-1 text-fg-subtle hover:text-fg"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
    </div>
  );
});
