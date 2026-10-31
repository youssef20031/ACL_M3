import { useSyncExternalStore } from 'react';
import { forgetVisitor } from './visitor';

/**
 * "accepted" adds this browser's timezone and visitor flags to the cookie-free page-view counter
 * (lib/visitor.ts); without it views are only counted anonymously. null means not asked yet.
 */
export type Consent = 'accepted' | 'declined' | null;

const KEY = 'pm-analytics-consent';
const listeners = new Set<() => void>();

function read(): Consent {
  try {
    const value = window.localStorage.getItem(KEY);
    if (value === 'accepted' || value === 'declined') return value;
  } catch {
    // storage blocked (private mode, disabled site data): treat as not asked
  }
  // Global Privacy Control is an explicit "no": honour it without showing the banner
  if ((navigator as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl) return 'declined';
  return null;
}

let current: Consent = typeof window === 'undefined' ? null : read();

export function getConsent(): Consent {
  return current;
}

export function setConsent(value: Consent) {
  current = value;
  if (value !== 'accepted') forgetVisitor();
  try {
    if (value) window.localStorage.setItem(KEY, value);
    else window.localStorage.removeItem(KEY);
  } catch {
    // keep the in-memory choice for this visit
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useConsent(): Consent {
  return useSyncExternalStore(subscribe, getConsent, () => null);
}
