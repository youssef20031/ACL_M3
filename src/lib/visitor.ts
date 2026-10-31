/**
 * What a page view tells the server beyond its path, read in this browser:
 * - for a visitor who accepted the consent banner, the browser's timezone and two flags kept in
 *   local storage (first visit ever, first view of the UTC day), so the owner's stats can count
 *   visitors and countries without any identifier;
 * - on the owner's own browsers ("This browser is me" on /stats), a flag that keeps their views
 *   apart from everyone else's.
 */

const SINCE = 'pm-visitor-since';
const LAST_DAY = 'pm-visitor-last-day';
const OWNER = 'pm-owner';

export interface VisitorDetails {
  timezone: string | null;
  new_visitor: boolean;
  first_today: boolean;
}

const utcDay = () => new Date().toISOString().slice(0, 10);

/** Call once per counted view, and only with consent: it updates the local-storage flags. */
export function visitorDetails(): VisitorDetails {
  let newVisitor = false;
  let firstToday = false;
  try {
    const today = utcDay();
    if (!window.localStorage.getItem(SINCE)) {
      window.localStorage.setItem(SINCE, today);
      newVisitor = true;
    }
    if (window.localStorage.getItem(LAST_DAY) !== today) {
      window.localStorage.setItem(LAST_DAY, today);
      firstToday = true;
    }
  } catch {
    // storage blocked: without the flags this browser isn't counted as a visitor
  }
  let timezone: string | null = null;
  try {
    timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || null;
  } catch {
    timezone = null;
  }
  return { timezone, new_visitor: newVisitor, first_today: firstToday };
}

/** Removes the visitor flags (consent declined or withdrawn). */
export function forgetVisitor() {
  try {
    window.localStorage.removeItem(SINCE);
    window.localStorage.removeItem(LAST_DAY);
  } catch {
    // nothing stored
  }
}

export function isOwnerBrowser(): boolean {
  try {
    return window.localStorage.getItem(OWNER) === '1';
  } catch {
    return false;
  }
}

export function setOwnerBrowser(on: boolean) {
  try {
    if (on) window.localStorage.setItem(OWNER, '1');
    else window.localStorage.removeItem(OWNER);
  } catch {
    // keep counting this browser as a visitor
  }
}
