import { useCallback, useRef, useState } from 'react';

/**
 * Width of an element in CSS pixels, kept current with a ResizeObserver, so an SVG chart can
 * draw at its real size (text stays legible instead of scaling with a viewBox).
 * Returns a callback ref to put on the element, and its width (`fallback` until measured).
 *
 * The observer lives and dies with the element: React calls the ref with null on unmount, which
 * disconnects it. (No effect cleanup: StrictMode's rehearsal unmount would disconnect it for good.)
 */
export function useElementWidth<T extends HTMLElement>(fallback = 600): [(el: T | null) => void, number] {
  const [width, setWidth] = useState(fallback);
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback((el: T | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!el) return;
    const measure = (w: number) => {
      if (w > 0) setWidth((prev) => (Math.abs(prev - w) >= 1 ? Math.floor(w) : prev));
    };
    measure(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return;
    observer.current = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) measure(entry.contentRect.width);
    });
    observer.current.observe(el);
  }, []);

  return [ref, width];
}
