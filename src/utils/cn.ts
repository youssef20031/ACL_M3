import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

// Teach tailwind-merge the custom theme values from tailwind.config.js. Unknown values
// aren't grouped, so a conflicting class is kept instead of replaced: `text-h1` was read
// as a colour and dropped after `text-fg`, and `max-w-3xl` couldn't override `max-w-page`.
const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': [{ text: ['display', 'h1', 'h2', 'h3', 'body', 'data', 'micro'] }],
      'max-w': [{ 'max-w': ['page'] }],
      h: [{ h: ['header'] }],
    },
  },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
