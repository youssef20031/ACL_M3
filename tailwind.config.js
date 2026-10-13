/** @type {import('tailwindcss').Config} */

// Colours resolve to the CSS variables in src/styles/tokens.css, so themes swap
// values without touching components. `<alpha-value>` keeps opacity modifiers working.
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: token('bg'),
        surface: { DEFAULT: token('surface'), 2: token('surface-2') },
        brand: { DEFAULT: token('brand'), 2: token('brand-2') },
        line: { DEFAULT: token('border'), strong: token('border-strong') },
        fg: { DEFAULT: token('text'), muted: token('text-muted'), subtle: token('text-subtle') },
        accent: { DEFAULT: token('accent'), 2: token('accent-2'), on: token('on-accent') },
        success: token('success'),
        warning: token('warning'),
        danger: token('danger'),
        info: token('info'),
        pos: { gk: token('pos-gk'), def: token('pos-def'), mid: token('pos-mid'), fwd: token('pos-fwd') },
        fdr: { 1: token('fdr-1'), 2: token('fdr-2'), 3: token('fdr-3'), 4: token('fdr-4'), 5: token('fdr-5') },
      },
      fontFamily: {
        display: ['"Outfit Variable"', 'Outfit', 'system-ui', 'sans-serif'],
        sans: ['"Inter Variable"', 'Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        // [size, { lineHeight, letterSpacing }] — display sizes use Outfit, the rest Inter
        display: ['2.5rem', { lineHeight: '2.75rem', letterSpacing: '-0.02em' }],
        h1: ['2rem', { lineHeight: '2.375rem', letterSpacing: '-0.015em' }],
        h2: ['1.5rem', { lineHeight: '1.875rem', letterSpacing: '-0.01em' }],
        h3: ['1.125rem', { lineHeight: '1.5rem' }],
        body: ['0.9375rem', { lineHeight: '1.375rem' }],
        data: ['0.8125rem', { lineHeight: '1.125rem' }], // dense tables
        micro: ['0.6875rem', { lineHeight: '0.875rem', letterSpacing: '0.04em' }],
      },
      borderRadius: {
        sm: 'var(--radius-sm)',
        md: 'var(--radius-md)',
        lg: 'var(--radius-lg)',
      },
      boxShadow: {
        card: 'var(--shadow-card)',
        pop: 'var(--shadow-pop)',
      },
      maxWidth: {
        page: 'var(--page-max)',
      },
      height: {
        header: 'var(--header-h)',
      },
    },
  },
  plugins: [],
};
