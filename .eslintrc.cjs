// ESLint 8 (eslintrc) config for the React/Vite frontend: `npm run lint` checks src/ and vite.config.ts.
// Python, data and build output have no .ts/.tsx files; dot-folders (.venv, .claude) are skipped by default.
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', 'node_modules', 'legacy', 'public', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    // Destructuring a prop out before spreading the rest (`{ node: _n, ...rest }`) is how the
    // markdown renderer keeps react-markdown's `node` off DOM elements.
    '@typescript-eslint/no-unused-vars': ['error', { ignoreRestSiblings: true }],
  },
  overrides: [
    {
      // Shared modules that keep small helpers, hooks and contexts next to the components using
      // them. The rule only affects dev hot reload (these files reload the page instead of
      // hot-swapping), so splitting them isn't worth the churn.
      files: [
        'src/components/assistant/cards/AnswerCards.tsx',
        'src/components/assistant/cards/shared.tsx',
        'src/components/shell/SiteHeader.tsx',
        'src/components/ui/Badges.tsx',
        'src/components/ui/Button.tsx',
      ],
      rules: { 'react-refresh/only-export-components': 'off' },
    },
  ],
}
