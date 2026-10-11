import { lazy, Suspense, useEffect, type ReactNode } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import './App.css';

// Route-level code splitting: recharts (Search/Compare), react-markdown + framer-motion (Q&A/Trivia)
// are only downloaded with the page that needs them.
const loadQAAssistant = () => import('./pages/QAAssistant').then((m) => ({ default: m.QAAssistant }));
const loadTrivia = () => import('./pages/Trivia').then((m) => ({ default: m.Trivia }));
const loadPlayerSearch = () => import('./pages/PlayerSearch').then((m) => ({ default: m.PlayerSearch }));
const loadPlayerComparison = () => import('./pages/PlayerComparison').then((m) => ({ default: m.PlayerComparison }));
const loadPredictions = () => import('./pages/Predictions').then((m) => ({ default: m.Predictions }));
const loadSettings = () => import('./pages/Settings').then((m) => ({ default: m.Settings }));

const QAAssistant = lazy(loadQAAssistant);
const Trivia = lazy(loadTrivia);
const PlayerSearch = lazy(loadPlayerSearch);
const PlayerComparison = lazy(loadPlayerComparison);
const Predictions = lazy(loadPredictions);
const Settings = lazy(loadSettings);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Suspense boundary per route so the Layout (sidebar/header) stays mounted while a page chunk loads.
function Page({ children }: { children: ReactNode }) {
  return <Suspense fallback={<div className="p-6 text-sm opacity-60">Loading…</div>}>{children}</Suspense>;
}

function App() {
  // Warm the remaining route chunks once the browser is idle so later navigation is instant.
  useEffect(() => {
    const preload = () => {
      [loadQAAssistant, loadPredictions, loadTrivia, loadPlayerSearch, loadPlayerComparison, loadSettings].forEach((load) => {
        void load().catch(() => undefined);
      });
    };
    const w = window as Window & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number;
      cancelIdleCallback?: (id: number) => void;
    };
    if (w.requestIdleCallback) {
      const id = w.requestIdleCallback(preload, { timeout: 4000 });
      return () => w.cancelIdleCallback?.(id);
    }
    const id = window.setTimeout(preload, 2000);
    return () => window.clearTimeout(id);
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/qa" replace />} />
            <Route path="qa" element={<Page><QAAssistant /></Page>} />
            <Route path="predictions" element={<Page><Predictions /></Page>} />
            <Route path="trivia" element={<Page><Trivia /></Page>} />
            <Route path="search" element={<Page><PlayerSearch /></Page>} />
            <Route path="compare" element={<Page><PlayerComparison /></Page>} />
            <Route path="settings" element={<Page><Settings /></Page>} />
          </Route>
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
