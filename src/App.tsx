import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { QAAssistant } from './pages/QAAssistant';
import { Trivia } from './pages/Trivia';
import { PlayerSearch } from './pages/PlayerSearch';
import { PlayerComparison } from './pages/PlayerComparison';
import { Settings } from './pages/Settings';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/qa" replace />} />
            <Route path="qa" element={<QAAssistant />} />
            <Route path="trivia" element={<Trivia />} />
            <Route path="search" element={<PlayerSearch />} />
            <Route path="compare" element={<PlayerComparison />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;