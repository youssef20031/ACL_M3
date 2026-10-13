import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { useAppStore } from '../store/appStore';

/**
 * Polls /health every 30 s and mirrors it into the store (pages read
 * neo4jConnected etc. from there). Shares the ['health'] key with Settings.
 */
export function useHealthSync() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (!health) return;
    const s = useAppStore.getState();
    s.setNeo4jConnected(health.neo4j === 'connected');
    s.setNeo4jStats(health.neo4j_stats ?? null);
    s.setEmbeddingsBuilt(health.embeddings_built);
    s.setEmbeddingCount(health.embedding_count);
    if (health.ml_available !== undefined) s.setMlAvailable(health.ml_available);
    if (health.ml_model_type !== undefined) s.setMlModelType(health.ml_model_type);
  }, [health]);

  return health;
}
