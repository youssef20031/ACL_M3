import { useCallback } from 'react';
import { apiService, handleApiError, StreamUnavailableError, type QueryResponse } from '../services/api';
import { useAppStore, type Message } from '../store/appStore';

// One answer streams at a time. Kept outside React so Clear chat can stop it even after the
// assistant page was left and opened again (the answer keeps arriving into the store meanwhile).
let inflight: AbortController | null = null;
let counter = 0;

/** Stops the answer that is streaming, if any. */
export function abortAnswer() {
  inflight?.abort();
  inflight = null;
}

interface AskOptions {
  onDone?: (response: QueryResponse) => void;
  onError?: (message: string) => void;
}

/**
 * Sends a question and streams the answer into the chat: the cards first (as soon as the
 * backend has them), then the text piece by piece (one store update per animation frame), then
 * the final response. Falls back to POST /api/query when the backend has no stream endpoint.
 */
export function useAssistantAsk({ onDone, onError }: AskOptions = {}) {
  return useCallback(
    async (question: string) => {
      const store = useAppStore.getState();
      if (store.chatHistory.some((m) => m.streaming)) return;
      const history = store.chatHistory.map((m) => ({ role: m.role, content: m.content }));
      const id = `answer-${Date.now().toString(36)}-${(counter++).toString(36)}`;
      store.addMessage({ role: 'user', content: question, timestamp: Date.now() });
      store.addMessage({ id, role: 'assistant', content: '', timestamp: Date.now(), streaming: true, cards: [], followUps: [] });

      const update = (patch: Partial<Message>) => useAppStore.getState().updateMessage(id, patch);
      const controller = new AbortController();
      inflight = controller;
      let text = '';
      let frame = 0;
      let finished = false;
      const finish = (response: QueryResponse) => {
        finished = true;
        cancelAnimationFrame(frame);
        update({ content: response.answer, cards: response.cards ?? [], followUps: response.follow_ups ?? [], streaming: false });
        onDone?.(response);
      };

      try {
        const body = {
          question,
          model: store.selectedModel,
          retrieval_method: store.retrievalMethod,
          embedding_model: store.embeddingModel,
          is_first_message: history.length === 0,
          chat_history: history,
        };
        try {
          await apiService.streamQuery(
            body,
            (event) => {
              switch (event.event) {
                case 'cards':
                  update({ cards: event.data.cards, followUps: event.data.follow_ups });
                  break;
                case 'delta':
                  text += event.data.text;
                  if (!frame) {
                    frame = requestAnimationFrame(() => {
                      frame = 0;
                      update({ content: text });
                    });
                  }
                  break;
                case 'answer':
                  text = event.data.text;
                  break;
                case 'done':
                  finish(event.data);
                  break;
                case 'error':
                  throw new Error(event.data.detail);
              }
            },
            controller.signal,
          );
        } catch (err) {
          if (!(err instanceof StreamUnavailableError)) throw err;
          finish(
            await apiService.queryFPL(question, body.model, body.retrieval_method, body.embedding_model, body.is_first_message, history),
          );
        }
        if (!finished) throw new Error('The answer was interrupted. Please try again.');
      } catch (err) {
        if (controller.signal.aborted) return; // Clear chat: the message is gone anyway
        cancelAnimationFrame(frame);
        const message = handleApiError(err);
        update({ content: `Error: ${message}`, streaming: false, cards: [], followUps: [] });
        onError?.(message);
      } finally {
        if (inflight === controller) inflight = null;
      }
    },
    [onDone, onError],
  );
}
