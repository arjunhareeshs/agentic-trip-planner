/**
 * stores/chatStore.ts — Lightweight reactive store for chat state.
 *
 * Uses React context + useReducer for global chat state management.
 * Manages: messages, session, images gallery, map data, loading state.
 */

import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { ChatMessage, ImageResult, MapData } from "../services/api";

// ── State Shape ───────────────────────────────────────────────────────────

export interface ChatState {
  sessionId: string | null;
  messages: ChatMessage[];
  galleryImages: ImageResult[];
  mapData: MapData | null;
  isLoading: boolean;
  isConnected: boolean;
  error: string | null;
}

const initialState: ChatState = {
  sessionId: null,
  messages: [],
  galleryImages: [],
  mapData: null,
  isLoading: false,
  isConnected: false,
  error: null,
};

// ── Actions ───────────────────────────────────────────────────────────────

type ChatAction =
  | { type: "SET_SESSION"; sessionId: string }
  | { type: "ADD_MESSAGE"; message: ChatMessage }
  | { type: "UPDATE_LAST_BOT_MESSAGE"; text: string }
  | { type: "ADD_IMAGES_TO_GALLERY"; images: ImageResult[] }
  | { type: "SET_MAP_DATA"; data: MapData }
  | { type: "SET_LOADING"; loading: boolean }
  | { type: "SET_CONNECTED"; connected: boolean }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "APPEND_BOT_TEXT"; text: string }
  | { type: "CLEAR_MESSAGES" };

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "SET_SESSION":
      return { ...state, sessionId: action.sessionId };

    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.message] };

    case "UPDATE_LAST_BOT_MESSAGE": {
      const msgs = [...state.messages];
      const lastBotIdx = msgs.findLastIndex((m) => m.sender === "bot");
      if (lastBotIdx >= 0) {
        msgs[lastBotIdx] = { ...msgs[lastBotIdx], text: action.text };
      }
      return { ...state, messages: msgs };
    }

    case "APPEND_BOT_TEXT": {
      const msgs = [...state.messages];
      const idx = msgs.findLastIndex((m) => m.sender === "bot");
      if (idx >= 0) {
        msgs[idx] = { ...msgs[idx], text: msgs[idx].text + action.text };
      }
      return { ...state, messages: msgs };
    }

    case "ADD_IMAGES_TO_GALLERY":
      return {
        ...state,
        galleryImages: [...state.galleryImages, ...action.images],
      };

    case "SET_MAP_DATA":
      return { ...state, mapData: action.data };

    case "SET_LOADING":
      return { ...state, isLoading: action.loading };

    case "SET_CONNECTED":
      return { ...state, isConnected: action.connected };

    case "SET_ERROR":
      return { ...state, error: action.error };

    case "CLEAR_MESSAGES":
      return { ...state, messages: [], galleryImages: [], mapData: null };

    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────

const ChatStateContext = createContext<ChatState>(initialState);
const ChatDispatchContext = createContext<Dispatch<ChatAction>>(() => {});

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  return (
    <ChatStateContext.Provider value={state}>
      <ChatDispatchContext.Provider value={dispatch}>
        {children}
      </ChatDispatchContext.Provider>
    </ChatStateContext.Provider>
  );
}

export function useChatState() {
  return useContext(ChatStateContext);
}

export function useChatDispatch() {
  return useContext(ChatDispatchContext);
}
