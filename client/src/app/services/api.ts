/**
 * services/api.ts — API service layer for communicating with the Trip Planner backend.
 *
 * All backend calls go through this module. Handles:
 *   - Chat messages (SSE streaming + sync fallback)
 *   - Session management
 *   - Image search
 *   - Geocoding
 *   - File uploads (images, PDFs)
 */

const API_BASE = "/api";

// ── Types ─────────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  text: string;
  sender: "user" | "bot";
  timestamp: Date;
  images?: ImageResult[];
  fileUrl?: string;
  fileType?: "image" | "pdf";
}

export interface ImageResult {
  title: string;
  image_url: string;
  source_url: string;
}

export interface MapData {
  lat?: number;
  lng?: number;
  place?: string;
}

export interface AgentEvent {
  event_type: "tool_call" | "tool_response" | "code_exec" | "code_result";
  agent: string;
  tool?: string;
  args?: Record<string, unknown>;
  result?: string;
  code?: string;
  output?: string;
  timestamp?: number;
}

export interface SSEChunk {
  type: "text" | "image" | "map" | "status" | "done" | "error" | "agent_event";
  content?: string;
  data?: Record<string, unknown>;
  session_id?: string;
  images?: ImageResult[];
  map_data?: MapData | null;
}

export interface UploadResult {
  file_id: string;
  filename: string;
  file_type: string;
  file_url: string;
  message: string;
}

// ── Chat API ──────────────────────────────────────────────────────────────

/**
 * Send a chat message and receive SSE-streamed response chunks.
 * Calls `onChunk` for each incoming SSE event.
 * Returns a cleanup function to abort the stream.
 */
export function chatStream(
  message: string,
  sessionId: string | null,
  onChunk: (chunk: SSEChunk) => void,
  onError?: (err: Error) => void
): AbortController {
  const controller = new AbortController();

  const body = JSON.stringify({
    message,
    session_id: sessionId,
    user_id: "default_user",
  });

  fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        throw new Error(`Chat API error: ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;

          try {
            const data = JSON.parse(trimmed.slice(6));
            onChunk(data as SSEChunk);
          } catch {
            // Skip malformed JSON
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim().startsWith("data: ")) {
        try {
          const data = JSON.parse(buffer.trim().slice(6));
          onChunk(data as SSEChunk);
        } catch {
          // Skip
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError?.(err);
      }
    });

  return controller;
}

/**
 * Send a chat message and wait for the full response (non-streaming).
 */
export async function chatSync(
  message: string,
  sessionId: string | null
): Promise<{
  session_id: string;
  response: string;
  images: ImageResult[];
  map_data: MapData | null;
}> {
  const res = await fetch(`${API_BASE}/chat/sync`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_id: "default_user",
    }),
  });

  if (!res.ok) throw new Error(`Chat sync error: ${res.status}`);
  return res.json();
}

// ── Session API ───────────────────────────────────────────────────────────

export async function createSession(): Promise<{
  session_id: string;
  user_id: string;
}> {
  const res = await fetch(`${API_BASE}/session`, { method: "POST" });
  if (!res.ok) throw new Error(`Session create error: ${res.status}`);
  return res.json();
}

// ── Image Search API ──────────────────────────────────────────────────────

export async function searchImages(
  query: string,
  max: number = 3
): Promise<ImageResult[]> {
  const params = new URLSearchParams({ q: query, max: String(max) });
  const res = await fetch(`${API_BASE}/images/search?${params}`);
  if (!res.ok) throw new Error(`Image search error: ${res.status}`);
  const data = await res.json();
  return data.images || [];
}

// ── Geocode API ───────────────────────────────────────────────────────────

export async function geocodePlace(
  place: string
): Promise<MapData> {
  const params = new URLSearchParams({ place });
  const res = await fetch(`${API_BASE}/geocode?${params}`);
  if (!res.ok) throw new Error(`Geocode error: ${res.status}`);
  return res.json();
}

// ── File Upload API ───────────────────────────────────────────────────────

export async function uploadFile(
  file: File,
  sessionId: string,
  message: string = ""
): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);
  formData.append("message", message);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error(`Upload error: ${res.status}`);
  return res.json();
}

// ── Voice Processing API ──────────────────────────────────────────────────

export async function processVoice(
  audioBlob: Blob,
  sessionId: string | null
): Promise<{ audio: Blob; transcript: string; response: string; sessionId: string }> {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  if (sessionId) formData.append("session_id", sessionId);
  formData.append("user_id", "default_user");

  const res = await fetch(`${API_BASE}/voice/process`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Voice process error: ${res.status} - ${errText}`);
  }

  // Response is the MP3 audio file
  const responseAudioBlob = await res.blob();

  // Custom headers contain the transcript and LLM text
  const userTranscript = res.headers.get("X-User-Transcript") || "";
  const botResponse = res.headers.get("X-Bot-Response") || "";
  const newSessionId = res.headers.get("X-Session-Id") || sessionId || "";

  return {
    audio: responseAudioBlob,
    transcript: decodeURIComponent(escape(userTranscript)),
    response: decodeURIComponent(escape(botResponse)),
    sessionId: newSessionId,
  };
}

// ── Health Check ──────────────────────────────────────────────────────────

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
