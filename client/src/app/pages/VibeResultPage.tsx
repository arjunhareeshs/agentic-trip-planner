/**
 * VibeResultPage.tsx — The main chat + map + gallery page (route: /result).
 *
 * Layout:
 *   Left 70%  : Chatbot with text input, image/PDF upload, streaming responses
 *   Right 30% : Map view (top) + Image gallery (bottom)
 *
 * Integration points:
 *   - SSE streaming chat via /api/chat
 *   - Image/PDF upload via /api/upload
 *   - Map data extraction from bot responses
 *   - Image gallery populated from bot search results
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Plus, Mic, Maximize2, Image, FileText, X, Send, Loader2, Square } from "lucide-react";
import {
  chatStream,
  uploadFile,
  processVoice,
  geocodePlace,
  type ChatMessage,
  type ImageResult,
  type MapData,
  type SSEChunk,
  type AgentEvent,
  healthCheck,
} from "../services/api";
import AgentActivityPanel from "../components/AgentActivityPanel";

// ── Component ─────────────────────────────────────────────────────────────

export default function VibeResultPage() {
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome-1",
      text: "Welcome! I'm your AI trip planner. Tell me about your dream travel experience — what's the vibe you're looking for? 🌍",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // File upload state
  const [showUploadMenu, setShowUploadMenu] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadingFile, setUploadingFile] = useState(false);

  // Gallery + Map state
  const [galleryImages, setGalleryImages] = useState<ImageResult[]>([]);
  const [mapData, setMapData] = useState<MapData | null>({ lat: 48.8566, lng: 2.3522, place: "Paris, France" });
  const [mapImageUrl, setMapImageUrl] = useState<string>(
    "https://www.openstreetmap.org/export/embed.html?bbox=2.2522,48.7566,2.4522,48.9566&layer=mapnik&marker=48.8566,2.3522"
  );
  const [mapLabel, setMapLabel] = useState("Paris, France");
  const [mapPlaces, setMapPlaces] = useState<Array<{ name: string; lat: number; lng: number }>>([
    { name: "Paris, France", lat: 48.8566, lng: 2.3522 }
  ]);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);

  // Track inline image URLs already synced to gallery (prevents re-render loops)
  const syncedImageUrls = useRef<Set<string>>(new Set());

  // Agent activity debug state
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([]);

  // Voice-to-Voice State
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState<"idle" | "listening" | "processing" | "speaking">("idle");
  const [audioLevel, setAudioLevel] = useState(0);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Voice Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const volumeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const noiseFloorRef = useRef<number>(0);         // Calibrated ambient noise level
  const showVoiceModalRef = useRef(false);          // Non-stale ref for async callbacks

  // ── Effects ───────────────────────────────────────────────────────────

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Health check on mount
  useEffect(() => {
    healthCheck().then(setIsConnected);
  }, []);

  // Cleanup voice resources on unmount
  useEffect(() => {
    return () => {
      stopVoiceActivity();
      if (audioElementRef.current) {
        audioElementRef.current.pause();
        audioElementRef.current = null;
      }
    };
  }, []);

  // ── Handlers ──────────────────────────────────────────────────────────

  const handleSendMessage = useCallback(() => {
    const text = inputValue.trim();
    if (!text || isLoading) return;

    // Add user message
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      text,
      sender: "user",
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue("");
    setIsLoading(true);

    // Create placeholder bot message for streaming
    const botId = (Date.now() + 1).toString();
    const botMsg: ChatMessage = {
      id: botId,
      text: "",
      sender: "bot",
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, botMsg]);

    // Stream the response
    const controller = chatStream(
      text,
      sessionId,
      (chunk: SSEChunk) => {
        switch (chunk.type) {
          case "text":
            setMessages((prev) => {
              const updated = [...prev];
              const idx = updated.findIndex((m) => m.id === botId);
              if (idx >= 0) {
                updated[idx] = {
                  ...updated[idx],
                  text: updated[idx].text + (chunk.content || ""),
                };
              }
              return updated;
            });
            break;

          case "image":
            if (chunk.data) {
              const imgData = chunk.data as unknown as ImageResult;
              setGalleryImages((prev) => {
                if (prev.some((i) => i.image_url === imgData.image_url)) return prev;
                return [...prev, imgData];
              });
            }
            break;

          case "map":
            if (chunk.data) {
              const md = chunk.data as unknown as MapData;
              setMapData(md);
              if (md.place) {
                setMapLabel(md.place);
                if (md.lat && md.lng) {
                  // Accumulate places for the map
                  setMapPlaces((prev) => {
                    const exists = prev.some(
                      (p) => p.lat === md.lat && p.lng === md.lng
                    );
                    const updated = exists
                      ? prev
                      : [...prev, { name: md.place!, lat: md.lat!, lng: md.lng! }];
                    // Build map URL with all markers
                    const lastPlace = updated[updated.length - 1];
                    const delta = updated.length > 1 ? 0.5 : 0.1;
                    const bbox = `${lastPlace.lng - delta},${lastPlace.lat - delta},${lastPlace.lng + delta},${lastPlace.lat + delta}`;
                    setMapImageUrl(
                      `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lastPlace.lat},${lastPlace.lng}`
                    );
                    return updated;
                  });
                } else {
                  // No coordinates — geocode the place name to get lat/lng
                  geocodePlace(md.place).then((geo) => {
                    if (geo && geo.lat && geo.lng) {
                      setMapData({ ...md, lat: geo.lat, lng: geo.lng });
                      setMapPlaces((prev) => {
                        const exists = prev.some(
                          (p) => p.lat === geo.lat && p.lng === geo.lng
                        );
                        const updated = exists
                          ? prev
                          : [...prev, { name: md.place!, lat: geo.lat!, lng: geo.lng! }];
                        const lastPlace = updated[updated.length - 1];
                        const delta = updated.length > 1 ? 0.5 : 0.1;
                        const bbox = `${lastPlace.lng - delta},${lastPlace.lat - delta},${lastPlace.lng + delta},${lastPlace.lat + delta}`;
                        setMapImageUrl(
                          `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lastPlace.lat},${lastPlace.lng}`
                        );
                        return updated;
                      });
                    }
                  }).catch(() => {});
                }
              }
            }
            break;

          case "done":
            setIsLoading(false);
            if (chunk.session_id) setSessionId(chunk.session_id);
            if (chunk.images && chunk.images.length > 0) {
              setGalleryImages((prev) => {
                const newImages = chunk.images!.filter(
                  (img) => !prev.some((p) => p.image_url === img.image_url)
                );
                return [...prev, ...newImages];
              });
            }
            break;

          case "error":
            setIsLoading(false);
            setMessages((prev) => {
              const updated = [...prev];
              const idx = updated.findIndex((m) => m.id === botId);
              if (idx >= 0) {
                updated[idx] = {
                  ...updated[idx],
                  text: `Sorry, something went wrong: ${chunk.content || "Unknown error"}`,
                };
              }
              return updated;
            });
            break;

          case "status":
            break;

          case "agent_event":
            if (chunk.data) {
              const evt = chunk.data as unknown as AgentEvent;
              evt.timestamp = Date.now();
              setAgentEvents((prev) => [...prev, evt]);
            }
            break;
        }
      },
      (err) => {
        setIsLoading(false);
        setMessages((prev) => {
          const updated = [...prev];
          const idx = updated.findIndex((m) => m.id === botId);
          if (idx >= 0) {
            updated[idx] = {
              ...updated[idx],
              text: isConnected
                ? `Connection error: ${err.message}. Please try again.`
                : "Backend is not connected. Please start the server with: uvicorn main:app --reload --port 8005",
            };
          }
          return updated;
        });
      }
    );

    abortRef.current = controller;
  }, [inputValue, isLoading, sessionId, isConnected]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // ── File Upload ───────────────────────────────────────────────────────

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setShowUploadMenu(false);
    setUploadingFile(true);

    try {
      const result = await uploadFile(file, sessionId || "", inputValue);
      const isImage = result.file_type === "image";

      const userMsg: ChatMessage = {
        id: Date.now().toString(),
        text: inputValue.trim()
          ? `${inputValue.trim()}\n\n📎 ${result.filename}`
          : `📎 Uploaded: ${result.filename}`,
        sender: "user",
        timestamp: new Date(),
        fileUrl: result.file_url,
        fileType: result.file_type as "image" | "pdf",
      };
      setMessages((prev) => [...prev, userMsg]);
      setInputValue("");

      if (isImage) {
        setGalleryImages((prev) => [
          ...prev,
          { title: result.filename, image_url: result.file_url, source_url: "" },
        ]);
      }

      const analyzeMsg = isImage
        ? `I've uploaded an image: ${result.filename}. Please analyze it and suggest similar travel destinations.`
        : `I've uploaded a PDF document: ${result.filename}. Please review it for travel planning context.`;

      const botId = (Date.now() + 1).toString();
      setMessages((prev) => [
        ...prev,
        { id: botId, text: "", sender: "bot" as const, timestamp: new Date() },
      ]);
      setIsLoading(true);

      chatStream(
        analyzeMsg,
        sessionId,
        (chunk: SSEChunk) => {
          if (chunk.type === "text") {
            setMessages((prev) => {
              const updated = [...prev];
              const idx = updated.findIndex((m) => m.id === botId);
              if (idx >= 0) {
                updated[idx] = {
                  ...updated[idx],
                  text: updated[idx].text + (chunk.content || ""),
                };
              }
              return updated;
            });
          } else if (chunk.type === "image" && chunk.data) {
            const imgData = chunk.data as unknown as ImageResult;
            setGalleryImages((prev) => {
              if (prev.some((i) => i.image_url === imgData.image_url)) return prev;
              return [...prev, imgData];
            });
          } else if (chunk.type === "map" && chunk.data) {
            const md = chunk.data as unknown as MapData;
            setMapData(md);
            if (md.place) {
              setMapLabel(md.place);
              if (md.lat && md.lng) {
                setMapPlaces((prev) => {
                  const exists = prev.some(
                    (p) => p.lat === md.lat && p.lng === md.lng
                  );
                  const updated = exists
                    ? prev
                    : [...prev, { name: md.place!, lat: md.lat!, lng: md.lng! }];
                  const lastPlace = updated[updated.length - 1];
                  const delta = updated.length > 1 ? 0.5 : 0.1;
                  const bbox = `${lastPlace.lng - delta},${lastPlace.lat - delta},${lastPlace.lng + delta},${lastPlace.lat + delta}`;
                  setMapImageUrl(
                    `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lastPlace.lat},${lastPlace.lng}`
                  );
                  return updated;
                });
              } else {
                // No coordinates — geocode the place name
                geocodePlace(md.place).then((geo) => {
                  if (geo && geo.lat && geo.lng) {
                    setMapData({ ...md, lat: geo.lat, lng: geo.lng });
                    setMapPlaces((prev) => {
                      const exists = prev.some(
                        (p) => p.lat === geo.lat && p.lng === geo.lng
                      );
                      const updated = exists
                        ? prev
                        : [...prev, { name: md.place!, lat: geo.lat!, lng: geo.lng! }];
                      const lastPlace = updated[updated.length - 1];
                      const delta = updated.length > 1 ? 0.5 : 0.1;
                      const bbox = `${lastPlace.lng - delta},${lastPlace.lat - delta},${lastPlace.lng + delta},${lastPlace.lat + delta}`;
                      setMapImageUrl(
                        `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${lastPlace.lat},${lastPlace.lng}`
                      );
                      return updated;
                    });
                  }
                }).catch(() => {});
              }
            }
          } else if (chunk.type === "done") {
            setIsLoading(false);
            if (chunk.session_id) setSessionId(chunk.session_id);
          } else if (chunk.type === "error") {
            setIsLoading(false);
          } else if (chunk.type === "agent_event" && chunk.data) {
            const evt = chunk.data as unknown as AgentEvent;
            evt.timestamp = Date.now();
            setAgentEvents((prev) => [...prev, evt]);
          }
        },
        () => setIsLoading(false)
      );
    } catch (err) {
      console.error("Upload failed:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          text: "Failed to upload file. Please try again.",
          sender: "bot" as const,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setUploadingFile(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const triggerFileUpload = (accept: string) => {
    setShowUploadMenu(false);
    if (fileInputRef.current) {
      fileInputRef.current.accept = accept;
      fileInputRef.current.click();
    }
  };

  // ── Voice Activity Detection (VAD) & Voice Modal ─────────────────────

  const startVoiceActivity = async () => {
    try {
      setVoiceStatus("listening");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);
      analyserRef.current = analyser;

      const bufferLength = analyser.fftSize;
      const timeDomainData = new Float32Array(bufferLength);

      const getRMS = (): number => {
        analyser.getFloatTimeDomainData(timeDomainData);
        let sumSq = 0;
        for (let i = 0; i < bufferLength; i++) {
          sumSq += timeDomainData[i] * timeDomainData[i];
        }
        return Math.sqrt(sumSq / bufferLength);
      };

      // ── Noise floor calibration (~500ms) ─────────────────────────
      const calibrationSamples: number[] = [];
      await new Promise<void>((resolve) => {
        const calInterval = setInterval(() => {
          calibrationSamples.push(getRMS());
          if (calibrationSamples.length >= 10) {
            clearInterval(calInterval);
            resolve();
          }
        }, 50);
      });
      const avgNoise = calibrationSamples.reduce((a, b) => a + b, 0) / calibrationSamples.length;
      noiseFloorRef.current = avgNoise;

      const SPEECH_THRESHOLD = Math.max(avgNoise * 2.5, avgNoise + 0.006, 0.006);
      const SPEECH_FRAMES_NEEDED = 3;     // 150ms to confirm speech start
      const SILENCE_AFTER_SPEECH_MS = 1200; // 1.2s since LAST loud frame → stop
      const MAX_RECORD_MS = 30000;
      const NO_SPEECH_TIMEOUT_MS = 10000; // 10s with zero speech → restart

      console.log(
        `[VAD] Calibrated — noise: ${avgNoise.toFixed(5)}, threshold: ${SPEECH_THRESHOLD.toFixed(5)}`
      );

      // ── Timestamp-based VAD (immune to single noise spikes) ──────
      let isSpeaking = false;
      let speechFrames = 0;            // pre-speech confirmation counter
      let lastAboveThresholdTime = 0;  // timestamp of last loud frame
      let speechStartTime = 0;
      const listenStartTime = Date.now();

      volumeIntervalRef.current = setInterval(() => {
        const rms = getRMS();
        const normalizedLevel = Math.min(100, (rms / Math.max(SPEECH_THRESHOLD * 2, 0.02)) * 50);
        setAudioLevel(normalizedLevel);

        if (rms > SPEECH_THRESHOLD) {
          lastAboveThresholdTime = Date.now();

          if (!isSpeaking) {
            speechFrames++;
            if (speechFrames >= SPEECH_FRAMES_NEEDED) {
              isSpeaking = true;
              speechStartTime = Date.now();
              console.log(`[VAD] Speech confirmed (RMS: ${rms.toFixed(5)})`);
            }
          }
        } else {
          // Reset pre-speech counter (we only need consecutive frames to START)
          if (!isSpeaking) {
            speechFrames = 0;
          }
          // Note: we do NOT reset lastAboveThresholdTime here.
          // A single noise spike just moves the deadline forward by ~50ms,
          // it doesn't reset a 1.2s counter like the old frame-counting did.
        }

        // ── Silence detection (timestamp-based, not frame-counting) ──
        // Once speaking, if enough time has passed since the LAST loud frame, stop.
        // This is robust: a random noise spike at T just moves deadline to T + 1200ms.
        // With frame-counting, a spike at frame 24/25 reset the entire 1.25s counter.
        if (isSpeaking && Date.now() - lastAboveThresholdTime > SILENCE_AFTER_SPEECH_MS) {
          console.log("[VAD] Silence confirmed → processing");
          isSpeaking = false;
          stopRecordingAndProcess();
          return;
        }

        // No-speech timeout → restart listener
        if (!isSpeaking && Date.now() - listenStartTime > NO_SPEECH_TIMEOUT_MS) {
          console.log("[VAD] No speech detected — restarting");
          if (volumeIntervalRef.current) {
            clearInterval(volumeIntervalRef.current);
            volumeIntervalRef.current = null;
          }
          stopVoiceActivity();
          if (showVoiceModalRef.current) {
            startVoiceActivity();
          }
          return;
        }

        // Max duration safety
        if (isSpeaking && Date.now() - speechStartTime > MAX_RECORD_MS) {
          console.log("[VAD] Max duration → processing");
          isSpeaking = false;
          stopRecordingAndProcess();
        }
      }, 50);

      // ── Setup MediaRecorder ──────────────────────────────────────
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start(200);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setVoiceStatus("idle");
      setShowVoiceModal(false);
      showVoiceModalRef.current = false;
    }
  };

  const stopVoiceActivity = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (volumeIntervalRef.current) {
      clearInterval(volumeIntervalRef.current);
      volumeIntervalRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => { });
      audioContextRef.current = null;
    }
  };

  const restartListeningIfOpen = () => {
    if (showVoiceModalRef.current) {
      setVoiceStatus("listening");
      startVoiceActivity();
    } else {
      setVoiceStatus("idle");
    }
  };

  const stopRecordingAndProcess = async () => {
    // Stop the VAD interval first (prevent re-entry)
    if (volumeIntervalRef.current) {
      clearInterval(volumeIntervalRef.current);
      volumeIntervalRef.current = null;
    }
    setVoiceStatus("processing");
    setAudioLevel(0);

    const recorder = mediaRecorderRef.current;

    // If no recorder or already stopped, restart
    if (!recorder || recorder.state === "inactive") {
      stopVoiceActivity();
      restartListeningIfOpen();
      return;
    }

    // Use onstop event to guarantee all chunks are flushed
    recorder.onstop = async () => {
      // Clean up stream & audio context
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close().catch(() => { });
        audioContextRef.current = null;
      }

      if (audioChunksRef.current.length === 0) {
        console.log("[Voice] No audio chunks — restarting");
        restartListeningIfOpen();
        return;
      }

      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      audioChunksRef.current = [];

      try {
        const result = await processVoice(audioBlob, sessionId);

        if (result.transcript) {
          const userMsg: ChatMessage = {
            id: Date.now().toString(),
            text: result.transcript,
            sender: "user",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, userMsg]);
        }

        if (result.response) {
          const botMsg: ChatMessage = {
            id: (Date.now() + 1).toString(),
            text: result.response,
            sender: "bot",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, botMsg]);
        }

        if (result.sessionId) setSessionId(result.sessionId);

        // Play TTS audio
        if (result.audio && result.audio.size > 0) {
          setVoiceStatus("speaking");
          const audioUrl = URL.createObjectURL(result.audio);
          const audio = new Audio(audioUrl);
          audioElementRef.current = audio;

          audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            restartListeningIfOpen();
          };

          // If audio fails to load/play, don't get stuck — restart listening
          audio.onerror = () => {
            console.error("[Voice] Audio playback error");
            URL.revokeObjectURL(audioUrl);
            restartListeningIfOpen();
          };

          try {
            await audio.play();
          } catch (playErr) {
            console.error("[Voice] Audio play rejected:", playErr);
            URL.revokeObjectURL(audioUrl);
            restartListeningIfOpen();
          }
        } else {
          // No audio in response — go back to listening
          console.log("[Voice] No audio in response");
          restartListeningIfOpen();
        }
      } catch (err: any) {
        console.error("Voice processing error:", err);
        // Show "didn't catch that" as a bot message so user gets feedback
        if (err?.message?.includes("400")) {
          const hintMsg: ChatMessage = {
            id: Date.now().toString(),
            text: "I didn't catch that. Could you say it again?",
            sender: "bot",
            timestamp: new Date(),
          };
          setMessages((prev) => [...prev, hintMsg]);
        }
        // Always restart listening — never get stuck
        restartListeningIfOpen();
      }
    };

    // Trigger stop — fires one final dataavailable, then onstop
    recorder.stop();
  };

  const handleVoiceModalOpen = () => {
    setShowVoiceModal(true);
    showVoiceModalRef.current = true;
    startVoiceActivity();
  };

  const handleVoiceModalClose = () => {
    setShowVoiceModal(false);
    showVoiceModalRef.current = false;
    setVoiceStatus("idle");
    stopVoiceActivity();
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current = null;
    }
  };

  // ── Clean tool-call / internal text from bot messages ──────────────────

  const cleanBotText = (text: string): string => {
    // 1) Remove ```json { ... } ``` code blocks that contain tool params
    let cleaned = text.replace(/```(?:json|tool_code|python|)\s*\n?\s*\{[\s\S]*?\}\s*\n?\s*```/gi, "");

    // 2) Remove XML-style tool call wrappers
    cleaned = cleaned.replace(/<function_call>[\s\S]*?<\/function_call>/g, "");
    cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "");

    // 3) Remove bare JSON blocks that look like tool params
    cleaned = cleaned.replace(
      /\{\s*"(?:function_call|name|tool_name|tool_call|args|parameters|query|keyword|url|prompt)"[\s\S]*?\}(?:\s*\})?/g,
      ""
    );

    // 3b) Remove empty code block markers (leftover after stripping JSON)
    cleaned = cleaned.replace(/```(?:json|tool_code|python|)\s*```/gi, "");
    cleaned = cleaned.replace(/`{3,}\s*`{3,}/g, "");
    cleaned = cleaned.replace(/`\s*`/g, "");

    // 4) Filter out lines that narrate tool usage or reference internal names
    const internalNames = [
      "web_search", "scrape_page", "extract_web_content", "deep_web_scrape",
      "search_place_images", "get_weather_forecast", "geocode",
      "match_destinations", "filter_destinations", "get_destination_details",
      "list_all_destinations", "get_graph_stats",
      "preference_agent", "itinerary_agent", "booking_agent",
      "image_analysis_agent", "web_automation_agent",
      "trip_planner_orchestrator",
    ];
    const narrationSignals = [
      "let me", "i'll", "i will", "calling", "using", "invoke",
      "delegate", "transfer", "pass to", "send to", "call ",
      "searching for", "scraping", "extracting", "fetching",
      "start by", "going to", "need to",
    ];
    const processNarration = [
      "let me search", "let me scrape", "let me extract",
      "let me look up", "let me find", "let me check",
      "i'll start by searching", "i'll search for",
      "i'll scrape", "i'll extract", "i'll look up",
      "i notice the web extraction didn't work",
      "the tool returned", "the search returned",
      "based on my search results", "from the search results",
      "here are the search results",
      "i'll use", "let me use", "i need to call",
    ];

    const toolRefPattern = /^\s*\{\s*"(?:function_call|name|tool_name|query|keyword)/m;
    const toolCallLabel = /^(?:Tool call|Calling tool|Function call|Parameters):\s*/im;

    const lines = cleaned.split("\n");
    const filteredLines = lines.filter((line) => {
      const trimmed = line.trim();
      if (!trimmed) return true;

      // Skip lone backtick lines
      if (/^`+$/.test(trimmed)) return false;

      // Skip lines that are bare tool-call-like JSON
      if (toolRefPattern.test(trimmed)) return false;
      if (toolCallLabel.test(trimmed)) return false;

      const lower = trimmed.toLowerCase();

      // Skip general process narration (even without tool name)
      if (processNarration.some((p) => lower.includes(p))) return false;

      // Skip narration lines referencing internal tool/agent names
      const hasInternalRef = internalNames.some((n) => lower.includes(n));
      if (hasInternalRef) {
        const isNarration = narrationSignals.some((s) => lower.includes(s));
        if (isNarration) return false;
      }

      return true;
    });

    return filteredLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  };

  // ── Structured Markdown renderer for bot messages ─────────────────────

  const renderMessageContent = (msg: ChatMessage) => {
    const rawText = msg.sender === "bot" ? cleanBotText(msg.text) : msg.text;

    // Extract all inline images from markdown syntax (deduplicated)
    const imgRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    const seenUrls = new Set<string>();
    const inlineImages: { alt: string; url: string }[] = [];
    let imgMatch;
    while ((imgMatch = imgRegex.exec(rawText)) !== null) {
      const imgUrl = imgMatch[2];
      const imgAlt = imgMatch[1];
      // Skip duplicates within the same message
      if (seenUrls.has(imgUrl)) continue;
      seenUrls.add(imgUrl);
      inlineImages.push({ alt: imgAlt, url: imgUrl });
    }
    // Sync unique inline images to gallery (skip already-synced URLs to prevent re-render loops)
    if (inlineImages.length > 0) {
      const unsyncedImages = inlineImages.filter((img) => !syncedImageUrls.current.has(img.url));
      if (unsyncedImages.length > 0) {
        unsyncedImages.forEach((img) => syncedImageUrls.current.add(img.url));
        setTimeout(() => {
          setGalleryImages((prev) => {
            const existingUrls = new Set(prev.map((i) => i.image_url));
            const newImages = unsyncedImages
              .filter((img) => !existingUrls.has(img.url))
              .map((img) => ({ title: img.alt || "Image", image_url: img.url, source_url: "" }));
            return newImages.length > 0 ? [...prev, ...newImages] : prev;
          });
        }, 0);
      }
    }

    // Remove markdown image syntax from text for clean rendering
    const textWithoutImages = rawText.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, "").trim();

    // Parse markdown into structured elements
    const renderStructuredText = (text: string) => {
      const lines = text.split("\n");
      const elements: React.ReactNode[] = [];
      let listItems: string[] = [];
      let orderedListItems: string[] = [];
      let listType: "ul" | "ol" | null = null;
      let keyIdx = 0;

      const flushList = () => {
        if (listType === "ul" && listItems.length > 0) {
          elements.push(
            <ul key={`ul-${keyIdx++}`} className="list-disc list-inside space-y-1 my-2 ml-2">
              {listItems.map((item, j) => (
                <li key={j} className="text-slate-200 text-[12px] leading-relaxed">
                  {renderInlineFormatting(item)}
                </li>
              ))}
            </ul>
          );
          listItems = [];
        }
        if (listType === "ol" && orderedListItems.length > 0) {
          elements.push(
            <ol key={`ol-${keyIdx++}`} className="list-decimal list-inside space-y-1 my-2 ml-2">
              {orderedListItems.map((item, j) => (
                <li key={j} className="text-slate-200 text-[12px] leading-relaxed">
                  {renderInlineFormatting(item)}
                </li>
              ))}
            </ol>
          );
          orderedListItems = [];
        }
        listType = null;
      };

      const renderInlineFormatting = (line: string): React.ReactNode[] => {
        const parts: React.ReactNode[] = [];
        // Process bold, italic, links, and inline code
        const inlineRegex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[([^\]]+)\]\(([^)]+)\))/g;
        let lastIdx = 0;
        let inlineMatch;

        while ((inlineMatch = inlineRegex.exec(line)) !== null) {
          if (inlineMatch.index > lastIdx) {
            parts.push(line.slice(lastIdx, inlineMatch.index));
          }

          if (inlineMatch[2]) {
            // Bold
            parts.push(
              <strong key={`b-${keyIdx++}`} className="font-semibold text-white">
                {inlineMatch[2]}
              </strong>
            );
          } else if (inlineMatch[3]) {
            // Italic
            parts.push(
              <em key={`i-${keyIdx++}`} className="italic text-slate-300">
                {inlineMatch[3]}
              </em>
            );
          } else if (inlineMatch[4]) {
            // Inline code
            parts.push(
              <code
                key={`c-${keyIdx++}`}
                className="bg-white/10 text-emerald-300 px-1.5 py-0.5 rounded text-[11px] font-mono"
              >
                {inlineMatch[4]}
              </code>
            );
          } else if (inlineMatch[5] && inlineMatch[6]) {
            // Link
            parts.push(
              <a
                key={`a-${keyIdx++}`}
                href={inlineMatch[6]}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
              >
                {inlineMatch[5]}
              </a>
            );
          }

          lastIdx = inlineMatch.index + inlineMatch[0].length;
        }

        if (lastIdx < line.length) {
          parts.push(line.slice(lastIdx));
        }

        return parts.length > 0 ? parts : [line];
      };

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        // Skip empty lines (but flush any pending list)
        if (!trimmed) {
          flushList();
          continue;
        }

        // Headers
        if (trimmed.startsWith("### ")) {
          flushList();
          elements.push(
            <h3 key={`h3-${keyIdx++}`} className="text-[13px] font-semibold text-white mt-3 mb-1.5 tracking-wide">
              {renderInlineFormatting(trimmed.slice(4))}
            </h3>
          );
          continue;
        }
        if (trimmed.startsWith("## ")) {
          flushList();
          elements.push(
            <h2 key={`h2-${keyIdx++}`} className="text-[14px] font-bold text-white mt-4 mb-2 tracking-wide">
              {renderInlineFormatting(trimmed.slice(3))}
            </h2>
          );
          continue;
        }
        if (trimmed.startsWith("# ")) {
          flushList();
          elements.push(
            <h1 key={`h1-${keyIdx++}`} className="text-[15px] font-bold text-white mt-4 mb-2 tracking-wide">
              {renderInlineFormatting(trimmed.slice(2))}
            </h1>
          );
          continue;
        }

        // Horizontal rule
        if (/^[-*_]{3,}$/.test(trimmed)) {
          flushList();
          elements.push(
            <hr key={`hr-${keyIdx++}`} className="border-white/10 my-3" />
          );
          continue;
        }

        // Unordered list items
        if (/^[-*•]\s+/.test(trimmed)) {
          if (listType !== "ul") {
            flushList();
            listType = "ul";
          }
          listItems.push(trimmed.replace(/^[-*•]\s+/, ""));
          continue;
        }

        // Ordered list items
        if (/^\d+[.)]\s+/.test(trimmed)) {
          if (listType !== "ol") {
            flushList();
            listType = "ol";
          }
          orderedListItems.push(trimmed.replace(/^\d+[.)]\s+/, ""));
          continue;
        }

        // Blockquote
        if (trimmed.startsWith("> ")) {
          flushList();
          elements.push(
            <blockquote
              key={`bq-${keyIdx++}`}
              className="border-l-2 border-white/20 pl-3 my-2 text-slate-300 text-[12px] italic"
            >
              {renderInlineFormatting(trimmed.slice(2))}
            </blockquote>
          );
          continue;
        }

        // Regular paragraph
        flushList();
        elements.push(
          <p key={`p-${keyIdx++}`} className="text-[12px] leading-relaxed text-slate-200 my-1">
            {renderInlineFormatting(trimmed)}
          </p>
        );
      }

      flushList();
      return elements;
    };

    return (
      <div className="space-y-1">
        {/* File attachments */}
        {msg.fileUrl && msg.fileType === "image" && (
          <div className="rounded-lg overflow-hidden max-w-[200px] mb-2">
            <img
              src={msg.fileUrl}
              alt="Uploaded"
              className="w-full h-auto object-cover rounded-lg cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => setLightboxImage(msg.fileUrl!)}
            />
          </div>
        )}
        {msg.fileUrl && msg.fileType === "pdf" && (
          <div className="flex items-center gap-2 bg-white/10 rounded-lg p-2 mb-2">
            <FileText className="w-5 h-5 text-orange-400" />
            <span className="text-xs text-slate-300">PDF Document</span>
          </div>
        )}

        {/* Structured text content */}
        {textWithoutImages && (
          <div className="structured-content">
            {renderStructuredText(textWithoutImages)}
          </div>
        )}

        {/* Inline images from markdown */}
        {inlineImages.length > 0 && (
          <div className="grid grid-cols-2 gap-2 mt-3">
            {inlineImages.map((img, i) => (
              <div
                key={`inline-img-${i}`}
                className="rounded-lg overflow-hidden bg-zinc-900/50 border border-white/5 group"
              >
                <img
                  src={img.url}
                  alt={img.alt}
                  className="w-full h-32 object-cover cursor-pointer hover:opacity-80 transition-opacity group-hover:scale-105 transition-transform duration-500"
                  loading="lazy"
                  onClick={() => setLightboxImage(img.url)}
                  onError={(e) => {
                    const el = e.target as HTMLImageElement;
                    if (!el.dataset.retried) {
                      el.dataset.retried = "1";
                      el.src = `https://wsrv.nl/?url=${encodeURIComponent(img.url)}&w=400&output=jpg`;
                    } else {
                      el.closest("[class*='rounded-lg']")?.classList.add("hidden");
                    }
                  }}
                />
                {img.alt && (
                  <p className="text-[10px] text-white/40 p-1.5 truncate">{img.alt}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="bg-black text-slate-100 h-screen w-full flex flex-col font-display overflow-hidden">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileSelect}
      />

      {/* Agent Activity Debug Panel */}
      <AgentActivityPanel
        events={agentEvents}
        onClear={() => setAgentEvents([])}
      />

      <div className="relative flex-1 w-full overflow-hidden blooming-bg flex flex-col px-4 py-8 sm:px-6 sm:py-12">
        {/* Connection status indicator */}
        <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${isConnected ? "bg-emerald-400" : "bg-red-400"
              } animate-pulse`}
          />
          <span className="text-[10px] text-white/40 uppercase tracking-widest">
            {isConnected ? "Connected" : "Offline"}
          </span>
        </div>

        <div className="flex-1 flex w-full min-h-0 bg-transparent overflow-hidden">
          {/* ─── Left Section: Chat Container ─────────────────────────── */}
          <section className="w-[70%] flex flex-col relative border-r border-white/5">
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-8 scrollbar-hide">
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex items-end gap-3 sm:gap-4 max-w-[95%] ${message.sender === "user" ? "ml-auto justify-end" : ""
                      }`}
                  >
                    {message.sender === "bot" && (
                      <div className="size-6 sm:size-7 rounded-full bg-zinc-900 shrink-0 border border-white/5 overflow-hidden flex items-center justify-center">
                        <img
                          className="w-full h-full object-cover"
                          src="https://lh3.googleusercontent.com/aida-public/AB6AXuBj7swQudqrQqVvNaV9LMoz-g-oL7cJiMSk9J_Ti8Wtj9WPApv6tIbpkS0vWWKiFbSpsqi8pXQqZa1bvLzKBDofkz3hYGUTM_4-gpHi34NQDuVohd8tO6xBIkmGo__BCEppJZWTUFFPfmd-wscMF28EvtAuTzIAIvW-SI836TWhTRtpMS_2n6XvzqTiurXyEbqGvaXVlRD-Bp3M6Jr7hj_ZIv0d_TUWrx1SkozUL6xeI3K7wyPjzH_o-MOQWvKs-x5GpWwXv-Yzfrw"
                          alt="Bot"
                        />
                      </div>
                    )}
                    <div className="flex flex-col gap-1.5">
                      <div
                        className={`px-3 py-2.5 rounded-2xl text-[12px] sm:text-[13px] leading-relaxed ${message.sender === "user"
                          ? "rounded-br-none bg-white/5 text-white"
                          : "rounded-bl-none bg-zinc-900/40 text-slate-200"
                          }`}
                      >
                        {message.sender === "bot"
                          ? renderMessageContent(message)
                          : message.text}

                        {message.sender === "bot" &&
                          !message.text &&
                          isLoading && (
                            <div className="flex items-center gap-1.5">
                              <Loader2 className="w-3 h-3 animate-spin text-white/40" />
                              <span className="text-white/40 text-[11px]">
                                Thinking...
                              </span>
                            </div>
                          )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>

            {/* ─── Input Area ──────────────────────────────────────────── */}
            <div className="p-4 sm:p-6 relative">
              {/* Upload menu popup */}
              <AnimatePresence>
                {showUploadMenu && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute bottom-20 left-4 sm:left-6 bg-zinc-900/95 backdrop-blur-xl rounded-xl border border-white/10 p-2 shadow-2xl z-30"
                  >
                    <button
                      onClick={() => triggerFileUpload("image/*")}
                      className="flex items-center gap-3 w-full px-4 py-2.5 rounded-lg hover:bg-white/5 transition-colors text-left"
                    >
                      <Image className="w-4 h-4 text-blue-400" />
                      <div>
                        <p className="text-[12px] text-white font-medium">
                          Upload Image
                        </p>
                        <p className="text-[10px] text-white/40">
                          PNG, JPG, WebP, GIF
                        </p>
                      </div>
                    </button>
                    <button
                      onClick={() => triggerFileUpload(".pdf")}
                      className="flex items-center gap-3 w-full px-4 py-2.5 rounded-lg hover:bg-white/5 transition-colors text-left"
                    >
                      <FileText className="w-4 h-4 text-orange-400" />
                      <div>
                        <p className="text-[12px] text-white font-medium">
                          Upload PDF
                        </p>
                        <p className="text-[10px] text-white/40">
                          Travel docs, itineraries
                        </p>
                      </div>
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex items-center gap-3 sm:gap-4 bg-white/[0.04] rounded-full px-4 py-2.5 backdrop-blur-md border border-white/5 focus-within:border-white/10 transition-colors">
                <button
                  onClick={() => setShowUploadMenu(!showUploadMenu)}
                  disabled={uploadingFile}
                  className="relative"
                >
                  {uploadingFile ? (
                    <Loader2 className="text-white/40 w-5 h-5 animate-spin" />
                  ) : (
                    <Plus
                      className={`w-5 h-5 transition-all cursor-pointer shrink-0 ${showUploadMenu
                        ? "text-white rotate-45"
                        : "text-silver-accent hover:text-white"
                        }`}
                    />
                  )}
                </button>

                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyPress}
                  disabled={isLoading}
                  className="flex-1 bg-transparent border-none focus:ring-0 text-[12px] sm:text-[13px] text-slate-200 placeholder-slate-600 p-0 outline-none disabled:opacity-50"
                  placeholder={
                    isLoading ? "Waiting for response..." : "Type a message..."
                  }
                />

                {inputValue.trim() ? (
                  <button
                    onClick={handleSendMessage}
                    disabled={isLoading}
                    className="text-white hover:text-primary transition-colors"
                  >
                    <Send className="w-5 h-5 shrink-0" />
                  </button>
                ) : (
                  <button onClick={handleVoiceModalOpen}>
                    <Mic className="text-silver-accent w-5 h-5 hover:text-white transition-colors cursor-pointer shrink-0" />
                  </button>
                )}
              </div>
            </div>
          </section>

          {/* ─── Right Section: Map & Gallery ─────────────────────────── */}
          <section className="w-[30%] flex flex-col h-full bg-transparent">
            {/* Map Area */}
            <div className="h-1/2 relative overflow-hidden border-b border-white/5">
              {mapImageUrl ? (
                <>
                  <div className="absolute inset-0 z-0">
                    <iframe
                      className="w-full h-full border-0"
                      src={mapImageUrl}
                      title="Map"
                      allowFullScreen
                    />
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
                  <div className="absolute bottom-4 left-4 z-10">
                    <p className="text-[10px] text-white/60 uppercase tracking-widest font-medium">
                      {mapLabel}
                    </p>
                    {mapData?.lat && mapData?.lng && (
                      <p className="text-[9px] text-white/30 mt-0.5">
                        {mapData.lat.toFixed(4)}, {mapData.lng.toFixed(4)}
                      </p>
                    )}
                    {mapPlaces.length > 1 && (
                      <p className="text-[9px] text-white/30 mt-0.5">
                        {mapPlaces.length} places on map
                      </p>
                    )}
                  </div>
                  {mapData?.lat && mapData?.lng && (
                    <a
                      href={`https://www.openstreetmap.org/?mlat=${mapData.lat}&mlon=${mapData.lng}#map=14/${mapData.lat}/${mapData.lng}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="absolute top-3 right-3 z-10 bg-black/50 backdrop-blur-sm rounded-full p-1.5 hover:bg-black/70 transition-colors"
                      title="Open in OpenStreetMap"
                    >
                      <Maximize2 className="w-3 h-3 text-white/60" />
                    </a>
                  )}
                </>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/50">
                  <div className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center mb-3">
                    <Maximize2 className="w-4 h-4 text-white/10" />
                  </div>
                  <p className="text-[11px] text-white/20 uppercase tracking-widest font-medium">
                    {mapLabel}
                  </p>
                  <p className="text-[10px] text-white/10 mt-1">
                    Places will appear on the map as you explore
                  </p>
                </div>
              )}
            </div>

            {/* Gallery Area */}
            <div className="h-1/2 flex flex-col overflow-y-auto custom-scrollbar p-2 sm:p-3 space-y-2 sm:space-y-3">
              {galleryImages.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-4">
                  <Image className="w-8 h-8 text-white/10 mb-3" />
                  <p className="text-[11px] text-white/20 uppercase tracking-widest font-medium">No images yet</p>
                  <p className="text-[10px] text-white/10 mt-1">Search for destinations to see photos here</p>
                </div>
              ) : (
                galleryImages.map((img, i) => (
                  <div
                    key={`gallery-${i}`}
                    className="aspect-square bg-zinc-900/30 rounded-lg overflow-hidden group relative border border-white/5"
                  >
                    <img
                      className="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-110 transition-all duration-700"
                      src={img.image_url}
                      alt={img.title || `Gallery ${i + 1}`}
                      loading="lazy"
                      onError={(e) => {
                        const el = e.target as HTMLImageElement;
                        if (!el.dataset.retried) {
                          // Try image proxy as first fallback
                          el.dataset.retried = "1";
                          el.src = `https://wsrv.nl/?url=${encodeURIComponent(img.image_url)}&w=400&output=jpg`;
                        } else {
                          // Hide broken image entirely
                          el.closest("[class*='aspect-square']")?.classList.add("hidden");
                        }
                      }}
                    />
                    <div
                      className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center cursor-pointer"
                      onClick={() => setLightboxImage(img.image_url)}
                    >
                      <Maximize2 className="text-white w-5 h-5" />
                      {img.title && (
                        <p className="text-white/70 text-[9px] mt-1 px-2 text-center line-clamp-2">
                          {img.title}
                        </p>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </div>

      {/* ─── Lightbox Modal ──────────────────────────────────────────── */}
      <AnimatePresence>
        {lightboxImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 sm:p-8"
            onClick={() => setLightboxImage(null)}
          >
            <button
              onClick={() => setLightboxImage(null)}
              className="absolute top-4 right-4 sm:top-6 sm:right-6 z-10 bg-black/50 hover:bg-black/80 rounded-full p-2 text-white/70 hover:text-white transition-all"
            >
              <X className="w-5 h-5 sm:w-6 sm:h-6" />
            </button>
            <motion.img
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              src={lightboxImage}
              alt="Full view"
              className="max-w-[90vw] max-h-[85vh] object-contain rounded-xl shadow-2xl"
              onClick={(e) => e.stopPropagation()}
              onError={(e) => {
                const img = e.target as HTMLImageElement;
                // Try proxying through a different path or show placeholder
                if (!img.dataset.retried) {
                  img.dataset.retried = "1";
                  img.src = `https://wsrv.nl/?url=${encodeURIComponent(lightboxImage)}&w=1200&output=jpg`;
                } else {
                  // Show error state — close lightbox since image is truly broken
                  setLightboxImage(null);
                }
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── Voice-to-Voice Modal ────────────────────────────────────── */}
      <AnimatePresence>
        {showVoiceModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-3xl flex flex-col items-center justify-center"
          >
            <button
              onClick={handleVoiceModalClose}
              className="absolute top-8 right-8 text-white/60 hover:text-white transition-colors p-2 bg-white/5 rounded-full hover:bg-white/10"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="flex flex-col items-center max-w-lg w-full px-6">
              <div className="relative flex items-center justify-center mb-12">
                {/* Expanding pulse rings when listening or speaking */}
                {(voiceStatus === "listening" || voiceStatus === "speaking") && (
                  <>
                    <motion.div
                      className={`absolute w-32 h-32 rounded-full ${voiceStatus === "listening" ? "bg-emerald-500/20" : "bg-blue-500/20"}`}
                      animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                    />
                    <motion.div
                      className={`absolute w-32 h-32 rounded-full ${voiceStatus === "listening" ? "bg-emerald-500/30" : "bg-blue-500/30"}`}
                      animate={{ scale: [1, 1.8, 1], opacity: [0.3, 0, 0.3] }}
                      transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
                    />
                  </>
                )}

                {/* Central Button/Visualizer — clickable during listening to manually stop */}
                <button
                  onClick={() => {
                    if (voiceStatus === "listening") {
                      stopRecordingAndProcess();
                    }
                  }}
                  disabled={voiceStatus !== "listening"}
                  className={`relative z-10 w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${voiceStatus === "listening" ? "bg-emerald-500/20 border-2 border-emerald-400 cursor-pointer hover:bg-emerald-500/30" :
                    voiceStatus === "processing" ? "bg-amber-500/20 border-2 border-amber-400 cursor-default" :
                      voiceStatus === "speaking" ? "bg-blue-500/20 border-2 border-blue-400 cursor-default" :
                        "bg-zinc-800 border-2 border-zinc-600 cursor-default"
                    }`}
                  style={{
                    transform: voiceStatus === "listening" ? `scale(${1 + Math.min(audioLevel / 100, 0.2)})` : 'scale(1)'
                  }}
                >
                  {voiceStatus === "listening" ? (
                    <Mic className="text-emerald-400 w-12 h-12" />
                  ) : voiceStatus === "processing" ? (
                    <Loader2 className="text-amber-400 w-12 h-12 animate-spin" />
                  ) : voiceStatus === "speaking" ? (
                    <div className="flex gap-1.5 items-end h-10">
                      {[1, 2, 3, 4, 5].map((bar) => (
                        <motion.div
                          key={bar}
                          className="w-1.5 bg-blue-400 rounded-full"
                          animate={{ height: ["20%", "100%", "20%"] }}
                          transition={{
                            duration: 0.6,
                            repeat: Infinity,
                            delay: bar * 0.1,
                          }}
                        />
                      ))}
                    </div>
                  ) : (
                    <Square className="text-zinc-400 w-10 h-10" />
                  )}
                </button>
              </div>

              {/* Status Text */}
              <h2 className="text-2xl font-bold text-white mb-2">
                {voiceStatus === "listening" && "I'm listening..."}
                {voiceStatus === "processing" && "Thinking..."}
                {voiceStatus === "speaking" && "Agent Planner"}
                {voiceStatus === "idle" && "Ready"}
              </h2>

              <p className="text-slate-400 text-center text-sm md:text-base mb-10 min-h-[48px]">
                {voiceStatus === "listening" && "Speak now — tap the mic to send immediately."}
                {voiceStatus === "processing" && "Processing your voice and generating my response."}
                {voiceStatus === "speaking" && "Playing response... The conversation will continue automatically."}
                {voiceStatus === "idle" && "Tap below to start the conversation."}
              </p>

              {/* Controls */}
              <div className="flex gap-4">
                {(voiceStatus === "processing" || voiceStatus === "speaking") && (
                  <button
                    onClick={stopRecordingAndProcess}
                    className="flex items-center gap-2 px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 text-white font-medium transition-colors border border-white/5"
                  >
                    <Square className="w-4 h-4" /> Stop
                  </button>
                )}
                {voiceStatus === "idle" && (
                  <button
                    onClick={startVoiceActivity}
                    className="flex items-center gap-2 px-6 py-3 rounded-full bg-emerald-500 hover:bg-emerald-600 text-white font-medium transition-colors"
                  >
                    <Mic className="w-4 h-4" /> Resume Listening
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        .blooming-bg {
          background: radial-gradient(circle at 50% 50%, #0a0a0a 0%, #000000 100%);
          position: relative;
        }
        .blooming-bg::after {
          content: '';
          position: absolute;
          inset: 0;
          background: radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.04) 0%, transparent 50%);
          pointer-events: none;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </div>
  );
}
