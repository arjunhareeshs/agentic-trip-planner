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
                : "Backend is not connected. Please start the server with: uvicorn main:app --reload --port 8000",
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
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      // Volume meter & Silence detection
      let isSpeaking = false;
      const SILENCE_THRESHOLD = 5; // Minimal volume threshold
      const SILENCE_DURATION_MS = 1500; // ~1.5 seconds of silence means stop

      volumeIntervalRef.current = setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const average = sum / bufferLength;
        setAudioLevel(average);

        if (average > SILENCE_THRESHOLD) {
          if (!isSpeaking) isSpeaking = true;
          if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
        } else if (isSpeaking) {
          if (!silenceTimerRef.current) {
            silenceTimerRef.current = setTimeout(() => {
              isSpeaking = false;
              stopRecordingAndProcess();
            }, SILENCE_DURATION_MS);
          }
        }
      }, 50);

      // Setup MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start(200); // chunk every 200ms
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setVoiceStatus("idle");
      setShowVoiceModal(false);
    }
  };

  const stopVoiceActivity = () => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (volumeIntervalRef.current) clearInterval(volumeIntervalRef.current);

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

  const stopRecordingAndProcess = async () => {
    stopVoiceActivity();
    setVoiceStatus("processing");
    setAudioLevel(0);

    // Give the recorder a tiny bit of time to finalize chunks
    setTimeout(async () => {
      if (audioChunksRef.current.length === 0) {
        setVoiceStatus("listening");
        startVoiceActivity(); // Restart if nothing recorded
        return;
      }

      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      audioChunksRef.current = [];

      try {
        const result = await processVoice(audioBlob, sessionId);

        // Update chat visually
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

        // Play the TTS audio response
        setVoiceStatus("speaking");
        const audioUrl = URL.createObjectURL(result.audio);
        const audio = new Audio(audioUrl);
        audioElementRef.current = audio;

        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          // Loop back to listening when done speaking, unless user closed modal
          if (showVoiceModal) {
            setVoiceStatus("listening");
            startVoiceActivity();
          }
        };

        await audio.play();
      } catch (err) {
        console.error("Voice processing error:", err);
        setVoiceStatus("idle");
      }
    }, 100);
  };

  const handleVoiceModalOpen = () => {
    setShowVoiceModal(true);
    startVoiceActivity();
  };

  const handleVoiceModalClose = () => {
    setShowVoiceModal(false);
    setVoiceStatus("idle");
    stopVoiceActivity();
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current = null;
    }
  };

  // ── Clean tool-call / internal text from bot messages ──────────────────

  const cleanBotText = (text: string): string => {
    // Remove tool call JSON blocks
    let cleaned = text.replace(/```tool_code[\s\S]*?```/g, "");
    cleaned = cleaned.replace(/<function_call>[\s\S]*?<\/function_call>/g, "");
    cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "");

    // Remove lines that look like tool call params
    const toolPatterns = [
      /^\s*\{\s*"function_call"/m,
      /^\s*\{\s*"name"\s*:\s*"\w+"\s*,\s*"args"\s*:/m,
      /^\s*\{\s*"tool_name"\s*:/m,
      /^Tool call:\s*\w+/im,
      /^Calling tool:\s*\w+/im,
      /^Function call:\s*\w+/im,
      /^Parameters:\s*\{/im,
      /^\s*\{\s*"query"/m,
      /^\s*\{\s*"keyword"/m,
    ];

    // Remove JSON-like blocks that are tool params
    cleaned = cleaned.replace(
      /\{[\s\S]*?"(?:function_call|name|tool_name|args|parameters)"[\s\S]*?\}/g,
      ""
    );

    // Remove lines matching tool patterns
    const lines = cleaned.split("\n");
    const filteredLines = lines.filter((line) => {
      const trimmed = line.trim();
      // Skip empty-ish lines resulting from removal
      if (!trimmed) return true;
      // Skip tool call pattern lines
      for (const pattern of toolPatterns) {
        if (pattern.test(trimmed)) return false;
      }
      return true;
    });

    return filteredLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  };

  // ── Structured Markdown renderer for bot messages ─────────────────────

  const renderMessageContent = (msg: ChatMessage) => {
    const rawText = msg.sender === "bot" ? cleanBotText(msg.text) : msg.text;

    // Extract all inline images from markdown syntax
    const imgRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    const inlineImages: { alt: string; url: string }[] = [];
    let imgMatch;
    while ((imgMatch = imgRegex.exec(rawText)) !== null) {
      inlineImages.push({ alt: imgMatch[1], url: imgMatch[2] });
      const imgUrl = imgMatch[2];
      const imgAlt = imgMatch[1];
      // Add inline images to the gallery
      setTimeout(() => {
        setGalleryImages((prev) => {
          if (prev.some((i) => i.image_url === imgUrl)) return prev;
          return [...prev, { title: imgAlt || "Image", image_url: imgUrl, source_url: "" }];
        });
      }, 0);
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
                    (e.target as HTMLImageElement).style.display = "none";
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
                        (e.target as HTMLImageElement).src =
                          "https://lh3.googleusercontent.com/aida-public/AB6AXuCc6nD11fKeq4FNV2lU7_8VhVcD8ZRNVBfPnjXUgK_rq-dZtnEFVHNcITIEjLmU_WbgIsm9JBIOZu2ocUlaPkSdetKlaRr1Ub0coCLt85ihckt4WqbHZuNiajOFuXS-nD-6Aq4C3zbjNbMxiztBzKBJoVDG2Ai5aAuTEPJWJ0WaYMLTNa5l0JQSI-sxM8AF3gAOSwFJYPZgeQiY3wV4yxRV36YrsgJQnF5To6KnhjKUfAhWpZQXCeksEriX1keyszqjaNGbFErEzMc";
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
            className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-8"
            onClick={() => setLightboxImage(null)}
          >
            <button
              onClick={() => setLightboxImage(null)}
              className="absolute top-6 right-6 text-white/60 hover:text-white transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
            <motion.img
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              src={lightboxImage}
              alt="Full view"
              className="max-w-full max-h-full object-contain rounded-lg"
              onClick={(e) => e.stopPropagation()}
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

                {/* Central Button/Visualizer */}
                <div
                  className={`relative z-10 w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${voiceStatus === "listening" ? "bg-emerald-500/20 border-2 border-emerald-400" :
                    voiceStatus === "processing" ? "bg-amber-500/20 border-2 border-amber-400" :
                      voiceStatus === "speaking" ? "bg-blue-500/20 border-2 border-blue-400" :
                        "bg-zinc-800 border-2 border-zinc-600"
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
                </div>
              </div>

              {/* Status Text */}
              <h2 className="text-2xl font-bold text-white mb-2">
                {voiceStatus === "listening" && "I'm listening..."}
                {voiceStatus === "processing" && "Thinking..."}
                {voiceStatus === "speaking" && "Agent Planner"}
                {voiceStatus === "idle" && "Voice Paused"}
              </h2>

              <p className="text-slate-400 text-center text-sm md:text-base mb-10 min-h-[48px]">
                {voiceStatus === "listening" && "Go ahead and speak. I'll detect when you stop."}
                {voiceStatus === "processing" && "Processing your voice and generating my response."}
                {voiceStatus === "speaking" && "Playing response... The conversation will continue automatically."}
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
