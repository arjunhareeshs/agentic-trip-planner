/**
 * AgentActivityPanel.tsx — Real-time debug overlay showing agent internals.
 *
 * Renders a small floating icon (top-left) that toggles a full overlay panel
 * displaying tool calls, tool responses, code executions, and multi-agent
 * coordination events in a structured card-grid layout.
 */

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Activity,
  X,
  Wrench,
  ArrowDownToLine,
  Code2,
  Terminal,
  ChevronDown,
  ChevronRight,
  Trash2,
} from "lucide-react";
import type { AgentEvent } from "../services/api";

// ── Types ─────────────────────────────────────────────────────────────────

interface AgentActivityPanelProps {
  events: AgentEvent[];
  onClear?: () => void;
}

// ── Helpers ───────────────────────────────────────────────────────────────

const EVENT_CONFIG: Record<
  AgentEvent["event_type"],
  { label: string; icon: React.ElementType; color: string; bg: string; border: string }
> = {
  tool_call: {
    label: "Tool Call",
    icon: Wrench,
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
  },
  tool_response: {
    label: "Tool Response",
    icon: ArrowDownToLine,
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
  },
  code_exec: {
    label: "Code Execution",
    icon: Code2,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
  },
  code_result: {
    label: "Code Result",
    icon: Terminal,
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    border: "border-purple-500/20",
  },
};

/** Group events by agent name for the parallel card layout. */
function groupByAgent(events: AgentEvent[]): Map<string, AgentEvent[]> {
  const map = new Map<string, AgentEvent[]>();
  for (const ev of events) {
    const key = ev.agent || "unknown";
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(ev);
  }
  return map;
}

/** Pretty-print a JSON-like object, truncated. */
function formatArgs(args: Record<string, unknown> | undefined): string {
  if (!args || Object.keys(args).length === 0) return "—";
  try {
    const str = JSON.stringify(args, null, 2);
    return str.length > 300 ? str.slice(0, 300) + "…" : str;
  } catch {
    return String(args);
  }
}

// ── Sub-components ────────────────────────────────────────────────────────

/** Single event card */
function EventCard({ event, index }: { event: AgentEvent; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = EVENT_CONFIG[event.event_type];
  const Icon = cfg.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03, duration: 0.2 }}
      className={`rounded-lg border ${cfg.border} ${cfg.bg} p-2.5 cursor-pointer select-none transition-colors hover:brightness-110`}
      onClick={() => setExpanded((p) => !p)}
    >
      {/* Header */}
      <div className="flex items-center gap-2">
        <Icon className={`w-3.5 h-3.5 ${cfg.color} shrink-0`} />
        <span className={`text-[11px] font-semibold uppercase tracking-wide ${cfg.color}`}>
          {cfg.label}
        </span>
        <span className="ml-auto">
          {expanded ? (
            <ChevronDown className="w-3 h-3 text-white/30" />
          ) : (
            <ChevronRight className="w-3 h-3 text-white/30" />
          )}
        </span>
      </div>

      {/* Tool name (always visible) */}
      {event.tool && (
        <p className="text-[11px] text-white/70 mt-1 font-mono truncate">
          {event.tool}
        </p>
      )}

      {/* Expandable details */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <div className="mt-2 pt-2 border-t border-white/5 space-y-1.5">
              {/* Args (tool_call) */}
              {event.args && Object.keys(event.args).length > 0 && (
                <div>
                  <span className="text-[10px] text-white/40 uppercase tracking-wider">
                    Parameters
                  </span>
                  <pre className="text-[10px] text-slate-300 bg-black/30 rounded p-1.5 mt-0.5 overflow-x-auto max-h-32 whitespace-pre-wrap break-all font-mono">
                    {formatArgs(event.args)}
                  </pre>
                </div>
              )}

              {/* Result (tool_response) */}
              {event.result && (
                <div>
                  <span className="text-[10px] text-white/40 uppercase tracking-wider">
                    Result
                  </span>
                  <pre className="text-[10px] text-slate-300 bg-black/30 rounded p-1.5 mt-0.5 overflow-x-auto max-h-32 whitespace-pre-wrap break-all font-mono">
                    {event.result}
                  </pre>
                </div>
              )}

              {/* Code (code_exec) */}
              {event.code && (
                <div>
                  <span className="text-[10px] text-white/40 uppercase tracking-wider">
                    Code
                  </span>
                  <pre className="text-[10px] text-slate-300 bg-black/30 rounded p-1.5 mt-0.5 overflow-x-auto max-h-32 whitespace-pre-wrap break-all font-mono">
                    {event.code}
                  </pre>
                </div>
              )}

              {/* Output (code_result) */}
              {event.output && (
                <div>
                  <span className="text-[10px] text-white/40 uppercase tracking-wider">
                    Output
                  </span>
                  <pre className="text-[10px] text-slate-300 bg-black/30 rounded p-1.5 mt-0.5 overflow-x-auto max-h-32 whitespace-pre-wrap break-all font-mono">
                    {event.output}
                  </pre>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/** Column for a single agent */
function AgentColumn({ agentName, events }: { agentName: string; events: AgentEvent[] }) {
  // Friendly label
  const label = agentName === "unknown" ? "Orchestrator" : agentName;

  return (
    <div className="flex-1 min-w-[220px] max-w-[340px]">
      {/* Agent header */}
      <div className="flex items-center gap-2 mb-2 px-1">
        <div className="w-2 h-2 rounded-full bg-white/30 animate-pulse" />
        <span className="text-[11px] font-bold text-white/80 uppercase tracking-widest truncate">
          {label}
        </span>
        <span className="text-[10px] text-white/30 ml-auto">{events.length}</span>
      </div>

      {/* Events list */}
      <div className="space-y-2 pr-1">
        {events.map((ev, i) => (
          <EventCard key={`${agentName}-${i}`} event={ev} index={i} />
        ))}
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export default function AgentActivityPanel({ events, onClear }: AgentActivityPanelProps) {
  const [open, setOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when new events arrive
  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events.length, open]);

  const grouped = groupByAgent(events);
  const agentCount = grouped.size;

  return (
    <>
      {/* ── Floating toggle button ─────────────────────────────────── */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen((p) => !p)}
        className={`fixed top-4 left-4 z-[60] p-2 rounded-xl border backdrop-blur-md shadow-lg transition-colors ${
          open
            ? "bg-white/15 border-white/20 text-white"
            : events.length > 0
              ? "bg-blue-500/20 border-blue-400/30 text-blue-400"
              : "bg-white/5 border-white/10 text-white/50 hover:text-white/80"
        }`}
        title="Agent Activity"
      >
        <Activity className="w-4 h-4" />
        {/* Live badge when there are events and panel is closed */}
        {!open && events.length > 0 && (
          <span className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full bg-blue-500 text-[8px] text-white font-bold flex items-center justify-center">
            {events.length > 99 ? "…" : events.length}
          </span>
        )}
      </motion.button>

      {/* ── Overlay panel ──────────────────────────────────────────── */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="fixed inset-4 z-[55] rounded-2xl border border-white/10 bg-black/90 backdrop-blur-xl shadow-2xl flex flex-col overflow-hidden"
          >
            {/* Panel header */}
            <div className="flex items-center gap-3 px-5 py-3 border-b border-white/5 shrink-0">
              <Activity className="w-4 h-4 text-blue-400" />
              <h2 className="text-sm font-semibold text-white tracking-wide">
                Agent Activity
              </h2>
              <span className="text-[10px] text-white/30 ml-1">
                {events.length} event{events.length !== 1 ? "s" : ""} · {agentCount} agent
                {agentCount !== 1 ? "s" : ""}
              </span>

              <div className="ml-auto flex items-center gap-2">
                {onClear && events.length > 0 && (
                  <button
                    onClick={onClear}
                    className="text-white/30 hover:text-red-400 transition-colors p-1 rounded-lg hover:bg-white/5"
                    title="Clear events"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  onClick={() => setOpen(false)}
                  className="text-white/30 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/5"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Panel body — parallel columns per agent */}
            <div ref={scrollRef} className="flex-1 overflow-auto p-4">
              {events.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center space-y-2">
                    <Activity className="w-8 h-8 text-white/10 mx-auto" />
                    <p className="text-sm text-white/20">No agent activity yet</p>
                    <p className="text-[11px] text-white/10">
                      Send a message to see tool calls, sub-agent work, and coordination here.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="flex gap-4 overflow-x-auto pb-2 items-start">
                  {Array.from(grouped.entries()).map(([agent, agentEvents]) => (
                    <AgentColumn key={agent} agentName={agent} events={agentEvents} />
                  ))}
                </div>
              )}
            </div>

            {/* Legend bar */}
            <div className="flex items-center gap-4 px-5 py-2 border-t border-white/5 shrink-0">
              {Object.entries(EVENT_CONFIG).map(([key, cfg]) => {
                const Icon = cfg.icon;
                return (
                  <div key={key} className="flex items-center gap-1.5">
                    <Icon className={`w-3 h-3 ${cfg.color}`} />
                    <span className={`text-[9px] uppercase tracking-wider ${cfg.color}`}>
                      {cfg.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
