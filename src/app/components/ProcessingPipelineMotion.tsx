import React, { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Play,
  Image,
  User,
  Volume2,
  Eye,
  Zap,
  Brain,
  Shield,
  Layers,
  Timer,
  Route,
  Network,
  Combine,
  Search,
  FileText,
  BarChart3,
} from 'lucide-react';

// 4:3 aspect ratio, sized for a side panel but usable full-width stacked
// Uses motion (Framer Motion v12) for glow + path pulses.

type NodeDef = {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  bg: string;
  icon: React.ReactNode;
};

type Stage = {
  id: string;
  label: string;
  nodes: string[];
  edges: Array<[string, string]>;
};

type ProcessingPipelineMotionProps = {
  animate?: boolean;
};

const nodes: NodeDef[] = [
  { id: 'video', label: 'Video', x: 70, y: 120, color: '#10b981', bg: '#e9fbf3', icon: <Play className="w-5 h-5" /> },
  { id: 'frames', label: 'Frames', x: 70, y: 200, color: '#10b981', bg: '#e9fbf3', icon: <Image className="w-5 h-5" /> },
  { id: 'face', label: 'Face', x: 70, y: 280, color: '#10b981', bg: '#e9fbf3', icon: <User className="w-5 h-5" /> },
  { id: 'audio', label: 'Audio', x: 70, y: 360, color: '#10b981', bg: '#e9fbf3', icon: <Volume2 className="w-5 h-5" /> },

  { id: 'bg', label: 'BG Model', x: 220, y: 150, color: '#3b82f6', bg: '#eef4ff', icon: <Eye className="w-5 h-5" /> },
  { id: 'av', label: 'AV Model', x: 220, y: 210, color: '#3b82f6', bg: '#eef4ff', icon: <Zap className="w-5 h-5" /> },
  { id: 'cm', label: 'CM Model', x: 220, y: 270, color: '#3b82f6', bg: '#eef4ff', icon: <Brain className="w-5 h-5" /> },
  { id: 'rr', label: 'RR Model', x: 220, y: 330, color: '#3b82f6', bg: '#eef4ff', icon: <Shield className="w-5 h-5" /> },
  { id: 'll', label: 'LL Model', x: 220, y: 390, color: '#3b82f6', bg: '#eef4ff', icon: <Layers className="w-5 h-5" /> },
  { id: 'tm', label: 'TM Model', x: 220, y: 450, color: '#3b82f6', bg: '#eef4ff', icon: <Timer className="w-5 h-5" /> },

  { id: 'router', label: 'Router', x: 380, y: 190, color: '#9b8efb', bg: '#f2ecff', icon: <Route className="w-5 h-5" /> },
  { id: 'langgraph', label: 'LangGraph', x: 380, y: 260, color: '#9b8efb', bg: '#f2ecff', icon: <Network className="w-5 h-5" /> },
  { id: 'aggregator', label: 'Aggregator', x: 380, y: 330, color: '#9b8efb', bg: '#f2ecff', icon: <Combine className="w-5 h-5" /> },
  { id: 'explainer', label: 'Explainer', x: 380, y: 400, color: '#9b8efb', bg: '#f2ecff', icon: <Search className="w-5 h-5" /> },

  { id: 'response', label: 'Response', x: 540, y: 250, color: '#f59e0b', bg: '#fff6e6', icon: <FileText className="w-5 h-5" /> },
  { id: 'heatmap', label: 'Heatmap', x: 540, y: 340, color: '#f59e0b', bg: '#fff6e6', icon: <BarChart3 className="w-5 h-5" /> },
];

const edges: Array<[string, string]> = [
  ['video', 'bg'],
  ['frames', 'av'],
  ['frames', 'cm'],
  ['face', 'rr'],
  ['face', 'll'],
  ['audio', 'rr'],
  ['bg', 'router'],
  ['av', 'langgraph'],
  ['cm', 'langgraph'],
  ['rr', 'aggregator'],
  ['ll', 'aggregator'],
  ['tm', 'aggregator'],
  ['router', 'langgraph'],
  ['langgraph', 'aggregator'],
  ['aggregator', 'explainer'],
  ['aggregator', 'response'],
  ['aggregator', 'heatmap'],
];

const stages: Stage[] = [
  {
    id: 'feature',
    label: 'Feature Extraction',
    nodes: ['video', 'frames', 'face', 'audio', 'bg', 'av', 'cm', 'rr'],
    edges: [
      ['video', 'bg'],
      ['frames', 'av'],
      ['frames', 'cm'],
      ['face', 'rr'],
      ['audio', 'rr'],
    ],
  },
  {
    id: 'inference',
    label: 'Model Inference',
    nodes: ['bg', 'av', 'cm', 'rr', 'll', 'tm', 'router', 'langgraph'],
    edges: [
      ['bg', 'router'],
      ['router', 'langgraph'],
      ['av', 'langgraph'],
      ['cm', 'langgraph'],
      ['ll', 'aggregator'],
      ['tm', 'aggregator'],
    ],
  },
  {
    id: 'aggregation',
    label: 'Result Aggregation',
    nodes: ['aggregator', 'explainer', 'response', 'heatmap', 'langgraph'],
    edges: [
      ['langgraph', 'aggregator'],
      ['aggregator', 'explainer'],
      ['aggregator', 'response'],
      ['aggregator', 'heatmap'],
    ],
  },
];

const toRgba = (hex: string, alpha: number) => {
  const clean = hex.replace('#', '');
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export const ProcessingPipelineMotion: React.FC<ProcessingPipelineMotionProps> = ({ animate }) => {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    if (!animate) {
      setStageIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setStageIndex((s) => (s + 1) % stages.length);
    }, 2300);
    return () => clearInterval(timer);
  }, [animate]);

  const stage = stages[stageIndex];

  const nodeMap = useMemo(() => nodes.reduce<Record<string, NodeDef>>((acc, n) => {
    acc[n.id] = n;
    return acc;
  }, {}), []);

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="mb-3 flex items-center justify-between text-sm text-slate-600 dark:text-slate-300 px-1">
        <div className="font-semibold text-slate-800 dark:text-white">Processing Pipeline</div>
        <div className="flex items-center gap-2 text-xs">
          <span className="px-3 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200 dark:bg-slate-800/70 dark:text-slate-100 dark:border-slate-700">
            {stage.label}
          </span>
          <span className={`px-3 py-1 rounded-full border text-[11px] ${animate ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-200 dark:border-emerald-500/40' : 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800/60 dark:text-slate-200 dark:border-slate-700'}`}>
            {animate ? 'Animating' : 'Idle'}
          </span>
        </div>
      </div>

      <div className="relative rounded-3xl border border-slate-200/80 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 shadow-[0_30px_90px_-45px_rgba(59,130,246,0.35)] overflow-hidden" style={{ aspectRatio: '4 / 3' }}>
        <div className="absolute inset-0 pointer-events-none" aria-hidden>
          <div className="grid grid-cols-4 text-[13px] font-semibold text-center text-slate-500 dark:text-slate-300 py-3 select-none">
            <span className="text-emerald-500">Inputs</span>
            <span className="text-blue-500">AI Model Stack</span>
            <span className="text-purple-500">Agent Orchestration Layer</span>
            <span className="text-amber-500">System Outputs</span>
          </div>
        </div>

        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 640 520">
          {edges.map(([from, to], i) => {
            const a = nodeMap[from];
            const b = nodeMap[to];
            const isActive = stage.edges.some(([f, t]) => f === from && t === to);
            return (
              <g key={`${from}-${to}-${i}`}>
                <line
                  x1={a.x + 30}
                  y1={a.y + 30}
                  x2={b.x + 30}
                  y2={b.y + 30}
                  stroke="#d7dce6"
                  strokeWidth={2}
                  strokeLinecap="round"
                />
                <AnimatePresence>
                  {animate && isActive && (
                    <motion.line
                      x1={a.x + 30}
                      y1={a.y + 30}
                      x2={b.x + 30}
                      y2={b.y + 30}
                      stroke="#60a5fa"
                      strokeWidth={3}
                      strokeLinecap="round"
                      strokeDasharray="10 12"
                      initial={{ strokeDashoffset: 0, opacity: 0.2 }}
                      animate={{ strokeDashoffset: -180, opacity: [0.7, 1, 0.7] }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 1.4, repeat: Infinity, ease: 'linear' }}
                    />
                  )}
                </AnimatePresence>
              </g>
            );
          })}
        </svg>

        <div className="absolute inset-0">
          {nodes.map((node) => {
            const active = animate && stage.nodes.includes(node.id);
            return (
              <motion.div
                key={node.id}
                className="absolute flex flex-col items-center gap-1 select-none"
                style={{ left: node.x, top: node.y }}
                animate={{ scale: active ? 1.08 : 1, y: active ? -2 : 0 }}
                transition={{ type: 'spring', stiffness: 260, damping: 18 }}
              >
                <div
                  className="w-14 h-14 rounded-full border-2 flex items-center justify-center shadow-sm"
                  style={{
                    borderColor: active ? node.color : '#e5e7eb',
                    backgroundColor: active ? node.bg : '#f8fafc',
                    color: node.color,
                    boxShadow: active
                      ? `0 0 0 6px ${toRgba(node.color, 0.08)}, 0 15px 35px -18px ${toRgba(node.color, 0.65)}`
                      : '0 12px 26px -18px rgba(15,23,42,0.25)',
                  }}
                >
                  {node.icon}
                </div>
                <span className="text-[11px] text-slate-700 dark:text-slate-200">{node.label}</span>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ProcessingPipelineMotion;
