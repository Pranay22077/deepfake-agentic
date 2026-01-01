import React, { useEffect, useMemo, useState } from 'react';
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

type ProcessingPipelineCanvasProps = {
  animate?: boolean;
};

type NodeDef = {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  bg: string;
  icon: React.ReactNode;
};

const nodes: NodeDef[] = [
  { id: 'video', label: 'Video', x: 60, y: 140, color: '#10B981', bg: '#e9fbf3', icon: <Play className="w-5 h-5" /> },
  { id: 'frames', label: 'Frames', x: 60, y: 210, color: '#10B981', bg: '#e9fbf3', icon: <Image className="w-5 h-5" /> },
  { id: 'face', label: 'Face', x: 60, y: 280, color: '#10B981', bg: '#e9fbf3', icon: <User className="w-5 h-5" /> },
  { id: 'audio', label: 'Audio', x: 60, y: 350, color: '#10B981', bg: '#e9fbf3', icon: <Volume2 className="w-5 h-5" /> },

  { id: 'bg', label: 'BG Model', x: 200, y: 170, color: '#3B82F6', bg: '#eef4ff', icon: <Eye className="w-5 h-5" /> },
  { id: 'av', label: 'AV Model', x: 200, y: 230, color: '#3B82F6', bg: '#eef4ff', icon: <Zap className="w-5 h-5" /> },
  { id: 'cm', label: 'CM Model', x: 200, y: 290, color: '#3B82F6', bg: '#eef4ff', icon: <Brain className="w-5 h-5" /> },
  { id: 'rr', label: 'RR Model', x: 200, y: 350, color: '#3B82F6', bg: '#eef4ff', icon: <Shield className="w-5 h-5" /> },
  { id: 'll', label: 'LL Model', x: 200, y: 410, color: '#3B82F6', bg: '#eef4ff', icon: <Layers className="w-5 h-5" /> },
  { id: 'tm', label: 'TM Model', x: 200, y: 470, color: '#3B82F6', bg: '#eef4ff', icon: <Timer className="w-5 h-5" /> },

  { id: 'router', label: 'Router', x: 360, y: 190, color: '#9b8efb', bg: '#f2ecff', icon: <Route className="w-5 h-5" /> },
  { id: 'langgraph', label: 'LangGraph', x: 360, y: 260, color: '#9b8efb', bg: '#f2ecff', icon: <Network className="w-5 h-5" /> },
  { id: 'aggregator', label: 'Aggregator', x: 360, y: 330, color: '#9b8efb', bg: '#f2ecff', icon: <Combine className="w-5 h-5" /> },
  { id: 'explainer', label: 'Explainer', x: 360, y: 400, color: '#9b8efb', bg: '#f2ecff', icon: <Search className="w-5 h-5" /> },

  { id: 'response', label: 'Response', x: 520, y: 260, color: '#f7b733', bg: '#fff6e6', icon: <FileText className="w-5 h-5" /> },
  { id: 'heatmap', label: 'Heatmap', x: 520, y: 340, color: '#f7b733', bg: '#fff6e6', icon: <BarChart3 className="w-5 h-5" /> },
];

const edges: Array<[string, string]> = [
  ['video', 'bg'],
  ['frames', 'av'],
  ['frames', 'cm'],
  ['face', 'rr'],
  ['face', 'll'],
  ['audio', 'rr'],
  ['bg', 'router'],
  ['router', 'langgraph'],
  ['av', 'langgraph'],
  ['cm', 'langgraph'],
  ['langgraph', 'aggregator'],
  ['rr', 'aggregator'],
  ['ll', 'aggregator'],
  ['tm', 'aggregator'],
  ['aggregator', 'explainer'],
  ['aggregator', 'response'],
  ['aggregator', 'heatmap'],
];

const ProcessingPipelineCanvas: React.FC<ProcessingPipelineCanvasProps> = ({ animate }) => {
  const findNode = (id: string) => nodes.find((n) => n.id === id)!;

  const SCALE = 0.9;
  const WIDTH = 620 * SCALE;
  const HEIGHT = 520 * SCALE;

  const [activeEdgeIndex, setActiveEdgeIndex] = useState<number | null>(null);
  const [activatedNodes, setActivatedNodes] = useState<Set<string>>(new Set());

  useEffect(() => {
    let edgeTimer: ReturnType<typeof setTimeout> | null = null;
    let nodeTimer: ReturnType<typeof setTimeout> | null = null;

    if (!animate) {
      setActiveEdgeIndex(null);
      setActivatedNodes(new Set());
      return () => {};
    }

    // Seed inputs as "on" so they glow lightly
    setActivatedNodes(new Set(['video', 'frames', 'face', 'audio']));

    let idx = 0;
    const cycleEdges = () => {
      setActiveEdgeIndex(idx);
      const [, to] = edges[idx];

      nodeTimer = setTimeout(() => {
        setActivatedNodes((prev) => {
          const next = new Set(prev);
          next.add(to);
          return next;
        });
      }, 750);

      idx = (idx + 1) % edges.length;
      edgeTimer = setTimeout(cycleEdges, 1200);
    };

    edgeTimer = setTimeout(cycleEdges, 200);

    return () => {
      if (edgeTimer) clearTimeout(edgeTimer);
      if (nodeTimer) clearTimeout(nodeTimer);
    };
  }, [animate]);

  const nodeStyle = useMemo(
    () =>
      nodes.reduce<Record<string, { fill: string; bg: string }>>((acc, node) => {
        acc[node.id] = { fill: node.color, bg: node.bg };
        return acc;
      }, {}),
    []
  );

  const toRgba = (hex: string, alpha: number) => {
    const clean = hex.replace('#', '');
    const bigint = parseInt(clean, 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  return (
    <div className="w-full">
      <style>{`
        @keyframes dashFlow {
          from { stroke-dashoffset: 0; }
          to { stroke-dashoffset: -140; }
        }
        @keyframes pulseGlow {
          0% { filter: drop-shadow(0 0 0 rgba(124, 58, 237, 0.18)); }
          50% { filter: drop-shadow(0 0 12px rgba(124, 58, 237, 0.36)); }
          100% { filter: drop-shadow(0 0 0 rgba(124, 58, 237, 0.18)); }
        }
        .edge-active {
          animation: dashFlow 1.35s linear infinite, pulseGlow 1.35s ease-in-out infinite;
          stroke-linecap: round;
        }
        .node-glow {
          box-shadow: 0 10px 30px -12px rgba(0,0,0,0.12);
        }
      `}</style>

      <div className="grid grid-cols-4 text-sm font-semibold text-gray-700 dark:text-slate-200 mb-6 px-2 select-none">
        <span className="text-center text-[#10B981]">Inputs</span>
        <span className="text-center text-[#3B82F6]">AI Model Stack</span>
        <span className="text-center text-[#8B5CF6]">Agent Orchestration Layer</span>
        <span className="text-center text-[#F59E0B]">System Outputs</span>
      </div>

      <div className="relative mx-auto" style={{ width: WIDTH, height: HEIGHT }}>
        <svg className="absolute inset-0 w-full h-full pointer-events-none" fill="none">
          {edges.map(([from, to], i) => {
            const a = findNode(from);
            const b = findNode(to);
            const isActive = activeEdgeIndex === i;
            const targetActive = activatedNodes.has(to);
            const strokeColor = isActive ? nodeStyle[to]?.fill || '#9b8efb' : targetActive ? '#cbd5e1' : '#d6dce5';

            return (
              <line
                key={`${from}-${to}-${i}`}
                x1={(a.x + 28) * SCALE}
                y1={(a.y + 28) * SCALE}
                x2={(b.x + 28) * SCALE}
                y2={(b.y + 28) * SCALE}
                stroke={strokeColor}
                strokeWidth={isActive ? 3 : 2}
                strokeDasharray={isActive ? '8 10' : undefined}
                className={isActive ? 'edge-active' : undefined}
                opacity={isActive ? 0.95 : targetActive ? 0.75 : 0.55}
              />
            );
          })}
        </svg>

        {nodes.map((node) => {
          const baseOn = !animate; // show base tint when idle to match mock
          const isLit = activatedNodes.has(node.id) || baseOn;
          const isTargetNow =
            activeEdgeIndex !== null && edges[activeEdgeIndex][1] === node.id;

          return (
            <div
              key={node.id}
              className="absolute flex flex-col items-center gap-1 select-none"
              style={{ left: node.x * SCALE, top: node.y * SCALE }}
            >
              <div
                className="w-14 h-14 rounded-full border-2 flex items-center justify-center node-glow transition-all duration-300"
                style={{
                  borderColor: isLit || isTargetNow ? node.color : '#e5e7eb',
                  backgroundColor: isLit || isTargetNow ? node.bg : '#f8fafc',
                  color: node.color,
                  boxShadow: isLit || isTargetNow
                    ? `0 0 0 6px ${toRgba(node.color, 0.08)}, 0 12px 28px -16px ${toRgba(node.color, 0.55)}`
                    : '0 10px 20px -15px rgba(0,0,0,0.08)',
                  transform: isLit || isTargetNow ? 'translateY(-2px)' : 'translateY(0)',
                }}
              >
                {node.icon}
              </div>
              <span className="text-[11px] text-gray-700 dark:text-slate-200">{node.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ProcessingPipelineCanvas;