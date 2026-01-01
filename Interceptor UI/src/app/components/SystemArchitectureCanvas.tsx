import React, { useEffect, useRef, useState } from 'react';
import { useArchitecture } from '../context/ArchitectureContext';
import { 
  Video, 
  Image, 
  ScanFace, 
  Music, 
  Brain, 
  Zap, 
  Network, 
  Sparkles,
  Layers,
  FileCode,
  Map as MapIcon
} from 'lucide-react';

interface Node {
  id: string;
  x: number;
  y: number;
  label: string;
  type: 'input' | 'model' | 'agent' | 'output';
  layer: number;
  icon: string;
}

interface Connection {
  from: string;
  to: string;
}

interface Particle {
  id: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
  color: string;
  size: number;
}

const SystemArchitectureCanvas: React.FC<{ view?: 'overview' | 'pipeline' }> = ({ view = 'pipeline' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { state } = useArchitecture();
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();
  const pulseRef = useRef<number>(0);
  const [isDark, setIsDark] = useState(false);
  const iconCanvasRef = useRef<Map<string, HTMLCanvasElement>>(new Map());

  // Detect theme changes
  useEffect(() => {
    const updateTheme = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    
    updateTheme();
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    
    return () => observer.disconnect();
  }, []);

  // Optimized node layout - FULLY VISIBLE WITH TIGHTER SPACING
  const nodes: Node[] = [
    // Layer 1: Input Modalities (x: 40)
    { id: 'video-input', x: 40, y: 100, label: 'Video', type: 'input', layer: 1, icon: 'video' },
    { id: 'frame-sampler', x: 40, y: 160, label: 'Frames', type: 'input', layer: 1, icon: 'image' },
    { id: 'face-detector', x: 40, y: 220, label: 'Face', type: 'input', layer: 1, icon: 'scan' },
    { id: 'audio-extractor', x: 40, y: 280, label: 'Audio', type: 'input', layer: 1, icon: 'music' },

    // Layer 2: AI Model Stack (x: 160)
    { id: 'bg-model', x: 160, y: 80, label: 'BG Model', type: 'model', layer: 2, icon: 'eye' },
    { id: 'av-model', x: 160, y: 135, label: 'AV Model', type: 'model', layer: 2, icon: 'headphones' },
    { id: 'cm-model', x: 160, y: 190, label: 'CM Model', type: 'model', layer: 2, icon: 'brain' },
    { id: 'rr-model', x: 160, y: 245, label: 'RR Model', type: 'model', layer: 2, icon: 'shield' },
    { id: 'll-model', x: 160, y: 300, label: 'LL Model', type: 'model', layer: 2, icon: 'grid' },
    { id: 'tm-model', x: 160, y: 355, label: 'TM Model', type: 'model', layer: 2, icon: 'clock' },

    // Layer 3: Agent Orchestration Layer (x: 290)
    { id: 'routing-engine', x: 290, y: 100, label: 'Router', type: 'agent', layer: 3, icon: 'zap' },
    { id: 'langgraph', x: 290, y: 170, label: 'LangGraph', type: 'agent', layer: 3, icon: 'server' },
    { id: 'aggregator', x: 290, y: 240, label: 'Aggregator', type: 'agent', layer: 3, icon: 'database' },
    { id: 'explainer', x: 290, y: 310, label: 'Explainer', type: 'agent', layer: 3, icon: 'search' },

    // Layer 4: System Outputs (x: 420)
    { id: 'api-response', x: 420, y: 190, label: 'Response', type: 'output', layer: 4, icon: 'message' },
    { id: 'heatmap', x: 420, y: 260, label: 'Heatmap', type: 'output', layer: 4, icon: 'table' },
  ];

  const connections: Connection[] = [
    { from: 'video-input', to: 'frame-sampler' },
    { from: 'video-input', to: 'audio-extractor' },
    { from: 'frame-sampler', to: 'face-detector' },
    { from: 'face-detector', to: 'bg-model' },
    { from: 'face-detector', to: 'av-model' },
    { from: 'face-detector', to: 'cm-model' },
    { from: 'face-detector', to: 'rr-model' },
    { from: 'face-detector', to: 'll-model' },
    { from: 'face-detector', to: 'tm-model' },
    { from: 'audio-extractor', to: 'av-model' },
    { from: 'bg-model', to: 'routing-engine' },
    { from: 'routing-engine', to: 'langgraph' },
    { from: 'av-model', to: 'aggregator' },
    { from: 'cm-model', to: 'aggregator' },
    { from: 'rr-model', to: 'aggregator' },
    { from: 'll-model', to: 'aggregator' },
    { from: 'tm-model', to: 'aggregator' },
    { from: 'langgraph', to: 'aggregator' },
    { from: 'aggregator', to: 'explainer' },
    { from: 'explainer', to: 'api-response' },
    { from: 'explainer', to: 'heatmap' },
  ];

  // Theme-aware colors
  const getNodeColor = (type: string, isActive: boolean) => {
    if (isDark) {
      const colors = {
        input: {
          active: { primary: '#10B981', secondary: '#34D399', glow: '#6EE7B7' },
          inactive: { primary: '#047857', secondary: '#059669', glow: '#10B981' }
        },
        model: {
          active: { primary: '#3B82F6', secondary: '#60A5FA', glow: '#93C5FD' },
          inactive: { primary: '#1E40AF', secondary: '#2563EB', glow: '#3B82F6' }
        },
        agent: {
          active: { primary: '#8B5CF6', secondary: '#A78BFA', glow: '#C4B5FD' },
          inactive: { primary: '#6D28D9', secondary: '#7C3AED', glow: '#8B5CF6' }
        },
        output: {
          active: { primary: '#F59E0B', secondary: '#FBBF24', glow: '#FCD34D' },
          inactive: { primary: '#B45309', secondary: '#D97706', glow: '#F59E0B' }
        }
      };
      return isActive ? colors[type].active : colors[type].inactive;
    } else {
      const colors = {
        input: {
          active: { primary: '#059669', secondary: '#10B981', glow: '#34D399' },
          inactive: { primary: '#D1FAE5', secondary: '#A7F3D0', glow: '#6EE7B7' }
        },
        model: {
          active: { primary: '#1D4ED8', secondary: '#3B82F6', glow: '#60A5FA' },
          inactive: { primary: '#DBEAFE', secondary: '#BFDBFE', glow: '#93C5FD' }
        },
        agent: {
          active: { primary: '#6D28D9', secondary: '#8B5CF6', glow: '#A78BFA' },
          inactive: { primary: '#EDE9FE', secondary: '#DDD6FE', glow: '#C4B5FD' }
        },
        output: {
          active: { primary: '#B45309', secondary: '#F59E0B', glow: '#FBBF24' },
          inactive: { primary: '#FEF3C7', secondary: '#FDE68A', glow: '#FCD34D' }
        }
      };
      return isActive ? colors[type].active : colors[type].inactive;
    }
  };

  // Create icon images for rendering
  const createIconCanvas = (iconName: string, color: string, size: number = 24): HTMLCanvasElement => {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return canvas;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const center = size / 2;
    const scale = size / 24;

    // Simplified icon paths
    ctx.save();
    ctx.translate(center, center);
    ctx.scale(scale, scale);

    switch (iconName) {
      case 'video':
        ctx.strokeRect(-10, -7, 14, 14);
        ctx.beginPath();
        ctx.moveTo(4, -4);
        ctx.lineTo(10, -1);
        ctx.lineTo(10, 1);
        ctx.lineTo(4, 4);
        ctx.closePath();
        ctx.fill();
        break;
      case 'image':
        ctx.strokeRect(-10, -10, 20, 20);
        ctx.beginPath();
        ctx.arc(-3, -3, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(-10, 5);
        ctx.lineTo(-3, -2);
        ctx.lineTo(3, 4);
        ctx.lineTo(10, -3);
        ctx.stroke();
        break;
      case 'scan':
        ctx.beginPath();
        ctx.arc(0, 0, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(-3, -2, 1.5, 0, Math.PI * 2);
        ctx.arc(3, -2, 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(0, 2, 3, 0, Math.PI);
        ctx.stroke();
        break;
      case 'music':
        ctx.beginPath();
        ctx.arc(-5, 4, 3, 0, Math.PI * 2);
        ctx.arc(3, 2, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillRect(-2, -8, 2, 12);
        ctx.fillRect(6, -10, 2, 12);
        ctx.fillRect(-2, -8, 8, 2);
        break;
      case 'brain':
        ctx.beginPath();
        ctx.arc(-3, -3, 4, 0, Math.PI * 2);
        ctx.arc(3, -3, 4, 0, Math.PI * 2);
        ctx.arc(-3, 4, 3, 0, Math.PI * 2);
        ctx.arc(3, 4, 3, 0, Math.PI * 2);
        ctx.stroke();
        break;
      case 'zap':
        ctx.beginPath();
        ctx.moveTo(-2, -10);
        ctx.lineTo(-6, 2);
        ctx.lineTo(0, 2);
        ctx.lineTo(2, 10);
        ctx.lineTo(6, -2);
        ctx.lineTo(0, -2);
        ctx.closePath();
        ctx.fill();
        break;
      case 'network':
        ctx.beginPath();
        ctx.arc(0, 0, 2, 0, Math.PI * 2);
        ctx.arc(-7, -7, 2, 0, Math.PI * 2);
        ctx.arc(7, -7, 2, 0, Math.PI * 2);
        ctx.arc(-7, 7, 2, 0, Math.PI * 2);
        ctx.arc(7, 7, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-7, -7);
        ctx.moveTo(0, 0);
        ctx.lineTo(7, -7);
        ctx.moveTo(0, 0);
        ctx.lineTo(-7, 7);
        ctx.moveTo(0, 0);
        ctx.lineTo(7, 7);
        ctx.stroke();
        break;
      case 'layers':
        ctx.beginPath();
        ctx.moveTo(0, -8);
        ctx.lineTo(8, -4);
        ctx.lineTo(8, 0);
        ctx.lineTo(0, 4);
        ctx.lineTo(-8, 0);
        ctx.lineTo(-8, -4);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-8, 2);
        ctx.lineTo(0, 6);
        ctx.lineTo(8, 2);
        ctx.stroke();
        break;
      case 'sparkles':
        ctx.beginPath();
        ctx.moveTo(0, -8);
        ctx.lineTo(-1, -1);
        ctx.lineTo(-8, 0);
        ctx.lineTo(-1, 1);
        ctx.lineTo(0, 8);
        ctx.lineTo(1, 1);
        ctx.lineTo(8, 0);
        ctx.lineTo(1, -1);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(5, -6);
        ctx.lineTo(4, -4);
        ctx.lineTo(6, -3);
        ctx.lineTo(4, -2);
        ctx.fill();
        break;
      case 'code':
        ctx.beginPath();
        ctx.moveTo(-8, 0);
        ctx.lineTo(-4, -4);
        ctx.lineTo(-4, 4);
        ctx.closePath();
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(8, 0);
        ctx.lineTo(4, -4);
        ctx.lineTo(4, 4);
        ctx.closePath();
        ctx.fill();
        break;
      case 'map':
        ctx.strokeRect(-10, -8, 20, 16);
        ctx.beginPath();
        ctx.moveTo(-10, -2);
        ctx.lineTo(10, -2);
        ctx.moveTo(-10, 4);
        ctx.lineTo(10, 4);
        ctx.moveTo(-4, -8);
        ctx.lineTo(-4, 8);
        ctx.moveTo(4, -8);
        ctx.lineTo(4, 8);
        ctx.stroke();
        break;
      case 'eye':
        ctx.beginPath();
        ctx.arc(0, 0, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(0, 0, 4, 0, Math.PI * 2);
        ctx.fill();
        break;
      case 'headphones':
        ctx.beginPath();
        ctx.arc(-5, 0, 3, 0, Math.PI * 2);
        ctx.arc(5, 0, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(-5, 0);
        ctx.lineTo(-5, -5);
        ctx.moveTo(5, 0);
        ctx.lineTo(5, -5);
        ctx.stroke();
        break;
      case 'shield':
        ctx.beginPath();
        ctx.moveTo(0, -10);
        ctx.lineTo(7, -6);
        ctx.lineTo(7, 2);
        ctx.lineTo(0, 10);
        ctx.lineTo(-7, 2);
        ctx.lineTo(-7, -6);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-3, 0);
        ctx.lineTo(0, 3);
        ctx.lineTo(5, -2);
        ctx.stroke();
        break;
      case 'grid':
        ctx.strokeRect(-8, -8, 16, 16);
        ctx.beginPath();
        ctx.moveTo(-8, -3);
        ctx.lineTo(8, -3);
        ctx.moveTo(-8, 3);
        ctx.lineTo(8, 3);
        ctx.moveTo(-3, -8);
        ctx.lineTo(-3, 8);
        ctx.moveTo(3, -8);
        ctx.lineTo(3, 8);
        ctx.stroke();
        break;
      case 'clock':
        ctx.beginPath();
        ctx.arc(0, 0, 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, -5);
        ctx.moveTo(0, 0);
        ctx.lineTo(4, 0);
        ctx.stroke();
        break;
      case 'server':
        ctx.beginPath();
        ctx.moveTo(-10, -10);
        ctx.lineTo(10, -10);
        ctx.lineTo(10, 10);
        ctx.lineTo(-10, 10);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-10, 0);
        ctx.lineTo(10, 0);
        ctx.stroke();
        break;
      case 'database':
        ctx.beginPath();
        ctx.moveTo(-10, -10);
        ctx.lineTo(10, -10);
        ctx.lineTo(10, 10);
        ctx.lineTo(-10, 10);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-10, 0);
        ctx.lineTo(10, 0);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-10, 5);
        ctx.lineTo(10, 5);
        ctx.stroke();
        break;
      case 'search':
        ctx.beginPath();
        ctx.arc(0, 0, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(8, 8);
        ctx.stroke();
        break;
      case 'message':
        ctx.beginPath();
        ctx.moveTo(-10, -10);
        ctx.lineTo(10, -10);
        ctx.lineTo(10, 10);
        ctx.lineTo(-10, 10);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(-10, 0);
        ctx.lineTo(10, 0);
        ctx.stroke();
        break;
      case 'table':
        ctx.strokeRect(-10, -8, 20, 16);
        ctx.beginPath();
        ctx.moveTo(-10, -2);
        ctx.lineTo(10, -2);
        ctx.moveTo(-10, 4);
        ctx.lineTo(10, 4);
        ctx.moveTo(-4, -8);
        ctx.lineTo(-4, 8);
        ctx.moveTo(4, -8);
        ctx.lineTo(4, 8);
        ctx.stroke();
        break;
    }

    ctx.restore();
    return canvas;
  };

  // Spawn particles - STOP when complete
  useEffect(() => {
    if (state.processingStage === 'idle' || state.processingStage === 'Analysis Complete') {
      particlesRef.current = [];
      return;
    }

    const interval = setInterval(() => {
      connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);
        
        if (!fromNode || !toNode) return;

        const isFromActive = state.activeModels.includes(conn.from);
        const isToActive = state.activeModels.includes(conn.to);

        const spawnRate = isFromActive && isToActive ? 0.4 : isFromActive ? 0.2 : 0.05;

        if (Math.random() < spawnRate) {
          const colors = getNodeColor(fromNode.type, true);
          particlesRef.current.push({
            id: `${conn.from}-${conn.to}-${Date.now()}-${Math.random()}`,
            fromX: fromNode.x,
            fromY: fromNode.y,
            toX: toNode.x,
            toY: toNode.y,
            progress: 0,
            speed: 0.01 + Math.random() * 0.015,
            color: colors.primary,
            size: 2.5 + Math.random() * 2.5,
          });
        }
      });

      if (particlesRef.current.length > 150) {
        particlesRef.current = particlesRef.current.slice(-100);
      }
    }, 60);

    return () => clearInterval(interval);
  }, [state.processingStage, state.activeModels, isDark]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.offsetWidth * dpr;
      canvas.height = canvas.offsetHeight * dpr;
      canvas.style.width = canvas.offsetWidth + 'px';
      canvas.style.height = canvas.offsetHeight + 'px';
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const animate = () => {
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;
      
      // Clear and reset transform
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const dpr = window.devicePixelRatio || 1;
      ctx.scale(dpr, dpr);
      
      // Theme-aware background
      if (isDark) {
        ctx.fillStyle = '#0F172A'; // dark slate
      } else {
        ctx.fillStyle = '#F8FAFC'; // light slate
      }
      ctx.fillRect(0, 0, width, height);
      
      if (state.processingStage !== 'idle' && state.processingStage !== 'Analysis Complete') {
        pulseRef.current += 0.04;
      }

      // Draw layer labels with theme
      const layers = [
        { x: 40, label: 'Inputs' },
        { x: 160, label: 'AI Model Stack' },
        { x: 290, label: 'Agent Orchestration Layer' },
        { x: 420, label: 'System Outputs' }
      ];

      ctx.font = '600 11px Inter, system-ui, sans-serif';
      ctx.fillStyle = isDark ? 'rgba(148, 163, 184, 0.85)' : 'rgba(71, 85, 105, 0.85)';
      ctx.textAlign = 'center';
      
      layers.forEach(layer => {
        ctx.fillText(layer.label, layer.x, 40);
      });

      // Draw connections with theme-aware styling
      connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);

        if (!fromNode || !toNode) return;

        const isActive = state.activeModels.includes(conn.from) || state.activeModels.includes(conn.to);
        const isBothActive = state.activeModels.includes(conn.from) && state.activeModels.includes(conn.to);

        if (isBothActive && state.processingStage !== 'Analysis Complete') {
          const gradient = ctx.createLinearGradient(fromNode.x, fromNode.y, toNode.x, toNode.y);
          const pulse = Math.sin(pulseRef.current + fromNode.x * 0.01) * 0.3 + 0.7;
          if (isDark) {
            gradient.addColorStop(0, `rgba(59, 130, 246, ${0.4 * pulse})`);
            gradient.addColorStop(0.5, `rgba(139, 92, 246, ${0.6 * pulse})`);
            gradient.addColorStop(1, `rgba(59, 130, 246, ${0.4 * pulse})`);
          } else {
            gradient.addColorStop(0, `rgba(29, 78, 216, ${0.5 * pulse})`);
            gradient.addColorStop(0.5, `rgba(109, 40, 217, ${0.7 * pulse})`);
            gradient.addColorStop(1, `rgba(29, 78, 216, ${0.5 * pulse})`);
          }
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 2.5;
        } else if (isActive) {
          ctx.strokeStyle = isDark ? 'rgba(59, 130, 246, 0.3)' : 'rgba(29, 78, 216, 0.4)';
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = isDark ? 'rgba(71, 85, 105, 0.2)' : 'rgba(203, 213, 225, 0.5)';
          ctx.lineWidth = 1.5;
        }

        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        
        const midX = (fromNode.x + toNode.x) / 2;
        const midY = (fromNode.y + toNode.y) / 2;
        const offsetY = Math.abs(toNode.y - fromNode.y) * 0.15;
        
        ctx.quadraticCurveTo(midX, midY - offsetY, toNode.x, toNode.y);
        ctx.stroke();
      });

      // Update and draw particles
      particlesRef.current = particlesRef.current.filter(particle => {
        particle.progress += particle.speed;
        if (particle.progress > 1) return false;

        const x = particle.fromX + (particle.toX - particle.fromX) * particle.progress;
        const y = particle.fromY + (particle.toY - particle.fromY) * particle.progress;

        const trailLength = 4;
        for (let i = 0; i < trailLength; i++) {
          const trailProgress = Math.max(0, particle.progress - i * 0.025);
          const tx = particle.fromX + (particle.toX - particle.fromX) * trailProgress;
          const ty = particle.fromY + (particle.toY - particle.fromY) * trailProgress;
          const opacity = (1 - i / trailLength) * (1 - particle.progress) * 0.8;

          const gradient = ctx.createRadialGradient(tx, ty, 0, tx, ty, particle.size);
          const alpha = Math.floor(opacity * 255).toString(16).padStart(2, '0');
          gradient.addColorStop(0, `${particle.color}${alpha}`);
          gradient.addColorStop(1, 'transparent');

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(tx, ty, particle.size * (1 - i / trailLength), 0, Math.PI * 2);
          ctx.fill();
        }

        return true;
      });

      // Draw nodes with icons
      nodes.forEach(node => {
        const isActive = state.activeModels.includes(node.id);
        const colors = getNodeColor(node.type, isActive);
        const pulse = (isActive && state.processingStage !== 'Analysis Complete') 
          ? Math.sin(pulseRef.current * 2.5 + node.x * 0.01) * 0.25 + 0.75 
          : 0.5;

        // Outer glow for active nodes
        if (isActive && state.processingStage !== 'Analysis Complete') {
          const glowSize = 45 + Math.sin(pulseRef.current * 3) * 8;
          const glowGradient = ctx.createRadialGradient(node.x, node.y, 18, node.x, node.y, glowSize);
          glowGradient.addColorStop(0, `${colors.glow}${isDark ? '50' : '60'}`);
          glowGradient.addColorStop(0.5, `${colors.glow}${isDark ? '25' : '35'}`);
          glowGradient.addColorStop(1, 'transparent');
          ctx.fillStyle = glowGradient;
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowSize, 0, Math.PI * 2);
          ctx.fill();
        }

        // Node background with gradient
        const bgGradient = ctx.createRadialGradient(
          node.x - 4, node.y - 4, 0,
          node.x, node.y, 20
        );
        if (isActive) {
          bgGradient.addColorStop(0, colors.secondary);
          bgGradient.addColorStop(1, colors.primary);
        } else {
          bgGradient.addColorStop(0, isDark ? colors.primary + 'CC' : colors.secondary);
          bgGradient.addColorStop(1, isDark ? colors.secondary + '99' : colors.primary);
        }
        
        ctx.fillStyle = bgGradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 18, 0, Math.PI * 2);
        ctx.fill();

        // Node border
        ctx.strokeStyle = isActive 
          ? (isDark ? '#FFFFFF' : '#1E293B')
          : (isDark ? 'rgba(203, 213, 225, 0.5)' : 'rgba(71, 85, 105, 0.4)');
        ctx.lineWidth = isActive ? 2.5 : 2;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 18, 0, Math.PI * 2);
        ctx.stroke();

        // Pulsing ring
        if (isActive && state.processingStage !== 'Analysis Complete') {
          ctx.strokeStyle = isDark 
            ? `rgba(255, 255, 255, ${0.3 * (1 - pulse)})`
            : `rgba(30, 41, 59, ${0.4 * (1 - pulse)})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 18 + pulse * 10, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Draw icon
        const iconColor = isActive 
          ? (isDark ? '#FFFFFF' : '#FFFFFF') 
          : (isDark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(30, 41, 59, 0.8)');
        const iconCanvas = createIconCanvas(node.icon, iconColor, 20);
        ctx.drawImage(iconCanvas, node.x - 10, node.y - 10, 20, 20);

        // Label with background
        ctx.font = '600 9px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        const labelWidth = ctx.measureText(node.label).width;
        
        // Label background
        ctx.fillStyle = isActive 
          ? (isDark ? 'rgba(0, 0, 0, 0.85)' : 'rgba(255, 255, 255, 0.95)')
          : (isDark ? 'rgba(0, 0, 0, 0.65)' : 'rgba(255, 255, 255, 0.85)');
        const padding = 5;
        ctx.fillRect(node.x - labelWidth / 2 - padding, node.y + 23, labelWidth + padding * 2, 14);
        
        // Label text
        ctx.fillStyle = isActive 
          ? (isDark ? '#FFFFFF' : '#1E293B')
          : (isDark ? 'rgba(203, 213, 225, 0.9)' : 'rgba(71, 85, 105, 0.9)');
        ctx.fillText(node.label, node.x, node.y + 33);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [state.activeModels, state.processingStage, isDark]);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full"
      style={{ background: 'transparent' }}
    />
  );
};

export default SystemArchitectureCanvas;