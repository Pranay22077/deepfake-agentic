import React, { useEffect, useRef } from 'react';
import { useArchitecture } from '../context/ArchitectureContext';

interface Node {
  id: string;
  x: number;
  y: number;
  label: string;
  type: 'input' | 'model' | 'agent' | 'output';
  icon: 'square' | 'circle' | 'triangle' | 'diamond';
  labelPosition: 'left' | 'right' | 'bottom';
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
}

const SystemArchitectureCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { state } = useArchitecture();
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  // Exact layout matching the reference image
  const nodes: Node[] = [
    // Layer 1: Input Processing (Green) - Labels on LEFT
    { id: 'video-input', x: 100, y: 65, label: 'Video Input', type: 'input', icon: 'square', labelPosition: 'left' },
    { id: 'video-decoder', x: 100, y: 125, label: 'Video Decoder', type: 'input', icon: 'square', labelPosition: 'left' },
    { id: 'frame-sampler', x: 100, y: 195, label: 'Frame Sampler', type: 'input', icon: 'square', labelPosition: 'left' },
    { id: 'face-detector', x: 100, y: 265, label: 'Face Detector', type: 'input', icon: 'square', labelPosition: 'left' },

    // Layer 2: Model Bank (Blue) - Labels on RIGHT
    { id: 'bg-model', x: 300, y: 55, label: 'BG-Model', type: 'model', icon: 'circle', labelPosition: 'right' },
    { id: 'av-model', x: 300, y: 110, label: 'AV-Model', type: 'model', icon: 'circle', labelPosition: 'right' },
    { id: 'cm-model', x: 300, y: 165, label: 'CM-Model', type: 'model', icon: 'circle', labelPosition: 'right' },
    { id: 'rr-model', x: 300, y: 220, label: 'RR-Model', type: 'model', icon: 'circle', labelPosition: 'right' },
    { id: 'll-model', x: 300, y: 275, label: 'LL-Model', type: 'model', icon: 'circle', labelPosition: 'right' },
    { id: 'tm-model', x: 300, y: 330, label: 'TM-Model', type: 'model', icon: 'circle', labelPosition: 'right' },

    // Layer 3: Agentic Intelligence (Purple) - Labels on RIGHT
    { id: 'routing-engine', x: 520, y: 75, label: 'Routing Engine', type: 'agent', icon: 'triangle', labelPosition: 'right' },
    { id: 'langgraph', x: 520, y: 135, label: 'LangGraph Agent', type: 'agent', icon: 'triangle', labelPosition: 'right' },
    { id: 'aggregator', x: 520, y: 205, label: 'Aggregator', type: 'agent', icon: 'triangle', labelPosition: 'right' },
    { id: 'bias-correction', x: 520, y: 275, label: 'Bias Correction', type: 'agent', icon: 'triangle', labelPosition: 'right' },

    // Layer 4: Output (Orange) - Labels on RIGHT
    { id: 'fastapi', x: 720, y: 140, label: 'FastAPI', type: 'output', icon: 'diamond', labelPosition: 'right' },
    { id: 'react-ui', x: 720, y: 220, label: 'React UI', type: 'output', icon: 'diamond', labelPosition: 'right' },
  ];

  const connections: Connection[] = [
    // Input layer vertical connections
    { from: 'video-input', to: 'video-decoder' },
    { from: 'video-decoder', to: 'frame-sampler' },
    { from: 'frame-sampler', to: 'face-detector' },
    
    // Input to Models (fan out)
    { from: 'video-input', to: 'bg-model' },
    { from: 'video-decoder', to: 'av-model' },
    { from: 'frame-sampler', to: 'cm-model' },
    { from: 'frame-sampler', to: 'rr-model' },
    { from: 'frame-sampler', to: 'tm-model' },
    { from: 'face-detector', to: 'll-model' },
    { from: 'face-detector', to: 'rr-model' },
    { from: 'face-detector', to: 'cm-model' },
    { from: 'face-detector', to: 'tm-model' },
    
    // Models to Agents
    { from: 'bg-model', to: 'routing-engine' },
    { from: 'av-model', to: 'langgraph' },
    { from: 'cm-model', to: 'aggregator' },
    { from: 'rr-model', to: 'aggregator' },
    { from: 'll-model', to: 'bias-correction' },
    { from: 'tm-model', to: 'aggregator' },
    
    // Agent interconnections
    { from: 'routing-engine', to: 'langgraph' },
    { from: 'langgraph', to: 'aggregator' },
    { from: 'aggregator', to: 'bias-correction' },
    
    // Agents to Output
    { from: 'routing-engine', to: 'fastapi' },
    { from: 'langgraph', to: 'fastapi' },
    { from: 'aggregator', to: 'react-ui' },
    { from: 'bias-correction', to: 'react-ui' },
  ];

  const getNodeColor = (type: string) => {
    const colors = {
      input: '#10B981',    // Green
      model: '#3B82F6',    // Blue  
      agent: '#8B5CF6',    // Purple
      output: '#F59E0B',   // Orange
    };
    return colors[type as keyof typeof colors] || '#6B7280';
  };

  // Draw icon inside node
  const drawIcon = (ctx: CanvasRenderingContext2D, x: number, y: number, icon: string, color: string) => {
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    switch (icon) {
      case 'square':
        ctx.fillRect(x - 5, y - 5, 10, 10);
        break;
      case 'circle':
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        break;
      case 'triangle':
        ctx.beginPath();
        ctx.moveTo(x, y - 7);
        ctx.lineTo(x + 7, y + 5);
        ctx.lineTo(x - 7, y + 5);
        ctx.closePath();
        ctx.fill();
        break;
      case 'diamond':
        ctx.beginPath();
        ctx.moveTo(x, y - 7);
        ctx.lineTo(x + 7, y);
        ctx.lineTo(x, y + 7);
        ctx.lineTo(x - 7, y);
        ctx.closePath();
        ctx.fill();
        break;
    }
  };

  // Spawn flowing particles
  useEffect(() => {
    if (state.processingStage === 'idle') return;

    const interval = setInterval(() => {
      connections.forEach(conn => {
        if (state.activeModels.includes(conn.from) && Math.random() < 0.06) {
          const fromNode = nodes.find(n => n.id === conn.from);
          const toNode = nodes.find(n => n.id === conn.to);

          if (fromNode && toNode) {
            particlesRef.current.push({
              id: `${conn.from}-${conn.to}-${Date.now()}-${Math.random()}`,
              fromX: fromNode.x,
              fromY: fromNode.y,
              toX: toNode.x,
              toY: toNode.y,
              progress: 0,
              speed: 0.012 + Math.random() * 0.008,
            });
          }
        }
      });
    }, 100);

    return () => clearInterval(interval);
  }, [state.processingStage, state.activeModels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      timeRef.current += 0.03;

      // Draw connections with small dots
      connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);
        if (!fromNode || !toNode) return;

        const isActive = state.activeModels.includes(conn.from) || state.activeModels.includes(conn.to);
        
        // Connection line
        ctx.strokeStyle = isActive ? 'rgba(59, 130, 246, 0.35)' : 'rgba(148, 163, 184, 0.2)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();

        // Small green dots along the line (like in reference)
        if (isActive) {
          const dx = toNode.x - fromNode.x;
          const dy = toNode.y - fromNode.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const numDots = Math.floor(distance / 20);
          
          for (let i = 1; i < numDots; i++) {
            const t = i / numDots;
            const dotX = fromNode.x + dx * t;
            const dotY = fromNode.y + dy * t;
            
            // Subtle pulse
            const pulse = Math.sin(timeRef.current * 2 + i * 0.8) * 0.25 + 0.75;
            
            ctx.fillStyle = `rgba(16, 185, 129, ${0.7 * pulse})`;
            ctx.beginPath();
            ctx.arc(dotX, dotY, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      });

      // Draw flowing particles
      particlesRef.current = particlesRef.current.filter(particle => {
        particle.progress += particle.speed;
        if (particle.progress > 1) return false;

        const x = particle.fromX + (particle.toX - particle.fromX) * particle.progress;
        const y = particle.fromY + (particle.toY - particle.fromY) * particle.progress;

        // Glowing green particle
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 5);
        gradient.addColorStop(0, 'rgba(16, 185, 129, 0.9)');
        gradient.addColorStop(0.5, 'rgba(16, 185, 129, 0.4)');
        gradient.addColorStop(1, 'transparent');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();

        return true;
      });

      // Draw nodes exactly like reference
      nodes.forEach(node => {
        const isActive = state.activeModels.includes(node.id);
        const nodeRadius = 18;
        const color = getNodeColor(node.type);

        // Glow effect for active nodes
        if (isActive) {
          const glowGradient = ctx.createRadialGradient(node.x, node.y, nodeRadius - 5, node.x, node.y, nodeRadius + 15);
          glowGradient.addColorStop(0, color + '50');
          glowGradient.addColorStop(1, 'transparent');
          ctx.fillStyle = glowGradient;
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius + 15, 0, Math.PI * 2);
          ctx.fill();
        }

        // Outer colored ring
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
        ctx.fill();

        // Inner white circle
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeRadius - 5, 0, Math.PI * 2);
        ctx.fill();

        // Icon inside
        drawIcon(ctx, node.x, node.y, node.icon, color);

        // Label with proper positioning
        ctx.font = '600 11px Inter, -apple-system, sans-serif';
        ctx.fillStyle = '#374151';
        
        if (node.labelPosition === 'left') {
          ctx.textAlign = 'right';
          ctx.fillText(node.label, node.x - nodeRadius - 8, node.y + 4);
        } else if (node.labelPosition === 'right') {
          ctx.textAlign = 'left';
          ctx.fillText(node.label, node.x + nodeRadius + 8, node.y + 4);
        } else {
          ctx.textAlign = 'center';
          ctx.fillText(node.label, node.x, node.y + nodeRadius + 14);
        }
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
  }, [state.activeModels]);

  return <canvas ref={canvasRef} className="w-full h-full" style={{ minHeight: '340px' }} />;
};

export default SystemArchitectureCanvas;
