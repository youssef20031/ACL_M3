import { useEffect, useRef } from 'react';

interface Node {
  id: string;
  label: string;
  type: string;
  data?: Record<string, any>;
}

interface Edge {
  from: string;
  to: string;
  label: string;
}

interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

interface GraphVisualizationProps {
  data: GraphData;
}

const NODE_COLORS: Record<string, string> = {
  player: '#3b82f6',
  team: '#f97316',
  season: '#10b981',
  position: '#ef4444',
  fixture: '#8b5cf6',
};

export function GraphVisualization({ data }: GraphVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const container = containerRef.current;
    canvas.width = container.clientWidth;
    canvas.height = 400;

    // Simple force-directed layout simulation
    const width = canvas.width;
    const height = canvas.height;

    // Initialize node positions
    const nodeMap = new Map<string, { x: number; y: number; vx: number; vy: number; node: Node }>();
    data.nodes.forEach((node, i) => {
      const angle = (i / data.nodes.length) * 2 * Math.PI;
      const radius = Math.min(width, height) / 3;
      nodeMap.set(node.id, {
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
        node,
      });
    });

    // Simple physics simulation
    const simulate = () => {
      // Repulsion between nodes
      nodeMap.forEach((n1, id1) => {
        nodeMap.forEach((n2, id2) => {
          if (id1 !== id2) {
            const dx = n1.x - n2.x;
            const dy = n1.y - n2.y;
            const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
            const force = 1000 / (dist * dist);
            n1.vx += (dx / dist) * force;
            n1.vy += (dy / dist) * force;
          }
        });
      });

      // Attraction along edges
      data.edges.forEach((edge) => {
        const source = nodeMap.get(edge.from);
        const target = nodeMap.get(edge.to);
        if (source && target) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) + 0.1;
          const force = dist * 0.01;
          source.vx += (dx / dist) * force;
          source.vy += (dy / dist) * force;
          target.vx -= (dx / dist) * force;
          target.vy -= (dy / dist) * force;
        }
      });

      // Center attraction
      nodeMap.forEach((node) => {
        const dx = width / 2 - node.x;
        const dy = height / 2 - node.y;
        node.vx += dx * 0.001;
        node.vy += dy * 0.001;
      });

      // Update positions with damping
      nodeMap.forEach((node) => {
        node.x += node.vx;
        node.y += node.vy;
        node.vx *= 0.8;
        node.vy *= 0.8;

        // Keep in bounds
        node.x = Math.max(30, Math.min(width - 30, node.x));
        node.y = Math.max(30, Math.min(height - 30, node.y));
      });
    };

    // Render
    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw edges
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 2;
      data.edges.forEach((edge) => {
        const source = nodeMap.get(edge.from);
        const target = nodeMap.get(edge.to);
        if (source && target) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }
      });

      // Draw nodes
      nodeMap.forEach((node) => {
        const color = NODE_COLORS[node.node.type] || '#64748b';
        
        // Node circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 20, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#1e293b';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const label = node.node.label.length > 15 
          ? node.node.label.substring(0, 12) + '...' 
          : node.node.label;
        ctx.fillText(label, node.x, node.y + 25);
      });
    };

    // Animation loop
    let frame = 0;
    const animate = () => {
      if (frame < 300) {
        simulate();
        render();
        frame++;
        requestAnimationFrame(animate);
      } else {
        render();
      }
    };

    animate();
  }, [data]);

  return (
    <div ref={containerRef} className="w-full bg-gray-50 rounded-lg p-4 border">
      <canvas ref={canvasRef} className="w-full" />
      <div className="mt-2 flex flex-wrap gap-3 text-xs">
        {Object.entries(NODE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="capitalize text-gray-600">{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}