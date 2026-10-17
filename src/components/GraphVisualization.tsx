import { useEffect, useRef } from 'react';
import { cn } from '../utils/cn';

interface Node {
  id: string;
  label: string;
  type: string;
  data?: Record<string, unknown>;
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
  /** @deprecated Dark is the only theme; kept so existing callers still compile. Ignored. */
  isDark?: boolean;
  className?: string;
}

// Node colours are design tokens (CSS variables holding RGB channels).
const NODE_TOKENS: Record<string, string> = {
  player: 'accent-2',
  team: 'pos-fwd',
  season: 'accent',
  position: 'pos-gk',
  fixture: 'text-muted',
};
const FALLBACK_TOKEN = 'text-subtle';

const CANVAS_HEIGHT = 400;
const NODE_RADIUS = 18;

/** CSS string for the legend (follows the tokens live). */
const cssColor = (token: string) => `rgb(var(--${token}))`;

/** Canvas can't resolve var(), so read the token's channels from the document. */
function canvasColor(token: string) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--${token}`).trim();
  return value ? `rgb(${value})` : 'rgb(128 128 128)';
}

export function GraphVisualization({ data, className }: GraphVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Size the canvas to the container, rendering at device pixel ratio for crisp lines.
    const width = canvas.clientWidth || containerRef.current.clientWidth;
    const height = CANVAS_HEIGHT;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const edgeColor = canvasColor('border-strong');
    const labelColor = canvasColor('text');
    const ringColor = canvasColor('surface');
    const fontFamily = getComputedStyle(document.body).fontFamily || 'sans-serif';
    const colors = new Map<string, string>();
    const colorFor = (type: string) => {
      const token = NODE_TOKENS[type] ?? FALLBACK_TOKEN;
      if (!colors.has(token)) colors.set(token, canvasColor(token));
      return colors.get(token)!;
    };

    // Initialize node positions on a circle
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

    // Simple force-directed layout simulation
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

      // Update positions with damping, keeping nodes in bounds
      nodeMap.forEach((node) => {
        node.x += node.vx;
        node.y += node.vy;
        node.vx *= 0.8;
        node.vy *= 0.8;
        node.x = Math.max(30, Math.min(width - 30, node.x));
        node.y = Math.max(30, Math.min(height - 30, node.y));
      });
    };

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Edges
      ctx.strokeStyle = edgeColor;
      ctx.lineWidth = 1.5;
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

      // Nodes
      nodeMap.forEach((node) => {
        ctx.fillStyle = colorFor(node.node.type);
        ctx.beginPath();
        ctx.arc(node.x, node.y, NODE_RADIUS, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.fillStyle = labelColor;
        ctx.font = `600 12px ${fontFamily}`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const label = node.node.label.length > 15 ? node.node.label.substring(0, 12) + '...' : node.node.label;
        ctx.fillText(label, node.x, node.y + NODE_RADIUS + 5);
      });
    };

    const STEPS = 300;
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    let rafId = 0;

    if (reduceMotion) {
      // Settle the layout off-screen and draw the final frame once.
      for (let i = 0; i < STEPS; i++) simulate();
      render();
    } else {
      let frame = 0;
      const animate = () => {
        if (frame < STEPS) {
          simulate();
          render();
          frame++;
          rafId = requestAnimationFrame(animate);
        } else {
          render();
        }
      };
      animate();
    }

    return () => cancelAnimationFrame(rafId);
  }, [data]);

  const types = Array.from(new Set([...Object.keys(NODE_TOKENS), ...data.nodes.map((n) => n.type)]));

  return (
    <div ref={containerRef} className={cn('w-full rounded-md border border-line bg-bg p-3', className)}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Knowledge graph with ${data.nodes.length} nodes and ${data.edges.length} relationships`}
        className="block w-full"
        style={{ height: CANVAS_HEIGHT }}
      />
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 border-t border-line pt-3 text-micro" aria-label="Node types">
        {types.map((type) => (
          <li key={type} className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: cssColor(NODE_TOKENS[type] ?? FALLBACK_TOKEN) }}
            />
            <span className="font-semibold uppercase text-fg-muted">{type}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
