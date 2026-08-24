const NODES = [
  { x: 8, y: 18 }, { x: 22, y: 8 }, { x: 38, y: 22 }, { x: 55, y: 10 },
  { x: 70, y: 26 }, { x: 88, y: 14 }, { x: 15, y: 45 }, { x: 33, y: 55 },
  { x: 50, y: 42 }, { x: 66, y: 58 }, { x: 82, y: 48 }, { x: 94, y: 66 },
  { x: 12, y: 78 }, { x: 30, y: 85 }, { x: 47, y: 74 }, { x: 62, y: 88 },
  { x: 78, y: 80 }, { x: 90, y: 92 },
];

const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 6], [1, 8], [2, 8],
  [3, 8], [4, 9], [5, 10], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
  [6, 12], [7, 13], [8, 14], [9, 15], [10, 16], [11, 17], [12, 13],
  [13, 14], [14, 15], [15, 16], [16, 17],
];

/**
 * Faint knowledge-graph motif used as a full-bleed background decoration.
 * Purely decorative (aria-hidden) — content sits in a layer above it.
 */
export default function GraphBackground() {
  return (
    <svg
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.35]"
      viewBox="0 0 100 100"
      preserveAspectRatio="xMidYMid slice"
    >
      {EDGES.map(([a, b], i) => (
        <line
          key={i}
          x1={NODES[a].x}
          y1={NODES[a].y}
          x2={NODES[b].x}
          y2={NODES[b].y}
          stroke="url(#edge-gradient)"
          strokeWidth="0.15"
        />
      ))}
      {NODES.map((n, i) => (
        <circle key={i} cx={n.x} cy={n.y} r={i % 3 === 0 ? 0.7 : 0.45} fill="#9bbfff" />
      ))}
      <defs>
        <linearGradient id="edge-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#4372fb" />
          <stop offset="100%" stopColor="#9bbfff" />
        </linearGradient>
      </defs>
    </svg>
  );
}
