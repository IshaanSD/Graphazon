import GraphBackground from "./GraphBackground";

const REPO_URL = "https://github.com/IshaanSD/Graphazon";

const TECH = [
  "LangChain",
  "Knowledge Graph Extraction",
  "Agentic AI",
  "Recommender Systems",
];

const PILLARS = [
  {
    title: "Query Understanding",
    body: "LLM chains (LangChain) parse ambiguous shopping queries to detect intent and extract attributes.",
  },
  {
    title: "Knowledge Graph",
    body: "A task-specific graph, built from structured metadata and LLM-assisted extraction, captures Exact / Substitute / Complement product relationships.",
  },
  {
    title: "Learning-to-Rank",
    body: "Graph-derived and semantic features feed a ranking model trained on real-world ESCI relevance labels.",
  },
];

function GitHubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.09 3.29 9.4 7.86 10.93.57.1.79-.25.79-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.88-1.36-3.88-1.36-.52-1.34-1.28-1.7-1.28-1.7-1.04-.72.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.23-1.28-5.23-5.71 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.18 1.18a11 11 0 0 1 2.9-.39c.98 0 1.97.13 2.9.39 2.2-1.49 3.17-1.18 3.17-1.18.63 1.6.23 2.77.11 3.06.74.8 1.19 1.83 1.19 3.09 0 4.44-2.69 5.42-5.25 5.7.41.36.78 1.06.78 2.15 0 1.55-.01 2.8-.01 3.18 0 .3.21.66.79.55A10.52 10.52 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z" />
    </svg>
  );
}

export default function App() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-graph-950">
      <GraphBackground />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-graph-950 via-graph-950/70 to-graph-950" />

      <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-6 py-24 text-center">
        <span className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium tracking-wide text-graph-200">
          Applied research project &middot; Amazon-style product search
        </span>

        <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl">
          Graph<span className="text-graph-400">azon</span>
        </h1>

        <p className="mt-6 max-w-2xl text-balance text-lg leading-relaxed text-slate-300">
          Graphazon combines <span className="text-white">query understanding</span>,{" "}
          <span className="text-white">knowledge graph construction</span>, and{" "}
          <span className="text-white">learning-to-rank models</span> for Amazon-style
          product search &mdash; using LLM reasoning and graph-derived features to map
          ambiguous queries to the right products.
        </p>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          {TECH.map((t) => (
            <span
              key={t}
              className="rounded-full border border-graph-700/60 bg-graph-900/60 px-3 py-1 text-xs font-medium text-graph-200"
            >
              {t}
            </span>
          ))}
        </div>

        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="group mt-10 inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3 font-semibold text-graph-950 shadow-lg shadow-graph-900/50 transition hover:bg-graph-100"
        >
          <GitHubIcon className="h-5 w-5" />
          View on GitHub
          <span className="transition-transform group-hover:translate-x-0.5">&rarr;</span>
        </a>

        <div className="mt-20 grid w-full gap-4 sm:grid-cols-3">
          {PILLARS.map((p) => (
            <div
              key={p.title}
              className="rounded-xl border border-white/10 bg-white/[0.03] p-5 text-left backdrop-blur-sm"
            >
              <h3 className="text-sm font-semibold text-graph-300">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{p.body}</p>
            </div>
          ))}
        </div>

        <footer className="mt-20 text-xs text-slate-500">
          Built on the{" "}
          <a
            href="https://github.com/amazon-science/esci-data"
            target="_blank"
            rel="noreferrer"
            className="underline decoration-dotted underline-offset-2 hover:text-slate-300"
          >
            Amazon Shopping Queries Dataset (ESCI)
          </a>
          . Research &amp; educational use.
        </footer>
      </div>
    </div>
  );
}
