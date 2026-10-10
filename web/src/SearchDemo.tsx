import { useEffect, useState } from "react";
import { ApiError, fetchExamples, submitSearch, type QueryUnderstandingResult } from "./api";

const MAX_QUERY_LENGTH = 200;

function AttributeChips({ label, values }: { label: string; values: string[] }) {
  if (values.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-xs font-medium text-slate-500">{label}:</span>
      {values.map((v) => (
        <span
          key={v}
          className="rounded-full border border-graph-700/60 bg-graph-900/60 px-2.5 py-0.5 text-xs text-graph-200"
        >
          {v}
        </span>
      ))}
    </div>
  );
}

function ResultCard({ result }: { result: QueryUnderstandingResult }) {
  return (
    <div className="mt-6 w-full rounded-xl border border-white/10 bg-white/[0.03] p-5 text-left backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm text-slate-400">
          &ldquo;{result.raw_query}&rdquo;
        </span>
        <span className="rounded-full border border-graph-500/50 bg-graph-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-graph-300">
          {result.intent.replace("_", " ")}
        </span>
      </div>

      <p className="mt-3 text-sm text-slate-300">
        Rewritten: <span className="text-white">{result.rewritten_query}</span>
      </p>

      <div className="mt-4 flex flex-col gap-2">
        {result.product_type && (
          <AttributeChips label="Product type" values={[result.product_type]} />
        )}
        <AttributeChips label="Brands" values={result.brands} />
        <AttributeChips label="Colors" values={result.colors} />
        <AttributeChips label="Constraints" values={result.constraints} />
        <AttributeChips label="Compatibility" values={result.compatibility} />
      </div>
    </div>
  );
}

export default function SearchDemo() {
  const [examples, setExamples] = useState<QueryUnderstandingResult[]>([]);
  const [examplesError, setExamplesError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryUnderstandingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchExamples()
      .then(setExamples)
      .catch((e: unknown) => {
        setExamplesError(e instanceof Error ? e.message : "Couldn't load example queries.");
      });
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    try {
      const res = await submitSearch(trimmed, "us");
      setResult(res);
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 429) {
        setError("Demo limit reached - try an example query below, or wait a minute.");
      } else {
        setError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="mt-20 w-full">
      <h2 className="text-lg font-semibold text-white">Try query understanding live</h2>
      <p className="mt-1 text-sm text-slate-400">
        Type a shopping query, or pick an example, to see the live LLM pipeline&apos;s
        predicted intent, rewritten query, and extracted attributes.
      </p>

      {examples.length > 0 && (
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {examples.map((ex) => (
            <button
              key={ex.query_id}
              type="button"
              onClick={() => {
                setResult(ex);
                setError(null);
              }}
              className="rounded-full border border-graph-700/60 bg-graph-900/60 px-3 py-1.5 text-xs font-medium text-graph-200 transition hover:bg-graph-800/60"
            >
              {ex.raw_query}
            </button>
          ))}
        </div>
      )}
      {examplesError && <p className="mt-3 text-xs text-slate-500">{examplesError}</p>}

      <form onSubmit={handleSubmit} className="mt-6 flex items-center gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          maxLength={MAX_QUERY_LENGTH}
          placeholder="e.g. waterproof hiking boots for men"
          className="flex-1 rounded-lg border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm text-white placeholder:text-slate-500 focus:border-graph-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="rounded-lg bg-white px-5 py-2.5 text-sm font-semibold text-graph-950 shadow-lg shadow-graph-900/50 transition hover:bg-graph-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <p className="mt-3 text-sm text-amber-400">{error}</p>}
      {result && <ResultCard result={result} />}
    </section>
  );
}
