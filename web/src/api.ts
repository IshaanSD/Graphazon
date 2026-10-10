const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export interface QueryUnderstandingResult {
  query_id: string;
  raw_query: string;
  locale: string;
  intent: string;
  rewritten_query: string;
  product_type: string | null;
  brands: string[];
  colors: string[];
  constraints: string[];
  compatibility: string[];
  model_name: string;
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch {
    // fall through to generic message
  }
  return `Request failed (${res.status})`;
}

export async function fetchExamples(): Promise<QueryUnderstandingResult[]> {
  const res = await fetch(`${API_BASE_URL}/examples`);
  if (!res.ok) throw new ApiError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function submitSearch(
  query: string,
  locale: string = "us",
): Promise<QueryUnderstandingResult> {
  const res = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, locale }),
  });
  if (!res.ok) throw new ApiError(res.status, await parseErrorDetail(res));
  return res.json();
}
