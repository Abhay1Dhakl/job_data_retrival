"use client";

import { useState } from "react";

const hints = [
  "senior data engineer remote",
  "machine learning internship",
  "product manager healthcare",
  "frontend developer react"
];

const parseSuggestions = (text) => {
  if (!text) return [];
  const items = [];
  const normalized = text.replace(/\r\n/g, "\n");

  const jobsMatch = normalized.match(/JOBS:\s*([\s\S]*)/i);
  if (jobsMatch) {
    const lines = jobsMatch[1].split("\n").map((line) => line.trim()).filter(Boolean);
    for (const line of lines) {
      if (!line.startsWith("-")) continue;
      const trimmed = line.replace(/^-\s*/, "");
      const parts = trimmed.split("|").map((part) => part.trim());
      if (parts.length < 4) continue;
      items.push({
        role: parts[0],
        company: parts[1],
        location: parts[2],
        description: parts.slice(3).join(" | ")
      });
    }
    if (items.length > 0) return items;
  }

  const bulletRegex = /(?:^|\s)[-•]\s*\*\*(.+?)\*\*\s*:?([\s\S]*?)(?=(?:\s[-•]\s*\*\*|$))/g;
  let match;
  while ((match = bulletRegex.exec(normalized)) !== null) {
    const titleLine = match[1].trim();
    const detailsRaw = match[2].trim();
    const parts = titleLine.split(",").map((part) => part.trim()).filter(Boolean);
    let role = parts[0] || titleLine;
    let company = parts[1] || "";
    let location = parts.slice(2).join(", ");

    if (!company && (titleLine.includes(" - ") || titleLine.includes(" – "))) {
      const sep = titleLine.includes(" - ") ? " - " : " – ";
      const split = titleLine.split(sep).map((part) => part.trim());
      role = split[0] || role;
      company = split[1] || company;
    }

    const details = detailsRaw.replace(/\*\*/g, "").replace(/\s+/g, " ").trim();
    const companyMatch = details.match(/Company:\s*([^]+?)(?=Location:|Reason:|$)/i);
    const locationMatch = details.match(/Location:\s*([^]+?)(?=Reason:|$)/i);
    const reasonMatch = details.match(/Reason:\s*([^]+)$/i);

    if (companyMatch) company = companyMatch[1].trim();
    if (locationMatch) location = locationMatch[1].trim();
    const description = reasonMatch ? reasonMatch[1].trim() : details;

    items.push({ role, company, location, description });
  }
  return items;
};

const extractSummary = (text) => {
  if (!text) return "";
  const normalized = text.replace(/\r\n/g, "\n");
  const summaryMatch = normalized.match(/SUMMARY:\s*([\s\S]*?)(?:\nJOBS:|$)/i);
  if (summaryMatch) return summaryMatch[1].trim();
  const topJobsIndex = normalized.toLowerCase().indexOf("top jobs:");
  if (topJobsIndex !== -1) return normalized.slice(0, topJobsIndex).trim();
  const lines = normalized.split("\n");
  const summary = [];
  for (const line of lines) {
    if (line.trim().match(/^[-*•\d+).]\s*\*\*/)) break;
    summary.push(line);
  }
  return summary.join("\n").trim();
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [useHybrid, setUseHybrid] = useState(false);
  const [useRerank, setUseRerank] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState([]);
  const [suggestions, setSuggestions] = useState([]);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setAnswer("");
    setHits([]);
    setSuggestions([]);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: Number(topK) || 5,
          use_hybrid: useHybrid,
          use_rerank: useRerank
        })
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Request failed");
      }

      const data = await res.json();
      const responseAnswer = data.answer || "";
      setAnswer(responseAnswer);
      setSuggestions(parseSuggestions(responseAnswer));
      setHits(data.hits || []);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const summary = suggestions.length > 0 ? extractSummary(answer) : answer;

  return (
    <main>
      <div className="header">
        <div>
          <span className="tag">RAG Console</span>
          <h1 className="title">Job Data Retrieval</h1>
          <p className="subtitle">
            Search the job corpus with hybrid retrieval, reranking, and optional
            LLM synthesis.
          </p>
        </div>
        <div>
          <div className="subtitle">Backend health</div>
          <div className="meta">FastAPI + Pinecone + Redis cache</div>
        </div>
      </div>

      <section className="panel">
        <form className="form" onSubmit={submit}>
          <div>
            <div className="label">Query</div>
            <input
              className="input"
              placeholder="Search roles, locations, seniority"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              required
              minLength={3}
            />
          </div>

          <div className="hints">
            {hints.map((hint) => (
              <button
                key={hint}
                type="button"
                className="hint"
                onClick={() => setQuery(hint)}
              >
                {hint}
              </button>
            ))}
          </div>

          <div className="row">
            <div>
              <div className="label">Top K</div>
              <input
                className="input"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={useHybrid}
                onChange={(e) => setUseHybrid(e.target.checked)}
              />
              Hybrid retrieval
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={useRerank}
                onChange={(e) => setUseRerank(e.target.checked)}
              />
              Cross-encoder rerank
            </label>
          </div>

          <button className="button" type="submit" disabled={loading}>
            {loading ? "Running search..." : "Query jobs"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {summary && <div className="answer">{summary}</div>}

        {suggestions.length > 0 && (
          <>
            <div className="section-title">Suggested Roles</div>
            <div className="cards">
              {suggestions.map((item, index) => (
                <div className="card" key={`${item.role}-${index}`}>
                  <div className="card-title">{item.role}</div>
                  <div className="card-meta">
                    {item.company && <span className="chip">{item.company}</span>}
                    {item.location && <span className="chip">{item.location}</span>}
                  </div>
                  <div className="card-body">{item.description}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {hits.length > 0 && (
          <>
            <div className="section-title">Retrieved Jobs</div>
            <div className="cards">
              {hits.map((hit) => (
                <div className="card" key={hit.id}>
                  <div className="card-title">{hit.job_title}</div>
                  <div className="card-meta">
                    <span className="chip">{hit.company}</span>
                    <span className="chip">{hit.location}</span>
                    <span className="chip">{hit.level}</span>
                    <span className="chip">score {hit.score.toFixed(2)}</span>
                  </div>
                  <div className="card-body">{hit.snippet}</div>
                </div>
              ))}
            </div>
          </>
        )}

        <div className="footer">
          Tip: Enable reranking only after setting a reranker model in your
          environment.
        </div>
      </section>
    </main>
  );
}
