from __future__ import annotations

from typing import List

from app.rag.retriever import RetrievedChunk


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = (
            f"[{idx}] {meta.get('job_title', 'Unknown Role')}"
            f" at {meta.get('company', 'Unknown Company')}"
            f" | {meta.get('location', 'N/A')}"
            f" | Level: {meta.get('level', 'N/A')}"
        )
        context_blocks.append(f"{header}\n{chunk.text}")

    context = "\n\n".join(context_blocks) if context_blocks else "No context found."

    return (
        "You are an expert job assistant. Use the context to answer the query. "
        "Return a concise answer and list the best-matching jobs with reasons. "
        "If the context is insufficient, say so clearly.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Respond with:\n"
        "1) Short answer (2-4 sentences).\n"
        "2) Top jobs (bullet list with title, company, location, and 1-sentence reason)."
    )
