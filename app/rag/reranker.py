from __future__ import annotations

from typing import List, Optional

from sentence_transformers import CrossEncoder

from app.rag.retriever import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        reranked = [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]
        return sorted(reranked, key=lambda c: c.score, reverse=True)


def build_reranker(model_name: Optional[str]) -> Optional[CrossEncoderReranker]:
    if not model_name:
        return None
    return CrossEncoderReranker(model_name)
