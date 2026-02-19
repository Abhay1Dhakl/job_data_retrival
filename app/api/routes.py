from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.rag.pipeline import RagPipeline, build_pipeline
from app.rag.schemas import JobHit, QueryRequest, QueryResponse

router = APIRouter()


@lru_cache
def get_pipeline() -> RagPipeline:
    settings = get_settings()
    return build_pipeline(settings)


def _to_hit(chunk) -> JobHit:
    meta = chunk.metadata
    snippet = chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
    return JobHit(
        id=chunk.id,
        score=chunk.score,
        job_title=str(meta.get("job_title", "")),
        company=str(meta.get("company", "")),
        location=str(meta.get("location", "")),
        level=str(meta.get("level", "")),
        snippet=snippet,
    )


@router.post("/api/query", response_model=QueryResponse)
def query_jobs(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> QueryResponse:
    top_k = payload.top_k or settings.top_k
    use_hybrid = payload.use_hybrid if payload.use_hybrid is not None else settings.use_hybrid
    use_rerank = payload.use_rerank if payload.use_rerank is not None else bool(settings.rerank_model)

    answer, results = pipeline.run(
        query=payload.query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )

    hits = [_to_hit(chunk) for chunk in results]
    return QueryResponse(answer=answer, hits=hits)
