from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from fastapi import APIRouter, Depends

from app.core.cache import get_cache
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


def _cache_key(payload: QueryRequest, top_k: int, use_hybrid: bool, use_rerank: bool) -> str:
    blob = json.dumps(
        {
            "query": payload.query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
        },
        sort_keys=True,
    )
    return f"query:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"


@router.post("/api/query", response_model=QueryResponse)
def query_jobs(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> QueryResponse:
    top_k = payload.top_k or settings.top_k
    use_hybrid = payload.use_hybrid if payload.use_hybrid is not None else settings.use_hybrid
    use_rerank = payload.use_rerank if payload.use_rerank is not None else bool(settings.rerank_model)

    cache = get_cache(settings)
    cache_key = None
    if cache and settings.cache_ttl_seconds > 0:
        cache_key = _cache_key(payload, top_k, use_hybrid, use_rerank)
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                return QueryResponse.model_validate_json(cached)
            except Exception:
                pass

    answer, results = pipeline.run(
        query=payload.query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )

    hits = [_to_hit(chunk) for chunk in results]
    response = QueryResponse(answer=answer, hits=hits)

    if cache and cache_key and settings.cache_ttl_seconds > 0:
        try:
            cache.setex(cache_key, settings.cache_ttl_seconds, response.model_dump_json())
        except Exception:
            pass

    return response
