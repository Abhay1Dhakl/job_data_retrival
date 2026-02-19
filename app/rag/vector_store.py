from __future__ import annotations

from typing import Any, Dict, List

import chromadb


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embeddings: List[List[float]], n_results: int) -> List[List[Dict[str, Any]]]:
        if not query_embeddings:
            return []
        result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances", "ids"],
        )
        hits: List[List[Dict[str, Any]]] = []
        for i in range(len(result.get("ids", []))):
            ids = result["ids"][i]
            docs = result["documents"][i]
            metas = result["metadatas"][i]
            dists = result.get("distances", [[]])[i]
            row: List[Dict[str, Any]] = []
            for j, doc_id in enumerate(ids):
                distance = dists[j] if j < len(dists) else 0.0
                score = 1.0 - float(distance)
                row.append(
                    {
                        "id": doc_id,
                        "document": docs[j],
                        "metadata": metas[j],
                        "score": score,
                    }
                )
            hits.append(row)
        return hits

    def count(self) -> int:
        return self._collection.count()
