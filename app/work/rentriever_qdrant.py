from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import config
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchParams, NamedVector, NamedVectors
from sentence_transformers import SentenceTransformer

@dataclass
class Hit:
    text: str
    payload: Dict[str, Any]
    score: float

class QdrantRetriever:
    def __init__(self):
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=(config.QDRANT_API_KEY or None),
            timeout=30
        )
        self.embed = SentenceTransformer(config.EMB_MODEL, device=config.EMB_DEVICE)
        self.vector_name: Optional[str] = None  # напр. "text"

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"]
        for key in ("node", "document", "doc", "data"):
            obj = payload.get(key)
            if isinstance(obj, dict):
                for tkey in ("text", "content", "body"):
                    if tkey in obj and isinstance(obj[tkey], str):
                        return obj[tkey]
        return str(payload)

    def search(self, query: str, top_k: int = None, min_score: float = None) -> List[Hit]:
        top_k = top_k or config.TOP_K
        min_score = config.MIN_SCORE if min_score is None else min_score

        q_vec = self.embed.encode([query], normalize_embeddings=True)[0].tolist()

        if self.vector_name:
            results = self.client.search(
                collection_name=config.QDRANT_COLLECTION,
                query=NamedVector(name=self.vector_name, vector=q_vec),
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128, exact=False),
                with_payload=True,
            )
        else:
            results = self.client.search(
                collection_name=config.QDRANT_COLLECTION,
                query_vector=q_vec,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128, exact=False),
                with_payload=True,
            )

        hits: List[Hit] = []
        for r in results:
            score = float(r.score) if hasattr(r, "score") else 0.0
            if score < min_score:
                continue
            payload = r.payload or {}
            text = self._extract_text(payload)[:2000] 
            hits.append(Hit(text=text, payload=payload, score=score))
        return hits
