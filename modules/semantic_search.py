# modules/semantic_search.py

from typing import List, Tuple

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

def build_sentence_index(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    Build semantic index for given texts.
    Returns (model, embeddings) or (None, None) if missing dependencies.
    """
    if SentenceTransformer is None:
        return None, None
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return model, embeddings

def semantic_search_query(
    model,
    embeddings,
    texts: List[str],
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Return top_k (text, score) matches for semantic search."""
    if model is None or embeddings is None or util is None:
        return []
    q_emb = model.encode([query], convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, embeddings)[0]
    scores = cos_scores.cpu().tolist()
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(texts[i], scores[i]) for i in ranked]
