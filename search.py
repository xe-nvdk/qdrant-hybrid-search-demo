from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
import argparse

class HybridSearcher:
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient("http://localhost:6333")
        self.qdrant_client.set_model(self.DENSE_MODEL)
        self.qdrant_client.set_sparse_model(self.SPARSE_MODEL)
        self.reranker = CrossEncoder(self.RERANK_MODEL)

    def search_with_reranking(self, text: str, user_id: int = None, limit: int = 5, rerank_top_k: int = 20):
        query_filter = None
        if user_id is not None:
            query_filter = {
                "must": [{"key": "user_id", "match": {"value": user_id}}]
            }

        hybrid_results = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=query_filter,
            limit=rerank_top_k
        )

        if not hybrid_results:
            return []

        # Step 2: Prepare pairs for re-ranking
        pairs = [[text, hit.metadata.get("text", "")] for hit in hybrid_results]

        # Step 3: Predict new scores
        rerank_scores = self.reranker.predict(pairs)

        # Step 4: Combine & sort
        reranked = [
            {
                "id": hit.id,
                "original_score": hit.score,
                "rerank_score": float(score),
                "metadata": hit.metadata
            }
            for hit, score in zip(hybrid_results, rerank_scores)
        ]

        reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:limit]
        return reranked

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="Query string")
    parser.add_argument("--user-id", type=int, help="User ID to filter on")
    args = parser.parse_args()

    query = " ".join(args.query)
    user_id = args.user_id

    print(f"\nQuery: {query}")
    if user_id:
        print(f"Filter: user_id = {user_id}")
    
    searcher = HybridSearcher("hybrid_search")
    results = searcher.search_with_reranking(query, user_id=user_id)

    if not results:
        print("No results found.")
    else:
        for i, res in enumerate(results, 1):
            print(f"\nResult #{i}")
            print(f"ID: {res['id']}")
            print(f"Original Score: {res['original_score']:.4f}")
            print(f"Re-rank Score: {res['rerank_score']:.4f}")
            print(f"User ID: {res['metadata'].get('user_id')}")
            print(f"Text: {res['metadata'].get('text', '')[:300]}")
            print("—" * 40)