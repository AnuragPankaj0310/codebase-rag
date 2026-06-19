from sentence_transformers import CrossEncoder

from storage import search as semantic_search
from bm25_search import bm25_search
from hybrid_search import reciprocal_rank_fusion

print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

DEBUG = False

# =====================================
# RERANK
# =====================================

def rerank_results(
    query,
    results,
    top_k=5
):
    """
    Rerank retrieved chunks using CrossEncoder.
    """

    pairs = []

    for result in results:

        document = f"""
Name:
{result['name']}

Type:
{result['type']}

File:
{result['file_path']}

Code:
{result.get('content', '')[:1000]}
"""

        pairs.append(
            [query, document]
        )

    if DEBUG:

        print("\nDEBUG DOCUMENTS\n")

        for i, pair in enumerate(
            pairs[:5],
            start=1
        ):
            print(
                f"\n[{i}]"
            )

            print(
                pair[1][:500]
            )

            print(
                "\n" + "-" * 50
            )
    
    scores = reranker.predict(
        pairs
    )

    reranked = []

    for result, score in zip(
        results,
        scores
    ):

        item = result.copy()

        item["rerank_score"] = float(score)

        semantic_score = float(
            result.get("score", 0)
        )

        item["final_score"] = (
            semantic_score * 0.30
            + float(score) * 0.70
        )

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return reranked[:top_k]

# =====================================
# FULL PIPELINE
# =====================================

def hybrid_search_with_rerank(
    query,
    top_k=5
):

    semantic = semantic_search(
        query,
        top_k=20
    )

    bm25 = bm25_search(
        query,
        top_k=20
    )

    fused = reciprocal_rank_fusion(
        semantic,
        bm25
    )

    reranked = rerank_results(
        query,
        fused[:50],
        top_k=top_k
    )

    return reranked

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    queries = [

        "how does flask create app",

        "request routing",

        "url route decorators",

        "celery task implementation"
    ]

    for query in queries:

        print(
            "\n" + "=" * 70
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "=" * 70
        )

        results = hybrid_search_with_rerank(
            query,
            top_k=5
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n[{rank}] "
                f"{result['name']}"
            )

            print(
                f"Rerank Score: "
                f"{result['rerank_score']:.4f}"
            )

            print(
                f"File: "
                f"{result['file_path']}"
            )

            print(
                f"Lines: "
                f"{result['start_line']}-"
                f"{result['end_line']}"
            )