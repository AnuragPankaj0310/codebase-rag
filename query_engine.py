# from reranker import hybrid_search_with_rerank
from hybrid_search import (
    reciprocal_rank_fusion
)

from storage import search as semantic_search
from bm25_search import bm25_search


from graph_traversal import (
    analyze_impact_by_id,
    display_impact,
    load_graph,
    build_graph_maps
)


def query_with_impact(
    question,
    top_k=5
):
    """
    Unified query:
    Retrieval + Impact Analysis
    """

    print(
        f"\n QUERY: {question}\n"
    )

    # -----------------------
    # Search
    # -----------------------

    # results = (
    #     hybrid_search_with_rerank(
    #         question,
    #         top_k=top_k
    #     )
    # )

    semantic = semantic_search(
    question,
    top_k=20
)

    bm25 = bm25_search(
        question,
        top_k=20
    )

    # results = reciprocal_rank_fusion(
    #     semantic,
    #     bm25
    # )
    print("\nSEMANTIC TOP 5\n")

    for r in semantic[:5]:

        print(
            r.payload["file_path"],
            r.payload["name"]
        )

    print("\nBM25 TOP 5\n")

    for r in bm25[:5]:

        print(
            r["chunk"]["file_path"],
            r["chunk"]["name"]
        )

    results = reciprocal_rank_fusion(
        semantic,
        bm25
    )

    results = results[:top_k]

    print(
        f"Top {len(results)} results:\n"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"[{rank}] "
            f"{result['name']}"
        )

        print(
            f"     "
            f"{result['file_path']}"
        )

        print(
            f"     "
            f"Score: "
            f"{result['score']:.4f}\n"
        )

    # -----------------------
    # Impact Analysis
    # -----------------------

    if not results:
        return

    chunks, edges = load_graph()

    (
        chunk_map,
        name_to_ids,
        forward,
        reverse
    ) = build_graph_maps(
        chunks,
        edges
    )

    top_chunk_id = (
    results[0]["id"]
)

    print(
        f"\nSelected chunk:"
        f"\n{top_chunk_id}"
    )

    impact = analyze_impact_by_id(
        top_chunk_id,
        chunks,
        forward,
        reverse
    )

    if impact:

        print(
            "\n IMPACT ANALYSIS"
        )

        display_impact(
            impact
        )


if __name__ == "__main__":

    while True:

        query = input(
            "\nAsk a repository question (or 'quit'): "
        )

        if query.lower() == "quit":
            break

        query_with_impact(
            query,
            top_k=5
        )