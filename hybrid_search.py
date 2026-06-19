"""
hybrid_search.py — Reciprocal Rank Fusion of semantic + BM25 results.

Path-based boosting/penalizing is now driven by rag_config.json
so it generalises across repos without hardcoded directory names.
"""

import os
import json
from storage import search as semantic_search
from bm25_search import bm25_search

RRF_K = 60
DEBUG = False

# =====================================
# CONFIG
# =====================================

def load_config(path="rag_config.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "boost_dirs": ["src", "lib", "core", "app"],
        "penalize_dirs": ["tests", "test", "docs", "benchmarks"],
    }

_CONFIG = load_config()
BOOST_DIRS   = _CONFIG.get("boost_dirs",   ["src", "lib", "core", "app"])
PENALIZE_DIRS = _CONFIG.get("penalize_dirs", ["tests", "test", "docs", "benchmarks"])


# =====================================
# RRF
# =====================================

def reciprocal_rank_fusion(semantic_results, bm25_results):

    fused = {}

    # Semantic results — double weight
    for rank, result in enumerate(semantic_results, start=1):
        payload = result.payload
        chunk_id = payload["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                "id":         chunk_id,
                "name":       payload["name"],
                "type":       payload["type"],
                "file_path":  payload["file_path"],
                "start_line": payload["start_line"],
                "end_line":   payload["end_line"],
                "content":    payload.get("content", ""),
                "score":      0.0,
            }

        fused[chunk_id]["score"] += 2 * (1 / (RRF_K + rank))

    # BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        chunk = result["chunk"]
        chunk_id = chunk["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                "id":         chunk_id,
                "name":       chunk["name"],
                "type":       chunk["type"],
                "file_path":  chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line":   chunk["end_line"],
                "content":    chunk.get("content", ""),
                "score":      0.0,
            }

        fused[chunk_id]["score"] += 1 / (RRF_K + rank)

    # Path-based boosting — driven by config, not hardcoded names
    for result in fused.values():
        fp = result["file_path"].replace("\\", "/").lower()

        for d in PENALIZE_DIRS:
            if f"/{d}/" in fp or fp.startswith(f"{d}/") or f"/{d}_" in fp:
                result["score"] *= 0.5
                break

        for d in BOOST_DIRS:
            if f"/{d}/" in fp or fp.startswith(f"{d}/"):
                result["score"] *= 1.5
                break

    if DEBUG:
        print("\nDEBUG FUSED RESULTS\n")
        for item in sorted(fused.values(), key=lambda x: x["score"], reverse=True):
            print(f"{item['score']:.4f}  {item['file_path']}  {item['name']}")

    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)


# =====================================
# MAIN (test)
# =====================================

if __name__ == "__main__":
    query = "create_app"
    semantic = semantic_search(query, top_k=20)
    bm25     = bm25_search(query, top_k=20)
    results  = reciprocal_rank_fusion(semantic, bm25)

    print("\nHybrid Results\n")
    for rank, result in enumerate(results[:10], start=1):
        print(f"\n[{rank}] {result['name']}")
        print(f"Score : {result['score']:.4f}")
        print(f"File  : {result['file_path']}")
        print(f"Lines : {result['start_line']}-{result['end_line']}")