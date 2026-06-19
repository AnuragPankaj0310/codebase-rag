import re
import json
from rank_bm25 import BM25Okapi

# =====================================
# LOAD CHUNKS
# =====================================

with open(
    "chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

# =====================================
# BUILD DOCUMENTS
# =====================================

documents = []

for chunk in chunks:

    name = chunk["name"]
    docstring = chunk.get("docstring", "")
    content = chunk["content"]
    file_path = chunk["file_path"]

    doc = f"""
    {name} {name} {name} {name} {name}

    Type:
    {chunk['type']}

    File:
    {file_path}

    Parent:
    {chunk.get('parent', '')}

    Imports:
    {' '.join(chunk.get('imports', []))}

    Calls:
    {' '.join(chunk.get('calls', []))}

    Decorators:
    {' '.join(chunk.get('decorators', []))}

    Inheritance:
    {' '.join(chunk.get('inherits', []))}

    Docstring:
    {docstring}

    Code:
    {content}
    """

    documents.append(doc)

# =====================================
# TOKENIZE
# =====================================

# tokenized_docs = [
#     doc.lower().split()
#     for doc in documents
# ]

tokenized_docs = [
    re.findall(r"\w+", doc.lower())
    for doc in documents
]

bm25 = BM25Okapi(
    tokenized_docs
)

# =====================================
# SEARCH
# =====================================

def bm25_search(
    query,
    top_k=20
):

    # query_tokens = (
    #     query
    #     .lower()
    #     .split()
    # )

    query_tokens = re.findall(
        r"\w+",
        query.lower()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    results_dict = {}

    for idx, score in ranked:

        chunk = chunks[idx]

        key = (
            f"{chunk['file_path']}::"
            f"{chunk['name']}::"
            f"{chunk['start_line']}"
        )

        if key not in results_dict:

            results_dict[key] = {

                "score":
                float(score),

                "chunk":
                chunk
            }

        if len(results_dict) >= top_k:
            break

    return list(results_dict.values())

# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    queries = [

        "create_app",

        "route decorator",

        "dispatch_request",

        "celery task"
    ]

    for query in queries:

        print("\n" + "=" * 60)

        print(
            f"QUERY: {query}"
        )

        print("=" * 60)

        results = bm25_search(
            query,
            top_k=3
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            chunk = result["chunk"]

            print(
                f"\n[{rank}] "
                f"{chunk['name']}"
            )

            print(
                f"Score: {result['score']:.2f}"
            )

            print(
                f"{chunk['file_path']}:{chunk['start_line']}"
            )