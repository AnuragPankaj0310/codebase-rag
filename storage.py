import json
from model import model

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

DEBUG = False
# =====================================
# CONFIG
# =====================================

COLLECTION_NAME = "code_chunks"

EMBEDDING_DIM = 768

# =====================================
# CLIENTS
# =====================================

qdrant = QdrantClient(
    host="localhost",
    port=6333
)

try:
    qdrant.get_collections()
    print("✓ Qdrant connected")
except Exception as e:
    print(f"✗ Qdrant unavailable: {e}")
    raise SystemExit(1)



# =====================================
# COLLECTION
# =====================================

def create_collection():

    collections = qdrant.get_collections()

    existing = {
        c.name
        for c in collections.collections
    }

    if COLLECTION_NAME in existing:

        print(
            f"✓ Collection exists: {COLLECTION_NAME}"
        )

        return

    print(
        f"Creating collection: {COLLECTION_NAME}"
    )

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )

    print(
        "✓ Collection created"
    )

# =====================================
# LOAD CHUNKS
# =====================================

def load_chunks():

    with open(
        "chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    print(
        f"✓ Loaded {len(chunks)} chunks"
    )

    return chunks

# =====================================
# BUILD EMBEDDING TEXT
# =====================================

def build_embedding_text(chunk):

    return f"""
Function Name:
{chunk['name']}

Entity Type:
{chunk['type']}

File Path:
{chunk['file_path']}

Parent:
{chunk.get('parent', '')}

Docstring:
{chunk.get('docstring', '')}

Imports:
{' '.join(chunk.get('imports', []))}

Calls:
{' '.join(chunk.get('calls', []))}

Inheritance:
{' '.join(chunk.get('inherits', []))}

Code:
{chunk['content']}
"""

# =====================================
# INGEST
# =====================================

def ingest_chunks():

    chunks = load_chunks()

    texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]

    print(
        "\nGenerating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(
        "✓ Embeddings generated"
    )

    points = []

    for idx, (
        chunk,
        embedding
    ) in enumerate(
        zip(
            chunks,
            embeddings
        )
    ):

        point = PointStruct(

            id=idx,

            vector=embedding.tolist(),

            payload={

                "chunk_id":
                chunk.get("id", ""),

                "name":
                chunk["name"],

                "type":
                chunk["type"],

                "is_test":
                chunk.get("is_test", False),

                "file_path":
                chunk["file_path"],

                "start_line":
                chunk["start_line"],

                "end_line":
                chunk["end_line"],

                "parent":
                chunk.get("parent"),

                "imports":
                chunk.get("imports", []),

                "calls":
                chunk.get("calls", []),

                "inherits":
                chunk.get("inherits", []),

                "decorators":
                chunk.get("decorators", []),

                "docstring":
                chunk.get("docstring", ""),

                "content":
                chunk["content"][:1000]
            }
        )

        points.append(point)

    print(
        "\nUploading vectors to Qdrant..."
    )

    BATCH_SIZE = 100

    for i in range(
        0,
        len(points),
        BATCH_SIZE
    ):

        batch = points[
            i:i + BATCH_SIZE
        ]

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )

        print(
            f"Uploaded "
            f"{min(i + BATCH_SIZE, len(points))}"
            f"/{len(points)}"
        )

    print(
        f"✓ Stored {len(points)} vectors"
    )

# =====================================
# SEARCH
# =====================================

def search(query, top_k=20):

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

    if DEBUG:

        print("\nRAW SEMANTIC RESULTS\n")

        for hit in response.points[:10]:

            print(
                hit.score,
                hit.payload["file_path"],
                hit.payload["name"]
            )

    return response.points

# =====================================
# DISPLAY RESULTS
# =====================================

def show_results(
    query,
    top_k=5
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"QUERY: {query}"
    )

    print(
        f"{'=' * 70}"
    )

    results = search(
        query,
        top_k
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print(
            f"\n[{rank}] "
            f"{payload['name']}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Type : {payload['type']}"
        )

        print(
            f"File : {payload['file_path']}"
        )

        print(
            f"Lines: "
            f"{payload['start_line']}-"
            f"{payload['end_line']}"
        )
# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    print(
        "Run ingest.py or search.py"
    )