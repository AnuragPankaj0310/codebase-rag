from qdrant_client import QdrantClient

q = QdrantClient(
    host="localhost",
    port=6333
)

print(q.get_collections())