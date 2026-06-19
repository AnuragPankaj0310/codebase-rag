import json

# Load your data
with open(
    "chunks.json",
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:
    chunks = json.load(f)

with open(
    "edges.json",
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:
    edges = json.load(f)

print(f"\n VALIDATION REPORT\n")
print(f"Total Chunks: {len(chunks)}")
print(f"Total Edges: {len(edges)}")

# Check 1: Decorators
chunks_with_decorators = [c for c in chunks if c.get("decorators")]
print(f"\n✓ Chunks with decorators: {len(chunks_with_decorators)}")
if chunks_with_decorators:
    print(f"  Sample: {chunks_with_decorators[0]['name']} -> {chunks_with_decorators[0]['decorators']}")

# Check 2: Inheritance
chunks_with_inheritance = [c for c in chunks if c.get("inherits")]
print(f"\n✓ Classes with inheritance: {len(chunks_with_inheritance)}")
if chunks_with_inheritance:
    print(f"  Sample: {chunks_with_inheritance[0]['name']} -> {chunks_with_inheritance[0]['inherits']}")

# Check 3: Duplicate edges
edge_tuples = [(e["source"], e["target"], e["type"]) for e in edges]
unique_edges = len(set(edge_tuples))
print(f"\n✓ Unique edges: {unique_edges} (out of {len(edges)})")
if unique_edges < len(edges):
    print(f"  Found {len(edges) - unique_edges} duplicate edges")

# Check 4: Edge distribution by type
edge_types = {}
for e in edges:
    t = e["type"]
    edge_types[t] = edge_types.get(t, 0) + 1

print(f"\n✓ Edge distribution:")
for edge_type, count in sorted(edge_types.items()):
    print(f"    {edge_type}: {count}")

# Check 5: Sanity check - are chunks referenced in edges?
chunk_names = {c["name"] for c in chunks}
edge_sources = {e["source"] for e in edges}
referenced = edge_sources & chunk_names
print(f"\n✓ Chunks referenced in edges: {len(referenced)} / {len(chunk_names)}")

print(f"\n Validation complete\n")