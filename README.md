# Codebase RAG

Repository-aware Retrieval-Augmented Generation system for codebase understanding.

## Current Status

### Indexed Repositories

* Flask
* Werkzeug

### Current Graph Statistics

* Chunks: ~4,056
* Edges: ~15,670
* Calls: ~12,791
* Resolved Calls: ~3,806

### Stack

* Python
* Qdrant
* Sentence Transformers
* Cross Encoder Reranking
* AST-based Code Parsing

## Pipeline

Repository Source Code
↓
AST Chunking
↓
Symbol Extraction
↓
Call Graph Construction
↓
Embedding Generation
↓
Qdrant Storage
↓
Hybrid Retrieval
↓
Cross Encoder Reranking
↓
LLM Answer Generation

## Setup

### Create Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Index Repositories

Configure repositories in:

```text
rag_config.json
```

Example:

```json
{
  "repos": [
    "flask_repo",
    "werkzeug_repo"
  ]
}
```

Build graph:

```bash
python chunker.py
```

Generate embeddings:

```bash
python ingest.py
```

## Run Agent

```bash
python agent.py
```

Example questions:

* How are blueprints registered?
* How does request dispatch work?
* How is session handling implemented?
* How does template rendering work?

## Current Limitations

* Symbol resolution still misses many indirect calls.
* Retrieval occasionally surfaces tests instead of implementation code.
* Multi-hop graph traversal needs improvement.
* Repository-specific ranking is not implemented yet.

## Next Priorities

1. Improve retrieval quality.
2. Improve symbol resolution.
3. Better graph traversal.
4. Strong benchmark suite.
5. Multi-repository support beyond Flask/Werkzeug.

## Version

Current checkpoint: v0.1
