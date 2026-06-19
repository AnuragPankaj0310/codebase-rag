# Codebase RAG

Repository-aware Retrieval-Augmented Generation (RAG) system for understanding large codebases.

## Features

* AST-based code chunking
* Call graph generation
* Semantic search with embeddings
* Cross-encoder reranking
* Qdrant vector storage
* Multi-repository indexing
* Graph-aware code retrieval
* LLM-powered repository Q&A

## Architecture

```text
Repository
    ↓
Chunker
    ↓
Call Graph
    ↓
Embeddings
    ↓
Qdrant
    ↓
Retrieval
    ↓
Reranker
    ↓
LLM Answer Generation
```

## Setup

### Clone

```bash
git clone https://github.com/AnuragPankaj0310/codebase-rag.git
cd codebase-rag
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Repositories

Example:

```json
{
  "repos": [
    "flask_repo",
    "werkzeug_repo"
  ]
}
```

Clone repositories:

```bash
git clone https://github.com/pallets/flask.git flask_repo
git clone https://github.com/pallets/werkzeug.git werkzeug_repo
```

## Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Build Graph

```bash
python chunker.py
```

## Generate Embeddings

```bash
python ingest.py
```

## Query Repository

```bash
python agent.py
```

Example questions:

* How are blueprints registered?
* How does request dispatch work?
* How is session handling implemented?
* How does Flask create an application?

## Project Structure

```text
agent.py
chunker.py
storage.py
reranker.py
search.py
hybrid_search.py
graph_traversal.py
ingest.py
rag_config.json
```

## Roadmap

* Better symbol resolution
* Improved hybrid retrieval
* Repository-specific ranking
* Multi-hop graph traversal
* Benchmark suite
* UI frontend

## License

MIT
