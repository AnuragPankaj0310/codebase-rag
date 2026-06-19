"""
agent.py — RAG query interface over a parsed code repository.

Features:
- Hybrid retrieval (semantic + BM25) with CrossEncoder reranking
- Graph-enriched context (callers, callees, related code)
- Per-repo config via rag_config.json (noise files/dirs loaded from config)
- Allowed-list hallucination guard injected into every prompt
"""

from dotenv import load_dotenv

load_dotenv()

from groq import Groq
import os
import json
from graph_traversal import (
    load_graph,
    build_graph_maps,
    analyze_impact_by_id
)
from reranker import hybrid_search_with_rerank

# =====================================
# CONFIG
# =====================================

def load_config(path="rag_config.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "noise_files": [],
        "noise_dirs": ["examples", "tutorial"],
        "rerank_score_cutoff": -3.0
    }

CONFIG = load_config()

NOISE_FILES = CONFIG.get("noise_files", [])
NOISE_DIRS  = CONFIG.get("noise_dirs", ["examples", "tutorial"])
RERANK_SCORE_CUTOFF = float(CONFIG.get("rerank_score_cutoff", -3.0))

# =====================================
# INIT
# =====================================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading graph...")
_chunks, _edges = load_graph()
_chunk_map, _name_to_ids, _forward, _reverse = build_graph_maps(_chunks, _edges)
print(f"Graph loaded: {len(_chunks)} chunks, {len(_edges)} edges")


# =====================================
# HELPERS
# =====================================

def is_noise(file_path):
    """Return True if this file should be excluded from retrieval results."""
    fp = file_path.replace("\\", "/")
    if any(n in fp for n in NOISE_FILES):
        return True
    for d in NOISE_DIRS:
        if f"/{d}/" in fp or fp.startswith(f"{d}/"):
            return True
    return False


def is_test(file_path):
    """Return True for test files — excluded from answers but kept in graph."""
    fp = file_path.replace("\\", "/")
    return "/tests/" in fp or "/test/" in fp or "test_" in os.path.basename(fp)


# =====================================
# CONTEXT BUILDER
# =====================================

def build_graph_context(results):
    """
    Build LLM context:
    - Primary retrieved chunks (SOURCE 1-5)
    - Direct callers/callees from the graph (RELATED CODE)
    """
    context_parts = []

    for i, result in enumerate(results[:5], 1):
        chunk_id = result["id"]

        chunk_section = (
            f"\nSOURCE {i}: {result['name']} ({result['type']})\n"
            f"File: {result['file_path']} "
            f"Lines {result['start_line']}-{result['end_line']}\n"
            f"\nCode:\n{result['content'][:800]}\n"
        )

        impact = analyze_impact_by_id(chunk_id, _chunks, _forward, _reverse)

        related_chunks = []
        filtered_callees = []
        filtered_callers = []
        MAX_RELATED = 2

        if impact:
            for callee in impact.get("callees", []):
                if callee.get("depth") != 1:
                    continue
                callee_id = callee["name"]
                if ":" not in callee_id:
                    continue
                c = _chunk_map.get(callee_id)
                if not c:
                    continue
                if is_noise(c["file_path"]) or is_test(c["file_path"]):
                    continue
                related_chunks.append(c)
                filtered_callees.append(callee)
                if len(filtered_callees) >= MAX_RELATED:
                    break

            for caller in impact.get("callers", []):
                if caller.get("depth") != 1:
                    continue
                caller_id = caller["name"]
                if ":" not in caller_id:
                    continue
                c = _chunk_map.get(caller_id)
                if not c:
                    continue
                if is_noise(c["file_path"]) or is_test(c["file_path"]):
                    continue
                related_chunks.append(c)
                filtered_callers.append(caller)
                if len(filtered_callers) >= MAX_RELATED:
                    break

            callers = filtered_callers
            callees = filtered_callees

            relationships = []
            if callers:
                relationships.append(
                    f"Called by: {', '.join(c['name'] for c in callers[:5])}"
                )
            if callees:
                relationships.append(
                    f"Calls: {', '.join(c['name'] for c in callees[:5])}"
                )

            if i == 1 and callees:
                flow = [result["name"]] + [c["name"] for c in callees[:4]]
                chunk_section += f"\nExecution Flow: {' -> '.join(flow)}\n"

            if relationships:
                chunk_section += "\nRelationships:\n"
                for rel in relationships:
                    chunk_section += f"  {rel}\n"

        if related_chunks:
            chunk_section += "\nRELATED CODE:\n"
            seen = set()
            for rc in related_chunks:
                if rc["id"] in seen:
                    continue
                seen.add(rc["id"])
                chunk_section += (
                    f"\nFunction: {rc['name']} "
                    f"(direct callee/caller of {result['name']})\n"
                    f"Type: {rc['type']}\n"
                    f"File: {rc['file_path']}\n"
                    f"Lines: {rc['start_line']}-{rc['end_line']}\n"
                    f"\nCode:\n{rc['content'][:400]}\n"
                )

        context_parts.append(chunk_section)

    return "\n---\n".join(context_parts)


# =====================================
# MAIN QUERY
# =====================================

def ask_repo(question):

    # Step 1: retrieve + rerank
    results = hybrid_search_with_rerank(question, top_k=30)

    # Step 2: config-driven routing (no more hardcoded keywords)
    question_lower = question.lower()
    routing = CONFIG.get("routing_categories", {})

    boost_files = []
    boost_names = []
    filter_files = []

    for keyword, category in routing.items():
        if keyword in question_lower:
            boost_files.extend(category.get("boost_files", []))
            boost_names.extend(category.get("boost_names", []))
            filter_files.extend(category.get("filter_files", []))

    # Apply boosts
    for r in results:
        if any(f in r["file_path"] for f in boost_files):
            r["final_score"] = r.get("final_score", 0) + 2.0
        if r["name"] in boost_names:
            r["final_score"] = r.get("final_score", 0) + 4.0

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # Apply filters
    if filter_files:
        results = [
            r for r in results
            if not any(f in r["file_path"].lower() for f in filter_files)
        ]

    # Step 3: filter noise, tests, low-confidence (rest stays the same)
    results = [
        r for r in results
        if r.get("final_score", 0) > RERANK_SCORE_CUTOFF
        and not is_noise(r["file_path"])
        and not is_test(r["file_path"])
    ][:5]
    
    # ... rest of function unchanged

    if not results:
        return "No relevant code found for that question."
                

    # Step 4: build allowed-list for hallucination guard
    allowed_functions = set()
    allowed_files = set()

    for r in results:
        allowed_functions.add(r["name"])
        allowed_files.add(r["file_path"])
        # Include direct graph neighbours that will appear in RELATED CODE
        for edge in _forward.get(r["id"], []):
            if edge.get("type") != "calls":   # ← add this check
                continue
            c = _chunk_map.get(edge["target"])
            if c and not is_noise(c["file_path"]) and not is_test(c["file_path"]):
                allowed_functions.add(c["name"])
                allowed_files.add(c["file_path"])
        for edge in _reverse.get(r["id"], []):
            if edge.get("type") != "calls":   # ← add this check
                continue
            c = _chunk_map.get(edge["source"])
            if c and not is_noise(c["file_path"]) and not is_test(c["file_path"]):
                allowed_functions.add(c["name"])
                allowed_files.add(c["file_path"])

    fn_list   = "\n".join(f"   - {x}" for x in sorted(allowed_functions))
    file_list = "\n".join(f"   - {x}" for x in sorted(allowed_files))

    # Step 5: build graph-enriched context
    context = build_graph_context(results)

    # Step 6: prompt
    prompt = f"""You are a senior software engineer analyzing a Python codebase.
You have been given code chunks AND their call graph relationships.

CRITICAL RULES:
0. ALLOWED FUNCTIONS (you may ONLY mention these — nothing else):
{fn_list}

   ALLOWED FILES (you may ONLY mention these):
{file_list}

   Before writing each sentence, verify every function/file name is in the lists above.
   Any name not in the lists must NOT appear in your answer. This is a hard rule.

1. Explain the FLOW between functions, not just what each function does.
2. Use the Execution Flow, Relationships, and RELATED CODE sections to trace the flow.
3. Format your answer as an actual flow: A() calls B() which calls C().
4. Mention exact file paths and line numbers from the context.
5. When a RELATED CODE section shows actual code, use it to explain what that function does.
6. Do NOT just summarize each source independently — connect them.

Question: {question}

Repository Context (with call graph relationships):
{context}

Answer format:

FLOW SUMMARY
[One paragraph explaining the complete execution flow from entry to exit]

STEP-BY-STEP TRACE
[Numbered steps showing exact function calls and what each does]

FILES INVOLVED
[List of files and their roles]

KEY RELATIONSHIPS
[What calls what and why it matters]
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content


# =====================================
# ENTRYPOINT
# =====================================

if __name__ == "__main__":
    while True:
        question = input("\nAsk a repository question: ")
        if question.lower() in ("quit", "exit", "q"):
            break
        answer = ask_repo(question)
        print("\nANSWER\n")
        print(answer)