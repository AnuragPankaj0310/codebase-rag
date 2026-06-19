"""
Graph traversal for impact analysis.
Given a function, find everything that depends on it.
"""

import json
from collections import defaultdict, deque

# =====================================
# LOAD GRAPH
# =====================================

def load_graph():
    """Load chunks and edges from JSON."""
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    with open("edges.json", "r", encoding="utf-8") as f:
        edges = json.load(f)
    
    return chunks, edges

# =====================================
# BUILD ADJACENCY LISTS
# =====================================

def build_graph_maps(chunks, edges):
    """
    Build forward and reverse adjacency lists.
    
    Forward: A -> B means "A calls B" or "A contains B"
    Reverse: B <- A means "A depends on B"
    """
    
    # Map chunk names to full chunk data
    chunk_map = {chunk["id"]: chunk for chunk in chunks}

    name_to_ids = defaultdict(list)

    for chunk in chunks:
        name_to_ids[chunk["name"]].append(
            chunk["id"]
        )
    
    # Forward edges: who does X call/contain?
    forward = defaultdict(list)
    
    # Reverse edges: who calls/contains X?
    reverse = defaultdict(list)
    
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        edge_type = edge["type"]
        
        forward[source].append({
            "target": target,
            "type": edge_type
        })
        
        reverse[target].append({
            "source": source,
            "type": edge_type
        })
    
    return (
    chunk_map,
    name_to_ids,
    forward,
    reverse
)

# =====================================
# FIND CALLERS (What calls this?)
# =====================================

def find_callers(
    function_name,
    reverse_edges,
    max_depth=3
):
    """
    Find all functions that call this function (upstream).
    BFS traversal following reverse edges.
    """
    
    visited = set()
    queue = deque([(function_name, 0)])  # (name, depth)
    results = []
    
    while queue:
        current, depth = queue.popleft()
        
        if current in visited or depth > max_depth:
            continue
        
        visited.add(current)
        
        # Find who calls current
        callers = reverse_edges.get(current, [])
        
        for caller_info in callers:
            if caller_info["type"] == "calls":
                caller = caller_info["source"]
                
                results.append({
                    "name": caller,
                    "type": caller_info["type"],
                    "depth": depth + 1
                })
                
                if depth + 1 <= max_depth:
                    queue.append((caller, depth + 1))
    
    return results

# =====================================
# FIND CALLEES (What does this call?)
# =====================================

def find_callees(
    function_name,
    forward_edges,
    max_depth=3
):
    """
    Find all functions this calls (downstream).
    BFS traversal following forward edges.
    """
    
    visited = set()
    queue = deque([(function_name, 0)])  # (name, depth)
    results = []
    
    while queue:
        current, depth = queue.popleft()
        
        if current in visited or depth > max_depth:
            continue
        
        visited.add(current)
        
        # Find who current calls
        callees = forward_edges.get(current, [])
        
        for callee_info in callees:
            if callee_info["type"] == "calls":
                callee = callee_info["target"]
                
                results.append({
                    "name": callee,
                    "type": callee_info["type"],
                    "depth": depth + 1
                })
                
                if depth + 1 <= max_depth:
                    queue.append((callee, depth + 1))
    
    return results

# =====================================
# IMPACT ANALYSIS
# =====================================

def analyze_impact(
    function_name,
    chunks,
    forward_edges,
    reverse_edges
):
    """
    Comprehensive impact analysis.
    What happens if we modify this function?
    """
    
    # Find the function
    matches = [
    c
    for c in chunks
    if c["name"] == function_name
]

    if not matches:
        return None

    print(
        f"\nFound "
        f"{len(matches)} matches "
        f"for '{function_name}'"
    )

    for idx, match in enumerate(
        matches,
        start=1
    ):
        print(
            f"[{idx}] "
            f"{match['file_path']}"
        )

    target = matches[0]
    
    # Find callers (what breaks if we change this?)
    callers = find_callers(
        function_name,
        reverse_edges,
        max_depth=3
    )
    
    # Find callees (what must this work with?)
    callees = find_callees(
        function_name,
        forward_edges,
        max_depth=3
    )
    
    return {
        "target": {
            "name": target["name"],
            "file": target["file_path"],
            "lines": f"{target['start_line']}-{target['end_line']}",
            "type": target["type"],
            "docstring": target.get("docstring", "")
        },
        "callers": callers,
        "callees": callees,
        "risk_level": "HIGH" if len(callers) > 5 else "MEDIUM" if len(callers) > 0 else "LOW"
    }


# =====================================
# IMPACT ANALYSIS BY CHUNK ID
# =====================================

def analyze_impact_by_id(
    chunk_id,
    chunks,
    forward_edges,
    reverse_edges
):
    """
    Analyze impact using a unique chunk id
    instead of a function name.
    """

    target = next(
        (
            c
            for c in chunks
            if c["id"] == chunk_id
        ),
        None
    )

    if not target:
        return None

    callers = find_callers(
        chunk_id,
        reverse_edges,
        max_depth=3
    )

    callees = find_callees(
        chunk_id,
        forward_edges,
        max_depth=3
    )

    return {
        "target": {
            "name": target["name"],
            "file": target["file_path"],
            "lines": (
                f"{target['start_line']}-"
                f"{target['end_line']}"
            ),
            "type": target["type"],
            "docstring": target.get(
                "docstring",
                ""
            )
        },
        "callers": callers,
        "callees": callees,
        "risk_level":
            "HIGH"
            if len(callers) > 5
            else "MEDIUM"
            if len(callers) > 0
            else "LOW"
    }


# =====================================
# DISPLAY IMPACT
# =====================================

def display_impact(impact):
    """Pretty print impact analysis."""
    
    if not impact:
        print("Function not found in graph.")
        return
    
    target = impact["target"]
    callers = impact["callers"]
    callees = impact["callees"]
    risk = impact["risk_level"]
    
    print(f"\n{'='*70}")
    print(f"IMPACT ANALYSIS: {target['name']}")
    print(f"{'='*70}")
    
    print(f"\nTarget Function:")
    print(f"  Name: {target['name']}")
    print(f"  Type: {target['type']}")
    print(f"  File: {target['file']}")
    print(f"  Lines: {target['lines']}")
    if target['docstring']:
        print(f"  Doc: {target['docstring'][:100]}...")
    
    print(f"\n Risk Level: {risk}")
    
    print(f"\n UPSTREAM ({len(callers)} functions call this):")
    if callers:
        for depth in range(1, 4):
            depth_callers = [c for c in callers if c["depth"] == depth]
            if depth_callers:
                indent = "  " * depth
                print(f"{indent}Depth {depth}:")
                for caller in depth_callers[:5]:  # Limit display
                    print(f"{indent}  → {caller['name']}")
                if len(depth_callers) > 5:
                    print(f"{indent}  ... and {len(depth_callers) - 5} more")
    else:
        print("  (No callers - this is a leaf function)")
    
    print(f"\n DOWNSTREAM ({len(callees)} functions this calls):")
    if callees:
        for depth in range(1, 4):
            depth_callees = [c for c in callees if c["depth"] == depth]
            if depth_callees:
                indent = "  " * depth
                print(f"{indent}Depth {depth}:")
                for callee in depth_callees[:5]:  # Limit display
                    print(f"{indent}  → {callee['name']}")
                if len(depth_callees) > 5:
                    print(f"{indent}  ... and {len(depth_callees) - 5} more")
    else:
        print("  (No callees - this doesn't call anything)")
    
    print(f"\n{'='*70}\n")

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    
    print("Loading graph...")
    chunks, edges = load_graph()
    chunk_map, name_to_ids, forward, reverse = (
        build_graph_maps(
            chunks,
            edges
        )
    )
    
    print(f"✓ Loaded {len(chunks)} chunks and {len(edges)} edges\n")
    
    # Test impact analysis on several functions
    test_functions = [
        "create_app",
        "dispatch_request",
        "celery_init_app",
        "match_request"
    ]
    
    for func in test_functions:
        impact = analyze_impact(func, chunks, forward, reverse)
        display_impact(impact)