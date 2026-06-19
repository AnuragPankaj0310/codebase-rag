"""
chunker.py — AST-based code graph builder.

Supports:
- Multiple repo roots (configured via rag_config.json)
- Python async functions
- Phase 1: per-file symbol table (var = ClassName())
- Phase 2: cross-file symbol propagation via imports
- Return type tracking: def f() -> ClassName / return ClassName(...)
- Tests parsed for graph edges but flagged for retrieval filtering
"""

from tree_sitter import Language, Parser
import tree_sitter_python
import os
import re
import json

# =====================================
# CONFIG
# =====================================

def load_config(path="rag_config.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "repos": ["repo"],
        "noise_files": [],
        "noise_dirs": ["examples", "tutorial"],
        "boost_dirs": ["src", "lib", "core", "app"],
        "penalize_dirs": ["tests", "test", "docs", "benchmarks"],
        "rerank_score_cutoff": -3.0,
        "skip_dirs": [
            ".git", "__pycache__", ".venv", "venv",
            "node_modules", ".pytest_cache", "build", "dist"
        ]
    }

CONFIG = load_config()

# =====================================
# TREE-SITTER SETUP
# =====================================

PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser()
parser.language = PY_LANGUAGE


# =====================================
# HELPERS
# =====================================

def node_text(node, source_bytes):
    return source_bytes[
        node.start_byte:node.end_byte
    ].decode("utf-8", errors="ignore")


def extract_docstring(node, source_bytes):
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    text = node_text(stmt, source_bytes)
                    if text.startswith('"') or text.startswith("'"):
                        return text
                    break
    return ""


def is_test_path(file_path):
    return (
        "\\tests\\" in file_path
        or "/tests/" in file_path
        or "\\test\\" in file_path
        or "/test/" in file_path
    )


# =====================================
# IMPORT EXTRACTION
# =====================================

def extract_imports(root, source_bytes):
    imports = []
    for node in root.children:
        if node.type in ("import_statement", "import_from_statement"):
            imports.append(node_text(node, source_bytes))
    return imports


# =====================================
# CALL EXTRACTION
# =====================================

def extract_calls(node, source_bytes):
    calls = []

    def walk(n):
        if n.type == "call":
            func = n.child_by_field_name("function")
            if func:
                calls.append(node_text(func, source_bytes))
        for child in n.children:
            walk(child)

    walk(node)
    return list(set(calls))


# =====================================
# ASSIGNMENT EXTRACTION
# =====================================

def extract_assignments(node, source_bytes):
    assignments = {}

    def walk(n):
        if n.type == "assignment":
            if len(n.children) >= 3:
                left = n.children[0]
                right = n.children[-1]
                if left.type == "identifier" and right.type == "call":
                    func = right.child_by_field_name("function")
                    if func:
                        func_name = node_text(func, source_bytes)
                        if not func_name.split(".")[-1][:1].isupper():
                            return
                        assignments[node_text(left, source_bytes)] = func_name
        for child in n.children:
            walk(child)

    walk(node)
    return assignments


# =====================================
# DECORATOR EXTRACTION
# =====================================

def extract_decorators(node, source_bytes):
    decorators = []
    parent = node.parent
    if not parent:
        return decorators
    try:
        node_index = parent.children.index(node)
    except ValueError:
        return decorators

    for i in range(node_index - 1, -1, -1):
        prev = parent.children[i]
        if prev.type == "decorator":
            text = node_text(prev, source_bytes).lstrip("@").strip()
            decorators.append(text)
        elif prev.type not in ("newline", "indent", "dedent", "comment"):
            break

    return list(reversed(decorators))


# =====================================
# INHERITANCE EXTRACTION
# =====================================

def extract_inheritance(node, source_bytes):
    parents = []
    if node.type != "class_definition":
        return parents
    for child in node.children:
        if child.type == "argument_list":
            arg_text = node_text(child, source_bytes).strip("()")
            bases = [b.strip() for b in arg_text.split(",") if b.strip()]
            parents.extend(bases)
    return parents


# =====================================
# RETURN TYPE EXTRACTION
# =====================================

def extract_return_type(node, source_bytes):
    """
    Extract the return type of a function two ways:
    1. Type annotation: def foo() -> MyClass:
    2. Return statement: return MyClass(...) or return MyClass
    Returns the class name string, or None.
    """
    # Way 1: annotation
    return_type_node = node.child_by_field_name("return_type")
    if return_type_node:
        text = node_text(return_type_node, source_bytes).lstrip("->").strip()
        inner = re.sub(r"Optional\[(.+)\]", r"\1", text)
        inner = re.sub(r"Union\[(.+?),.+\]", r"\1", inner)
        inner = inner.strip().split("[")[0].strip()
        skip = {"None", "Any", "bool", "int", "str", "list", "dict", "tuple", "set"}
        if inner and inner[0].isupper() and inner not in skip:
            return inner

    # Way 2: scan return statements
    def find_return_class(n):
        if n.type == "return_statement":
            for child in n.children:
                if child.type == "call":
                    func = child.child_by_field_name("function")
                    if func:
                        t = node_text(func, source_bytes)
                        last = t.split(".")[-1]
                        if last and last[0].isupper():
                            return last
                elif child.type == "identifier":
                    t = node_text(child, source_bytes)
                    if t and t[0].isupper() and t not in ("None", "True", "False"):
                        return t
        for child in n.children:
            result = find_return_class(child)
            if result:
                return result
        return None

    return find_return_class(node)


# =====================================
# EDGE DEDUPLICATION
# =====================================

def deduplicate_edges(edges):
    seen = set()
    unique = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


# =====================================
# SYMBOL TABLE BUILDER  (Phase 1)
# =====================================

def build_symbol_table(tree, source_bytes, file_path):
    """
    Build a per-file symbol table: scope_key -> {var_name -> ClassName}

    Captures:
    - app = Flask(...)          -> {file: {app: Flask}}
    - self.db = Database()      -> {scope: {self.db: Database}}
    - def create_app() -> Flask -> {file: {create_app: Flask}}
    - def f(): return Flask()   -> {file: {f: Flask}}
    """
    symbol_table = {}
    return_types = {}

    def collect_return_types(node):

        if node.type in (
            "function_definition",
            "async_function_definition"
        ):

            name_node = node.child_by_field_name(
                "name"
            )

            if name_node:

                func_name = node_text(
                    name_node,
                    source_bytes
                )

                ret_class = extract_return_type(
                    node,
                    source_bytes
                )

                if ret_class:

                    return_types[
                        func_name
                    ] = ret_class

        for child in node.children:
            collect_return_types(child)

    def get_scope_key(node):
        parent = node.parent
        scope_parts = []
        while parent:
            if parent.type in (
                "function_definition",
                "class_definition",
                "async_function_definition"
            ):
                name_node = parent.child_by_field_name("name")
                if name_node:
                    scope_parts.append(node_text(name_node, source_bytes))
            parent = parent.parent
        if scope_parts:
            return f"{file_path}:{':'.join(reversed(scope_parts))}"
        return file_path

    def extract_class_name(call_node):
        func = call_node.child_by_field_name("function")
        if not func:
            return None
        func_text = node_text(func, source_bytes)
        if func_text and func_text[0].isupper():
            return func_text
        if "." in func_text:
            last = func_text.split(".")[-1]
            if last and last[0].isupper():
                return last
        return None
    
    def walk(node):
        # Variable assignment: x = ClassName(...)
        if node.type == "assignment":
            children = node.children
            if len(children) >= 3:
                left = children[0]
                right = children[-1]
                if right.type == "call":
                    class_name = extract_class_name(right)

                    if not class_name:
                        func = right.child_by_field_name("function")

                        if func:

                            func_name = node_text(
                                func,
                                source_bytes
                            )

                            class_name = return_types.get(
                                func_name
                            )
                    if class_name:
                        scope_key = get_scope_key(node)
                        symbol_table.setdefault(scope_key, {})
                        if left.type == "identifier":
                            symbol_table[scope_key][node_text(left, source_bytes)] = class_name
                        elif left.type == "attribute":
                            symbol_table[scope_key][node_text(left, source_bytes)] = class_name

        # Function: track return type as {func_name: ReturnClass}
        if node.type in ("function_definition", "async_function_definition"):
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = node_text(name_node, source_bytes)
                ret_class = extract_return_type(node, source_bytes)
                if ret_class:
                    symbol_table.setdefault(file_path, {})
                    symbol_table[file_path][func_name] = ret_class

        for child in node.children:
            walk(child)

    
    collect_return_types(
        tree.root_node
)

    walk(
        tree.root_node
    )

    return symbol_table


# =====================================
# OBJECT.METHOD RESOLVER
# =====================================

def resolve_call_generic(
    call_text,
    scope_key,
    file_path,
    symbol_table,
    name_to_ids,
    chunk_map
):
    """
    Resolve "app.register_blueprint" -> chunk_id via symbol table.
    """
    if "." not in call_text:
        return None

    parts = call_text.split(".")
    obj_name = parts[0]
    method_name = parts[-1]

    class_name = None

    if scope_key in symbol_table:
        class_name = (
            symbol_table[scope_key].get(obj_name)
            or symbol_table[scope_key].get(f"self.{obj_name}")
        )

    if not class_name and file_path in symbol_table:
        class_name = symbol_table[file_path].get(obj_name)

    if not class_name:
        return None

    candidates = name_to_ids.get(method_name, [])
    if not candidates:
        return None

    matching = []
    for chunk_id in candidates:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        parent_id = chunk.get("parent")
        if not parent_id:
            continue
        parent_chunk = chunk_map.get(parent_id)
        if not parent_chunk:
            continue
        if parent_chunk["name"] == class_name:
            matching.append(chunk_id)

    if matching:
        return matching[0]

    if len(candidates) == 1:
        return candidates[0]

    return None


# =====================================
# PARSE SINGLE FILE
# =====================================

def parse_python_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception:
        return [], [], {}

    source_bytes = source_code.encode()
    tree = parser.parse(source_bytes)
    imports = extract_imports(tree.root_node, source_bytes)

    chunks = []
    edges = []

    FUNC_TYPES = (
        "function_definition",
        "async_function_definition",
        "class_definition"
    )

    def walk(node, parent=None):
        if node.type in FUNC_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = node_text(name_node, source_bytes)
                chunk_id = f"{file_path}:{name}:{node.start_point[0] + 1}"

                calls = extract_calls(node, source_bytes)
                inherits = extract_inheritance(node, source_bytes)
                assignments = extract_assignments(node, source_bytes)
                return_type = extract_return_type(node, source_bytes)

                chunk = {
                    "id": chunk_id,
                    "name": name,
                    "type": (
                        "async_function"
                        if node.type == "async_function_definition"
                        else node.type.replace("_definition", "")
                    ),
                    "is_test": is_test_path(file_path),
                    "parent": parent,
                    "decorators": extract_decorators(node, source_bytes),
                    "file_path": file_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "docstring": extract_docstring(node, source_bytes),
                    "imports": imports,
                    "calls": calls,
                    "assignments": assignments,
                    "inherits": inherits,
                    "return_type": return_type,
                    "content": node_text(node, source_bytes),
                }

                chunks.append(chunk)

                if parent:
                    edges.append({
                        "source": parent,
                        "target": chunk_id,
                        "type": "contains"
                    })

                for call in calls:
                    edges.append({
                        "source": chunk_id,
                        "target": call,
                        "type": "calls"
                    })

                for base in inherits:
                    edges.append({
                        "source": chunk_id,
                        "target": base,
                        "type": "inherits"
                    })

                current_parent = chunk_id
            else:
                current_parent = parent
        else:
            current_parent = parent

        for child in node.children:
            walk(child, current_parent)

    walk(tree.root_node)

    # Phase 1: per-file symbol resolution of dotted calls
    chunk_map_local = {c["id"]: c for c in chunks}
    name_to_ids_local = {}
    for c in chunks:
        name_to_ids_local.setdefault(c["name"], []).append(c["id"])

    symbol_table = build_symbol_table(tree, source_bytes, file_path)

    resolved = []
    for edge in edges:
        if edge["type"] != "calls" or "." not in edge["target"]:
            resolved.append(edge)
            continue

        rid = resolve_call_generic(
            edge["target"],
            edge["source"],
            file_path,
            symbol_table,
            name_to_ids_local,
            chunk_map_local
        )

        if rid:
            resolved.append({"source": edge["source"], "target": rid, "type": "calls"})
        else:
            resolved.append(edge)

    return chunks, resolved, symbol_table


# =====================================
# PARSE REPOSITORY
# =====================================

def parse_repository(repo_path):
    """
    Walk repo, parse every .py file.
    Tests are included (contribute graph edges) but flagged via is_test.
    """
    all_chunks = []
    all_edges = []
    all_symbol_tables = {}

    skip_dirs = set(CONFIG.get("skip_dirs", []))

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            chunks, edges, symbol_table = parse_python_file(path)
            all_chunks.extend(chunks)
            all_edges.extend(edges)
            all_symbol_tables[path] = symbol_table

    return all_chunks, all_edges, all_symbol_tables


# =====================================
# IMPORT MAP  (Phase 2 helper)
# =====================================

def build_import_map(chunks):
    """
    For each file, map local imported names -> original names.
    from myapp import app        -> {"app": "app"}
    from flask import Flask      -> {"Flask": "Flask"}
    from x import create_app    -> {"create_app": "create_app"}
    """
    import_map = {}
    for chunk in chunks:
        fp = chunk["file_path"]
        if fp not in import_map:
            import_map[fp] = {}
        for imp in chunk.get("imports", []):
            m = re.match(r"from\s+[\w.]+\s+import\s+(.+)", imp.strip())
            if m:
                for part in m.group(1).split(","):
                    part = part.strip()
                    if " as " in part:
                        orig, alias = part.split(" as ", 1)
                        import_map[fp][alias.strip()] = orig.strip()
                    elif part:
                        import_map[fp][part] = part
    return import_map


# =====================================
# SYMBOL PROPAGATION  (Phase 2)
# =====================================

def propagate_symbols(all_symbol_tables, import_map, chunks):
    """
    Cross-file propagation:
    - If routes.py imports `app` and app.py says {app: Flask},
      copy that into routes.py's symbol table.
    - If routes.py imports `create_app` and create_app's return type
      is Flask, copy {create_app: Flask} into routes.py.
    """
    added = 0
    global_symbols = {}
    for file_path, sym_table in all_symbol_tables.items():
        for scope, vars_dict in sym_table.items():
            for var_name, class_name in vars_dict.items():
                if var_name not in global_symbols:
                    global_symbols[var_name] = class_name

    for file_path, local_imports in import_map.items():
        all_symbol_tables.setdefault(file_path, {})
        module_scope = all_symbol_tables[file_path].setdefault(file_path, {})
        for local_name, original_name in local_imports.items():
            if local_name not in module_scope and original_name in global_symbols:
                module_scope[local_name] = global_symbols[original_name]
                added += 1

    print(f"Phase 2 propagated symbols: {added}")
    return all_symbol_tables


# =====================================
# SECOND-PASS EDGE RESOLUTION  (Phase 2)
# =====================================

def resolve_edges_pass2(edges, chunks, all_symbol_tables, name_to_ids, chunk_map):
    """
    Re-resolve unresolved `calls` edges using enriched cross-file symbol tables.
    """
    parent_name_map = {
        (c["parent"], c["name"]): c["id"] for c in chunks
    }

    resolved = []

    for edge in edges:
        if edge["type"] != "calls":
            resolved.append(edge)
            continue

        target = edge["target"]
        source_chunk = chunk_map.get(edge["source"])

        if source_chunk is None:
            resolved.append(edge)
            continue

        file_path = source_chunk["file_path"]

        # 1. self.method -> parent class
        if target.startswith("self."):
            method_name = target.split(".")[-1]
            key = (source_chunk["parent"], method_name)
            rid = parent_name_map.get(key)
            if rid:
                resolved.append({"source": edge["source"], "target": rid, "type": "calls"})
                continue

        # 2. object.method -> symbol table
        if "." in target:
            file_sym = all_symbol_tables.get(file_path, {})
            rid = resolve_call_generic(
                target, edge["source"], file_path,
                file_sym, name_to_ids, chunk_map
            )
            if rid:
                resolved.append({"source": edge["source"], "target": rid, "type": "calls"})
                continue

        # 3. Fallback: unique name
        call_name = target.split(".")[-1]
        candidates = name_to_ids.get(call_name, [])
        if len(candidates) == 1:
            resolved.append({"source": edge["source"], "target": candidates[0], "type": "calls"})
        else:
            resolved.append(edge)

    return resolved


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    repos = CONFIG.get("repos", ["repo"])

    all_chunks = []
    all_edges = []
    all_symbol_tables = {}

    for repo in repos:
        if not os.path.isdir(repo):
            print(f"WARNING: repo path '{repo}' not found, skipping.")
            continue
        print(f"Parsing {repo}...")
        chunks, edges, sym_tables = parse_repository(repo)
        all_chunks.extend(chunks)
        all_edges.extend(edges)
        all_symbol_tables.update(sym_tables)
        print(f"  -> {len(chunks)} chunks, {len(edges)} edges")

    print(f"\nTotal: {len(all_chunks)} chunks, {len(all_edges)} edges")

    # Phase 2
    print("\nPhase 2: building import map...")
    import_map = build_import_map(all_chunks)
    print("Phase 2: propagating symbols...")
    all_symbol_tables = propagate_symbols(all_symbol_tables, import_map, all_chunks)

    # Build lookup maps
    name_to_ids = {}
    for chunk in all_chunks:
        name_to_ids.setdefault(chunk["name"], []).append(chunk["id"])
    chunk_map = {c["id"]: c for c in all_chunks}

    # Second-pass resolution
    print("Phase 2: resolving edges...")
    resolved_edges = resolve_edges_pass2(
        all_edges, all_chunks, all_symbol_tables, name_to_ids, chunk_map
    )
    edges = deduplicate_edges(resolved_edges)

    # Stats
    calls_edges = [e for e in edges if e["type"] == "calls"]

    production_calls = []

    for edge in calls_edges:

        source = chunk_map.get(edge["source"])

        if not source:
            continue

        if source.get("is_test"):
            continue

        production_calls.append(edge)

    resolved_prod = [
        e for e in production_calls
        if ":" in e["target"]
    ]       
    

    resolved_calls = [e for e in calls_edges if ":" in e["target"]]
    test_chunks = [c for c in all_chunks if c["is_test"]]
    prod_chunks = [c for c in all_chunks if not c["is_test"]]

    print(f"\n{'='*45}")
    print(f"Chunks total     : {len(all_chunks)}")
    print(f"  Production     : {len(prod_chunks)}")
    print(f"  Test           : {len(test_chunks)}")
    print(f"Edges total      : {len(edges)}")
    print(f"Calls total      : {len(calls_edges)}")
    print(f"Calls resolved   : {len(resolved_calls)}")

    print(f"Production calls : {len(production_calls)}")
    print(f"Resolved prod    : {len(resolved_prod)}")
    print(
        f"Prod resolution  : "
        f"{len(resolved_prod)/max(len(production_calls),1)*100:.1f}%"
    )

    print(
        f"Resolution rate  : "
        f"{len(resolved_calls)/max(len(calls_edges),1)*100:.1f}%"
    )

    print(f"{'='*45}")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    with open("edges.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)

    print("\nSaved: chunks.json, edges.json")