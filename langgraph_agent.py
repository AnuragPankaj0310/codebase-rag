from langgraph.graph import StateGraph, END
from agent import build_graph_context
from reranker import hybrid_search_with_rerank
from typing import TypedDict, List
from agent import generate_from_context
from agent import build_graph_context, generate_from_context, build_allowed_lists


class AgentState(TypedDict):
    question: str
    query_type: str
    retrieval_results: List
    graph_context: str
    answer: str
    confidence: float
    retry_count: int
    fn_list: str
    file_list: str


def planner_node(state):
    q = state["question"].lower()

    if any(x in q for x in [
        "what calls",
        "who calls",
        "callers of"
    ]):
        query_type = "impact"

    elif any(x in q for x in [
        "what breaks",
        "if i change",
        "modify"
    ]):
        query_type = "impact"

    elif any(x in q for x in [
        "where is",
        "find",
        "locate"
    ]):
        query_type = "lookup"

    else:
        query_type = "flow"

    return {
        **state,
        "query_type": query_type
    }


def retriever_node(state):

    top_k = 5

    if state["query_type"] == "flow":
        top_k = 10

    results = hybrid_search_with_rerank(
        state["question"],
        top_k=top_k
    )

    return {
        **state,
        "retrieval_results": results
    }


def graph_agent_node(state):

    results = state["retrieval_results"]
    allowed_functions = set()
    allowed_files = set()

    for r in results:
        allowed_functions.add(r["name"])
        allowed_files.add(r["file_path"])

    fn_list = "\n".join(
        f"   - {x}" for x in sorted(allowed_functions)
    )

    file_list = "\n".join(
        f"   - {x}" for x in sorted(allowed_files)
    )

    if not results:
        return {
            **state,
            "graph_context": "",
            "confidence": 0.0
        }

    print("\nQUERY TYPE:", state["query_type"])

    if state["query_type"] == "lookup":

        context = ""

        for r in results[:5]:
            context += f"\n\nSOURCE: {r['name']}\n"
            context += r["content"][:1000]

    elif state["query_type"] == "impact":

        context = "IMPACT ANALYSIS\n\n"
        context += build_graph_context(results)

    else:
        context = build_graph_context(results)

    confidence = 0.8

    if len(context.strip()) < 100:
        confidence = 0.3

    return {
    **state,
    "graph_context": context,
    "confidence": confidence,
    "fn_list": fn_list,
    "file_list": file_list
}


def impact_node(state):

    print("\nUSING IMPACT NODE")

    results = state["retrieval_results"]

    allowed_functions = set()
    allowed_files = set()

    for r in results:
        allowed_functions.add(r["name"])
        allowed_files.add(r["file_path"])

    fn_list = "\n".join(
        f"   - {x}" for x in sorted(allowed_functions)
    )

    file_list = "\n".join(
        f"   - {x}" for x in sorted(allowed_files)
    )

    context = "IMPACT ANALYSIS\n\n"

    for r in results[:3]:

        context += f"""
    TARGET FUNCTION:
    {r['name']}

    FILE:
    {r['file_path']}

    IF THIS FUNCTION CHANGES:
    Analyze:
    - Direct callers
    - Direct callees
    - What depends on it
    - What would break

    """

        context += build_graph_context([r])

    return {
    **state,
    "graph_context": context,
    "confidence": 0.8,
    "fn_list": fn_list,
    "file_list": file_list
}


def route_after_retrieval(state):

    if state["query_type"] == "impact":
        return "impact"

    return "graph"



def generator_node(state):

    answer = generate_from_context(
        state["question"],
        state["graph_context"],
        state["fn_list"],
        state["file_list"]
    )

    return {
        **state,
        "answer": answer
    }

workflow = StateGraph(AgentState)

workflow.add_node(
    "planner",
    planner_node
)

workflow.set_entry_point(
    "planner"
)

workflow.add_node(
    "retriever",
    retriever_node
)

workflow.add_edge(
    "planner",
    "retriever"
)

# workflow.add_edge(
#     "retriever",
#     "graph_agent"
# )

workflow.add_conditional_edges(
    "retriever",
    route_after_retrieval,
    {
        "graph": "graph_agent",
        "impact": "impact_node"
    }
)

workflow.add_edge(
    "graph_agent",
    "generator"
)

workflow.add_edge(
    "impact_node",
    "generator"
)

workflow.add_edge(
    "generator",
    END
)

workflow.add_node(
    "graph_agent",
    graph_agent_node
)

workflow.add_node(
    "impact_node",
    impact_node
)

workflow.add_node(
    "generator",
    generator_node
)

app = workflow.compile()


if __name__ == "__main__":

    result = app.invoke({
        "question": "What breaks if I modify register_blueprint?",
        "query_type": "",
        "retrieval_results": [],
        "graph_context": "",
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0,
        "fn_list": "",
        "file_list": ""
    })

    
    print()
    print(result["answer"])