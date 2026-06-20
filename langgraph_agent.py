from langgraph.graph import StateGraph, END
from agent import build_graph_context
from reranker import hybrid_search_with_rerank
from typing import TypedDict, List


class AgentState(TypedDict):
    question: str
    query_type: str
    retrieval_results: List
    graph_context: str
    answer: str
    confidence: float
    retry_count: int


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

    if not results:
        return {
            **state,
            "graph_context": "",
            "confidence": 0.0
        }

    context = build_graph_context(results)

    confidence = 0.8

    if len(context.strip()) < 100:
        confidence = 0.3

    return {
        **state,
        "graph_context": context,
        "confidence": confidence
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

workflow.add_edge(
    "retriever",
    "graph_agent"
)

workflow.add_edge(
    "graph_agent",
    END
)

workflow.add_node(
    "graph_agent",
    graph_agent_node
)

app = workflow.compile()


if __name__ == "__main__":

    result = app.invoke({
        "question": "How are blueprints registered?",
        "query_type": "",
        "retrieval_results": [],
        "graph_context": "",
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0
    })
print(result["query_type"])
print()

print("Retrieved:", len(result["retrieval_results"]))
print()

print(result["graph_context"][:1000])