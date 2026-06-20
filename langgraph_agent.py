from langgraph.graph import StateGraph, END
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


workflow = StateGraph(AgentState)

workflow.add_node(
    "planner",
    planner_node
)

workflow.set_entry_point(
    "planner"
)

workflow.add_edge(
    "planner",
    END
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

    print(result)