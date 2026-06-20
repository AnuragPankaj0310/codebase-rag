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


if __name__ == "__main__":

    tests = [
        "How are blueprints registered?",
        "What calls dispatch_request?",
        "Where is session saving implemented?",
        "What breaks if I modify register_blueprint?"
    ]

    for t in tests:
        result = planner_node({
            "question": t,
            "query_type": "",
            "retrieval_results": [],
            "graph_context": "",
            "answer": "",
            "confidence": 0.0,
            "retry_count": 0
        })

        print(t)
        print("->", result["query_type"])
        print()