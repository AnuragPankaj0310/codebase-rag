
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

def parse_bullet_list(text):
    if not text:
        return []

    return [
        line.replace("-", "").strip()
        for line in text.splitlines()
        if line.strip()
    ]

from langgraph_agent import app

GROUND_TRUTH = Path("eval/ground_truth.json")
QUESTIONS = Path("benchmark_questions.txt")
OUTPUT = Path("eval/dataset.json")

ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

gt_lookup = {
    item["question"]: item
    for item in ground_truth
}

dataset = []

questions = [
    q.strip()
    for q in QUESTIONS.read_text(encoding="utf-8").splitlines()
    if q.strip()
]

for i, question in enumerate(questions, start=1):

    print(f"[{i}] {question}")

    state = app.invoke({
        "question": question,
        "retry_count": 0
    })

    dataset.append({
        "id": i,
        "question": question,

        "generated_answer": state["answer"],

        "query_type": state["query_type"],

        "retrieval_results": state["retrieval_results"],

        "allowed_files": parse_bullet_list(
            state.get("file_list", "")
        ),

        "allowed_functions": parse_bullet_list(
            state.get("fn_list", "")
        ),

        "ground_truth": gt_lookup[question]
    })

OUTPUT.parent.mkdir(exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nSaved {len(dataset)} examples to {OUTPUT}")