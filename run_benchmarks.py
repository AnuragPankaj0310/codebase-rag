from langgraph_agent import app

with open("benchmark_questions.txt", encoding="utf-8") as f:
    questions = [q.strip() for q in f if q.strip()]

with open("benchmarks_new.txt", "w", encoding="utf-8") as out:

    for i, q in enumerate(questions, 1):

        print(f"\n[{i}] {q}")

        result = app.invoke({
            "question": q,
            "retry_count": 0
        })

        out.write("=" * 80 + "\n")
        out.write(f"QUESTION: {q}\n\n")
        out.write(result["answer"])
        out.write("\n\n")

print("\nDone. Results saved to benchmarks_new.txt")