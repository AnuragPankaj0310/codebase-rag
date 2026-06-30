import json
from pathlib import Path

DATASET = Path("eval/dataset.json")

dataset = json.loads(DATASET.read_text(encoding="utf-8"))

# ----------------------------
# Running totals
# ----------------------------

total_raw_file_precision = 0
total_raw_file_recall = 0
total_raw_file_f1 = 0

total_context_file_precision = 0
total_context_file_recall = 0
total_context_file_f1 = 0

total_fn_precision = 0
total_fn_recall = 0
total_fn_f1 = 0

total_hit1 = 0
total_hit3 = 0
total_hit5 = 0

total_function_hit1 = 0
total_function_hit3 = 0
total_function_hit5 = 0

count = 0

results = []


# ----------------------------
# Metrics
# ----------------------------

def precision(predicted, expected):
    predicted = set(predicted)
    expected = set(expected)

    if not predicted:
        return 0

    return len(predicted & expected) / len(predicted)


def recall(predicted, expected):
    predicted = set(predicted)
    expected = set(expected)

    if not expected:
        return 1

    return len(predicted & expected) / len(expected)


def f1(p, r):
    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)


def hit_at_k(retrieval_results, expected_files, k):
    retrieved = {
        chunk["file_path"]
        for chunk in retrieval_results[:k]
    }

    return int(
        len(retrieved & set(expected_files)) > 0
    )

def function_hit_at_k(retrieval_results, expected_functions, k):
    top_k = retrieval_results[:k]

    retrieved = {
        r["name"]
        for r in top_k
        if r.get("type") == "function"
    }

    return int(
        len(retrieved & set(expected_functions)) > 0
    )


print("=" * 80)

# ----------------------------
# Evaluate every benchmark
# ----------------------------

for sample in dataset:

    gt = sample["ground_truth"]

    if gt.get("skip_deterministic_eval"):
        continue

    expected_files = gt["expected_files"]
    expected_functions = gt["expected_functions"]

    # ----------------------------
    # RAW RETRIEVAL
    # ----------------------------

    raw_files = list({
        chunk["file_path"]
        for chunk in sample["retrieval_results"]
    })

    raw_file_p = precision(raw_files, expected_files)
    raw_file_r = recall(raw_files, expected_files)
    raw_file_f1 = f1(raw_file_p, raw_file_r)

    # ----------------------------
    # FINAL CONTEXT
    # ----------------------------

    context_files = sample["allowed_files"]
    context_functions = sample["allowed_functions"]

    context_file_p = precision(
        context_files,
        expected_files
    )

    context_file_r = recall(
        context_files,
        expected_files
    )

    context_file_f1 = f1(
        context_file_p,
        context_file_r
    )

    fn_p = precision(
        context_functions,
        expected_functions
    )

    fn_r = recall(
        context_functions,
        expected_functions
    )

    fn_f1 = f1(fn_p, fn_r)

    # ----------------------------
    # Hit@K
    # ----------------------------

    hit1 = hit_at_k(
        sample["retrieval_results"],
        expected_files,
        1,
    )

    hit3 = hit_at_k(
        sample["retrieval_results"],
        expected_files,
        3,
    )

    hit5 = hit_at_k(
        sample["retrieval_results"],
        expected_files,
        5,
    )

    function_hit1 = function_hit_at_k(
        sample["retrieval_results"],
        expected_functions,
        1
    )

    function_hit3 = function_hit_at_k(
        sample["retrieval_results"],
        expected_functions,
        3
    )

    function_hit5 = function_hit_at_k(
        sample["retrieval_results"],
        expected_functions,
        5
    )

    # ----------------------------
    # Save
    # ----------------------------

    results.append({
        "id": sample["id"],
        "question": sample["question"],

        "raw_file_precision": round(raw_file_p, 3),
        "raw_file_recall": round(raw_file_r, 3),
        "raw_file_f1": round(raw_file_f1, 3),

        "context_file_precision": round(context_file_p, 3),
        "context_file_recall": round(context_file_r, 3),
        "context_file_f1": round(context_file_f1, 3),

        "function_precision": round(fn_p, 3),
        "function_recall": round(fn_r, 3),
        "function_f1": round(fn_f1, 3),

        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,

        "function_hit@1": function_hit1,
        "function_hit@3": function_hit3,
        "function_hit@5": function_hit5,
    })

    # ----------------------------
    # Totals
    # ----------------------------

    total_raw_file_precision += raw_file_p
    total_raw_file_recall += raw_file_r
    total_raw_file_f1 += raw_file_f1

    total_context_file_precision += context_file_p
    total_context_file_recall += context_file_r
    total_context_file_f1 += context_file_f1

    total_fn_precision += fn_p
    total_fn_recall += fn_r
    total_fn_f1 += fn_f1

    total_hit1 += hit1
    total_hit3 += hit3
    total_hit5 += hit5

    total_function_hit1 += function_hit1
    total_function_hit3 += function_hit3
    total_function_hit5 += function_hit5

    count += 1

    # ----------------------------
    # Console output
    # ----------------------------

    print(f"\nQuestion {sample['id']}")
    print(sample["question"])

    print("\nRAW RETRIEVAL")
    print(f"Precision : {raw_file_p:.2f}")
    print(f"Recall    : {raw_file_r:.2f}")
    print(f"F1        : {raw_file_f1:.2f}")

    print("\nFINAL CONTEXT")
    print(f"Precision : {context_file_p:.2f}")
    print(f"Recall    : {context_file_r:.2f}")
    print(f"F1        : {context_file_f1:.2f}")

    print("\nCONTEXT FUNCTIONS")
    print(f"Precision : {fn_p:.2f}")
    print(f"Recall    : {fn_r:.2f}")
    print(f"F1        : {fn_f1:.2f}")

    print("\nFILE HIT@K")
    print(f"Hit@1 : {hit1}")
    print(f"Hit@3 : {hit3}")
    print(f"Hit@5 : {hit5}")

    print("\nFUNCTION HIT@K")
    print(f"Hit@1 : {function_hit1}")
    print(f"Hit@3 : {function_hit3}")
    print(f"Hit@5 : {function_hit5}")

# ----------------------------
# Summary
# ----------------------------

print("\n" + "=" * 80)
print("AVERAGES")

print("\nRAW RETRIEVAL")
print(f"Precision : {total_raw_file_precision / count:.3f}")
print(f"Recall    : {total_raw_file_recall / count:.3f}")
print(f"F1        : {total_raw_file_f1 / count:.3f}")

print("\nFINAL CONTEXT")
print(f"Precision : {total_context_file_precision / count:.3f}")
print(f"Recall    : {total_context_file_recall / count:.3f}")
print(f"F1        : {total_context_file_f1 / count:.3f}")

print("\n CONTEXT FUNCTIONS")
print(f"Precision : {total_fn_precision / count:.3f}")
print(f"Recall    : {total_fn_recall / count:.3f}")
print(f"F1        : {total_fn_f1 / count:.3f}")

print("\nFILE HIT@K")
print(f"Hit@1 : {total_hit1 / count:.3f}")
print(f"Hit@3 : {total_hit3 / count:.3f}")
print(f"Hit@5 : {total_hit5 / count:.3f}")

print("\nFUNCTION HIT@K")
print(f"Hit@1 : {total_function_hit1/count:.3f}")
print(f"Hit@3 : {total_function_hit3/count:.3f}")
print(f"Hit@5 : {total_function_hit5/count:.3f}")

Path("eval/evaluation_results.json").write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("\nSaved eval/evaluation_results.json")
print("=" * 80)