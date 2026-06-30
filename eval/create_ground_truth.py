import json
import re
from pathlib import Path

BENCHMARK_FILE = Path("benchmarks_new.txt")
OUTPUT_FILE = Path("eval/ground_truth.json")

text = BENCHMARK_FILE.read_text(encoding="utf-8")

# Split into benchmark sections
sections = re.split(
    r"={20,}\nQUESTION:\s*",
    text
)

ground_truth = []

for idx, section in enumerate(sections[1:], start=1):

    lines = section.splitlines()

    question = lines[0].strip()

    # -----------------------------
    # Extract FILES INVOLVED section
    # -----------------------------
    files = []

    m = re.search(
        r"FILES INVOLVED(.*?)(KEY RELATIONSHIPS|$)",
        section,
        re.DOTALL,
    )

    if m:
        block = m.group(1)

        files = sorted(set(
            re.findall(
                r"(?:flask_repo|werkzeug_repo)[^\s:`]*",
                block
            )
        ))

    # -----------------------------
    # Guess query type
    # -----------------------------
    if "Where is" in question:
        query_type = "lookup"
    elif "breaks if" in question:
        query_type = "impact"
    else:
        query_type = "flow"

    functions = set()

    m = re.search(
        r"KEY RELATIONSHIPS(.*?)(={20,}|$)",
        section,
        re.DOTALL,
    )

    if m:
        rel = m.group(1)

        matches = re.findall(r"`([^`]+)`", rel)

        for fn in matches:
            if fn.isidentifier():
                functions.add(fn)

    summary = ""

    m = re.search(
        r"FLOW SUMMARY(.*?)(STEP-BY-STEP TRACE|$)",
        section,
        re.DOTALL,
    )

    if m:
        summary = " ".join(
            line.strip()
            for line in m.group(1).splitlines()
            if line.strip()
        )

    ground_truth.append({
        "id": idx,
        "question": question,
        "query_type": query_type,
        "expected_files": files,
        "expected_functions": sorted(functions),
        "reference_answer": summary,
        "notes": ""
    })

OUTPUT_FILE.parent.mkdir(exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Created {OUTPUT_FILE}")