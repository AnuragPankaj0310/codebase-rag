import json
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

import os
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

llm = LangchainLLMWrapper(
    ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
)

embeddings = LangchainEmbeddingsWrapper(
    MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
)

DATASET_PATH = "eval/dataset.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

ragas_rows = []

for sample in data:

    contexts = [
        chunk["content"]
        for chunk in sample["retrieval_results"]
    ]

    ragas_rows.append(
        {
            "question": sample["question"],
            "answer": sample["generated_answer"],
            "contexts": contexts,
            "ground_truth": sample["ground_truth"]["reference_answer"],
        }
    )

dataset = Dataset.from_list(ragas_rows)

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings,
)

print(result)

result_df = result.to_pandas()

result_df.to_json(
    "eval/ragas_results.json",
    orient="records",
    indent=4,
)

print("\nSaved eval/ragas_results.json")