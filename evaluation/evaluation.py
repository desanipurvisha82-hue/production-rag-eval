
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

from datasets import Dataset

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# =====================================================
# LOAD LOCAL OLLAMA MODEL
# =====================================================

llm = ChatOllama(
    model="phi3",
    temperature=0
)

ragas_llm = LangchainLLMWrapper(llm)

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    embedding_model
)

# =====================================================
# SAMPLE DATA
# =====================================================

data = {
    "question": [
        "What is this document about?"
    ],

    "answer": [
        "The document explains change management concepts."
    ],

    "contexts": [[
        "The PDF discusses change management models, force field analysis, and transition phases."
    ]],

    "ground_truth": [
        "The document explains change management."
    ]
}

dataset = Dataset.from_dict(data)

# =====================================================
# RUN EVALUATION
# =====================================================

result = evaluate(
    dataset=dataset,

    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ],

    llm=ragas_llm,
    embeddings=ragas_embeddings
)

# =====================================================
# PRINT RESULTS
# =====================================================

print("\nEvaluation Results:\n")

print(result)

