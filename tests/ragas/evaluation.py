import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
import asyncio
import ast

# Import your pipeline components
from core.models.generator import VLLMClient
from retrieval.pipeline import RetrievalPipeline
from retrieval.answer.prompt_builder import build_prompt
from configs.config import MILVUS_DATABASE


# ==========================================
# 1. DEFINE YOUR RAG INTERFACE
# ==========================================
async def my_rag_pipeline_async(query: str, database: str = None):
    """
    Connect this to your actual RAG application.
    Returns:
        answer (str): The final generated response.
        contexts (list[str]): The list of text chunks retrieved by your Hybrid Search.
    """
    # Use specified database or fall back to config
    database = database or MILVUS_DATABASE

    # Initialize pipeline components
    retrieval_pipeline = RetrievalPipeline(database=database)
    vllm_client = VLLMClient()

    # Retrieve relevant documents
    results = await retrieval_pipeline.retrieve(
        query, top_k=5, retrieval_k=20, rerank_method="weighted"
    )

    # Extract context chunks as list of strings
    contexts = []
    if results:
        contexts = [result["answer"] for result in results]

    # Build prompt and generate answer
    prompt = build_prompt(results, query, None)
    response = await vllm_client.generate(prompt, tools=None, tool_choice=None)

    # Extract answer text
    if isinstance(response, dict):
        answer = response.get("content", str(response))
    else:
        answer = str(response)

    return answer, contexts


def my_rag_pipeline(query: str, database: str = None):
    """
    Synchronous wrapper for the async RAG pipeline.
    Returns:
        answer (str): The final generated response.
        contexts (list[str]): The list of text chunks retrieved by your Hybrid Search.
    """
    return asyncio.run(my_rag_pipeline_async(query, database))


# ==========================================
# 2. THE EVALUATION HARNESS
# ==========================================
def run_evaluation(csv_path, output_name, database=None):
    print(f"--- Starting Evaluation for {csv_path} ---")
    if database:
        print(f"Using database: {database}")

    # Load the Golden Data
    df = pd.read_csv(csv_path)

    # Parse the 'answers' column to extract ground_truth
    # The answers column is a string representation of a dict like:
    # "{'text': array(['answer text'], dtype=object), 'answer_start': array([123], dtype=int32)}"
    def extract_ground_truth(answers_str):
        try:
            # Parse the string as a Python literal
            answers_dict = ast.literal_eval(answers_str)
            # Extract the first text answer
            if isinstance(answers_dict, dict) and "text" in answers_dict:
                text_array = answers_dict["text"]
                if hasattr(text_array, "__iter__") and not isinstance(text_array, str):
                    return str(list(text_array)[0]) if len(text_array) > 0 else ""
                return str(text_array)
            return str(answers_dict)
        except:
            # Fallback: try to extract text between quotes
            import re

            match = re.search(r"'text':\s*array\(\['([^']+)'\]", answers_str)
            if match:
                return match.group(1)
            return ""

    df["ground_truth"] = df["answers"].apply(extract_ground_truth)

    # Prepare lists to store your RAG's output
    generated_answers = []
    retrieved_contexts = []

    # Loop through questions and query your RAG
    print(f"Processing {len(df)} queries...")
    for index, row in df.iterrows():
        question = row["question"]

        try:
            # CALL YOUR SYSTEM
            ans, ctx = my_rag_pipeline(question, database=database)

            generated_answers.append(ans)
            retrieved_contexts.append(ctx)

            # Simple progress log
            if index % 10 == 0:
                print(f"Processed {index}/{len(df)}")

        except Exception as e:
            print(f"Error on row {index}: {e}")
            generated_answers.append("Error")
            retrieved_contexts.append([])

    # Add results back to the dataframe
    df["answer"] = generated_answers
    df["contexts"] = retrieved_contexts  # RAGAS needs this column name strictly

    # Ensure we have all required columns: question, answer, contexts, ground_truth
    print(f"Dataset preview:\n{df[['question', 'answer', 'ground_truth']].head(2)}")

    # Convert to HuggingFace Dataset
    ragas_dataset = Dataset.from_pandas(
        df[["question", "answer", "contexts", "ground_truth"]]
    )

    # Run Metrics
    print("Calculating RAGAS metrics...")
    results = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,  # Quality of Reranker
            context_recall,  # Quality of Hybrid Search
            faithfulness,  # Quality of LLM adherence to context
            answer_relevancy,  # Quality of Agent/Answer
        ],
    )

    print(f"Scores for {output_name}: {results}")

    # Save detailed breakdown (essential for debugging)
    results_df = results.to_pandas()
    results_df.to_csv(f"{output_name}_detailed_results.csv", index=False)
    print(f"Saved details to {output_name}_detailed_results.csv")


# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Run evaluation on train.csv dataset (SQuAD format)
    # Ensure your Milvus database is populated before running this!
    # CSV columns: 'id', 'title', 'context', 'question', 'answers'
    # The script will automatically extract ground_truth from the 'answers' column
    run_evaluation(
        "tests/RAGAS/dataset/train.csv", "train_evaluation_results", database=None
    )
